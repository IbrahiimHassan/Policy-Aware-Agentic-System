"""
Run ONLY the validation ablation with cached interpreter outputs.
Updates the ablation chart with clean, isolated results.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score

from engine.memory_agent import apply_memory_to_matching_policies
from engine.policy_engine import can_make_decision, policy_checker
from interpreter.llm_interpreter import llm_interpreter
from nodes.decision_node import decision_node
from nodes.validation_node import validation_node
from run_system import _normalize_out_decision
from utils.config import POLICY_JSON_PATH

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CSV_ABLATION_FULL = RESULTS_DIR / "ablation_full.csv"
PNG_ABLATION_FULL = FIGURES_DIR / "ablation_full.png"


def _adapt_case(row):
    return {
        "id": row.get("case_id") or row.get("id"),
        "input_text": row.get("input_text") or row.get("input") or row.get("request"),
        "expected_decision": row.get("expected_decision") or row.get("decision"),
        "_memory_store_patch": row.get("_memory_store_patch") or row.get("memory_store_patch"),
    }


def _load_cases():
    from book_benchmark_cases import CASES
    return [_adapt_case(c) for c in CASES]


def _compute_metrics(y_true, y_pred):
    labels = ["APPROVE", "REJECT", "ESCALATE", "NEED_MORE_INFO"]
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true) * 100
    macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": macro}


def run_with_validation(case_data, memory_patch):
    """Full pipeline with validation."""
    val = validation_node({"case_data": case_data})
    errors = [] if val.get("is_valid", True) else val.get("issues", [])

    policy_result = policy_checker(case_data, POLICY_JSON_PATH)
    matching = policy_result.get("matching_policies", [])
    matching, _, hints = apply_memory_to_matching_policies(case_data, matching, store_patch=memory_patch)

    if errors:
        decision = "REJECT"
    elif not matching and not can_make_decision(case_data, POLICY_JSON_PATH):
        decision = "NEED_MORE_INFO"
    else:
        decision = decision_node({"validation_errors": errors, "matching_policies": matching}).get("decision", "APPROVED")

    return _normalize_out_decision(decision)


def run_without_validation(case_data, memory_patch):
    """Pipeline skipping validation node."""
    errors = []

    policy_result = policy_checker(case_data, POLICY_JSON_PATH)
    matching = policy_result.get("matching_policies", [])
    matching, _, hints = apply_memory_to_matching_policies(case_data, matching, store_patch=memory_patch)

    if errors:
        decision = "REJECT"
    elif not matching and not can_make_decision(case_data, POLICY_JSON_PATH):
        decision = "NEED_MORE_INFO"
    else:
        decision = decision_node({"validation_errors": errors, "matching_policies": matching}).get("decision", "APPROVED")

    return _normalize_out_decision(decision)


def main():
    if not CSV_ABLATION_FULL.exists():
        print(f"[ERROR] {CSV_ABLATION_FULL} not found. Run ablation_full.py first.")
        return

    cases = _load_cases()

    # Phase 1: cache interpreter outputs once
    print("[Phase 1] Running interpreter once and caching...")
    cached = {}
    for case in cases:
        cached[case["id"]] = llm_interpreter(case["input_text"]).get("case_data", {})
    print(f"  Cached {len(cached)} outputs.")

    # Phase 2: run both variants with identical inputs
    print("\n[Phase 2] Running Full Pipeline (cached)...")
    y_true_full, y_pred_full = [], []
    for case in cases:
        y_true_full.append(case["expected_decision"])
        y_pred_full.append(run_with_validation(cached[case["id"]], case.get("_memory_store_patch")))
    full_metrics = _compute_metrics(y_true_full, y_pred_full)
    print(f"  Accuracy: {full_metrics['accuracy']:.2f}%")

    print("\n[Phase 3] Running No Validation (cached)...")
    y_true_nv, y_pred_nv = [], []
    for case in cases:
        y_true_nv.append(case["expected_decision"])
        y_pred_nv.append(run_without_validation(cached[case["id"]], case.get("_memory_store_patch")))
    nv_metrics = _compute_metrics(y_true_nv, y_pred_nv)
    print(f"  Accuracy: {nv_metrics['accuracy']:.2f}%")

    # Show cases where validation made a difference
    diff_count = sum(1 for a, b in zip(y_pred_full, y_pred_nv) if a != b)
    print(f"\n  Cases where validation changed the outcome: {diff_count}")

    # Update CSV
    df = pd.read_csv(CSV_ABLATION_FULL)
    mask = df["configuration"].str.contains("No Validation", na=False)
    old_acc = df.loc[mask, "accuracy"].values[0] if mask.any() else None
    df.loc[mask, "accuracy"] = nv_metrics["accuracy"]
    df.loc[mask, "macro_f1"] = nv_metrics["macro_f1"]

    # Recompute Full Pipeline row too (for consistency)
    mask_full = df["configuration"] == "Full Pipeline"
    df.loc[mask_full, "accuracy"] = full_metrics["accuracy"]
    df.loc[mask_full, "macro_f1"] = full_metrics["macro_f1"]
    df.to_csv(CSV_ABLATION_FULL, index=False)

    print(f"\n  Updated No Validation: {old_acc:.2f}% → {nv_metrics['accuracy']:.2f}%")
    print(f"  Updated Full Pipeline: {full_metrics['accuracy']:.2f}%")

    # Regenerate chart
    fig, ax = plt.subplots(figsize=(10, 6))
    configs = df["configuration"].tolist()
    accs = df["accuracy"].tolist()
    full_acc = df.loc[mask_full, "accuracy"].values[0]
    colors = ["#3498db"] + ["#e74c3c"] * (len(configs) - 1)

    bars = ax.barh(configs[::-1], accs[::-1], color=colors[::-1])
    ax.axvline(full_acc, color="#2ecc71", linestyle="--", linewidth=2, label="Full Pipeline")
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Node Ablation Study — Accuracy Drop per Removed Component")
    ax.set_xlim(0, 105)

    for bar, acc in zip(bars, accs[::-1]):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%", ha="left", va="center", fontsize=9)

    ax.legend()
    fig.tight_layout()
    fig.savefig(PNG_ABLATION_FULL, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"\nSaved updated chart: {PNG_ABLATION_FULL}")
    print("Done.")


if __name__ == "__main__":
    main()
