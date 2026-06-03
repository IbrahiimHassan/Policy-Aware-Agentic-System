"""
Patch the ablation chart: re-run ONLY the Decision Node ablation,
update the CSV + PNG without touching other ablations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score

from engine.memory_agent import apply_memory_to_matching_policies
from engine.policy_engine import can_make_decision, policy_checker
from interpreter.llm_interpreter import llm_interpreter
from run_system import _normalize_out_decision
from utils.config import POLICY_JSON_PATH

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CSV_ABLATION_FULL = RESULTS_DIR / "ablation_full.csv"
PNG_ABLATION_FULL = FIGURES_DIR / "ablation_full.png"


def _adapt_case(row):
    return {
        "input_text": row.get("input_text") or row.get("input") or row.get("request"),
        "expected_decision": row.get("expected_decision") or row.get("decision"),
        "_memory_store_patch": row.get("_memory_store_patch") or row.get("memory_store_patch"),
    }


def _load_cases():
    from book_benchmark_cases import CASES
    return [_adapt_case(c) for c in CASES]


def run_no_decision_node(request: str, memory_store_patch: Optional[Dict] = None) -> Dict[str, Any]:
    """First matching policy's action wins — no REJECT-first precedence."""
    case_data = llm_interpreter(request).get("case_data", {})
    policy_result = policy_checker(case_data, POLICY_JSON_PATH)
    matching = policy_result.get("matching_policies", [])
    matching, _, _ = apply_memory_to_matching_policies(
        case_data, matching, store_patch=memory_store_patch
    )

    if not matching and not can_make_decision(case_data, POLICY_JSON_PATH):
        decision = "NEED_MORE_INFO"
    elif matching:
        decision = matching[0].get("action", "REJECT")
    else:
        decision = "APPROVED"

    return {"decision": _normalize_out_decision(decision)}


def main():
    if not CSV_ABLATION_FULL.exists():
        print(f"[ERROR] {CSV_ABLATION_FULL} not found. Run ablation_full.py first.")
        return

    df = pd.read_csv(CSV_ABLATION_FULL)

    # Run ONLY the decision node ablation
    cases = _load_cases()
    y_true = []
    y_pred = []
    print("Re-running: No Decision Node (first-match-wins) ...")
    for case in cases:
        result = run_no_decision_node(case["input_text"], case.get("_memory_store_patch"))
        y_true.append(case["expected_decision"])
        y_pred.append(result["decision"])

    labels = ["APPROVE", "REJECT", "ESCALATE", "NEED_MORE_INFO"]
    new_acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true) * 100
    new_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    # Update the row in the DataFrame
    mask = df["configuration"].str.contains("No Decision Node", na=False)
    old_acc = df.loc[mask, "accuracy"].values[0] if mask.any() else None
    df.loc[mask, "accuracy"] = new_acc
    df.loc[mask, "macro_f1"] = new_macro

    # Save updated CSV
    df.to_csv(CSV_ABLATION_FULL, index=False)
    print(f"  Updated accuracy: {old_acc:.2f}% → {new_acc:.2f}% (drop: {new_acc - df.loc[0, 'accuracy']:+.1f})")

    # Regenerate chart
    fig, ax = plt.subplots(figsize=(10, 6))
    configs = df["configuration"].tolist()
    accs = df["accuracy"].tolist()
    full_acc = accs[0]
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
    print(f"  Saved updated chart: {PNG_ABLATION_FULL}")
    print("\nDone.")


if __name__ == "__main__":
    main()
