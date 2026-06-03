"""
Comprehensive node ablation study.
Removes every pipeline node one at a time and measures accuracy drop.
Assumes batch1_eval.py has been run and results/main_evaluation.csv exists.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from engine.memory_agent import apply_memory_to_matching_policies
from engine.policy_engine import can_make_decision, policy_checker
from interpreter.llm_interpreter import heuristic_extract
from nodes.decision_node import decision_node
from nodes.validation_node import validation_node
from run_system import _normalize_out_decision
from utils.config import POLICY_JSON_PATH

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CSV_MAIN = RESULTS_DIR / "main_evaluation.csv"
CSV_ABLATION_FULL = RESULTS_DIR / "ablation_full.csv"
PNG_ABLATION_FULL = FIGURES_DIR / "ablation_full.png"

os.environ.setdefault("OPENAI_MODEL", "gpt-4o-mini")


def _adapt_case(row):
    """Normalize book_benchmark_cases row to script field names."""
    return {
        "id": row.get("case_id") or row.get("id"),
        "input": row.get("input_text") or row.get("input") or row.get("request"),
        "input_text": row.get("input_text") or row.get("input") or row.get("request"),
        "expected_decision": row.get("expected_decision") or row.get("decision"),
        "expected_policy_ids": row.get("expected_policy_ids") or row.get("policy_ids", []),
        "_memory_store_patch": row.get("_memory_store_patch") or row.get("memory_store_patch"),
        "_is_noisy": row.get("_is_noisy", False),
    }


def _load_benchmark_cases():
    from book_benchmark_cases import CASES
    return [_adapt_case(c) for c in CASES]


def _compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    labels = ["APPROVE", "REJECT", "ESCALATE", "NEED_MORE_INFO"]
    accuracy = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true) * 100
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    reject_f1 = f1_score(y_true, y_pred, labels=["REJECT"], average="macro", zero_division=0)
    nmi_f1 = f1_score(y_true, y_pred, labels=["NEED_MORE_INFO"], average="macro", zero_division=0)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "reject_f1": reject_f1,
        "nmi_f1": nmi_f1,
    }


def _pipeline_from_parts(
    case_data: Dict[str, Any],
    input_text: str,
    memory_store_patch: Optional[Dict] = None,
    *,
    use_validation: bool = True,
    use_memory: bool = True,
    use_decision_node: bool = True,
) -> Dict[str, Any]:
    """Run pipeline from structured case_data with toggles for each node."""
    if use_validation:
        validation_result = validation_node({"case_data": case_data})
        validation_errors: List[str] = (
            [] if validation_result.get("is_valid", True) else validation_result.get("issues", [])
        )
    else:
        validation_errors = []

    policy_result = policy_checker(case_data, POLICY_JSON_PATH)
    matching = policy_result.get("matching_policies", [])

    if use_memory:
        matching, _, hints = apply_memory_to_matching_policies(
            case_data, matching, store_patch=memory_store_patch
        )
    else:
        hints = []

    matched_ids = sorted([p.get("policy_id") for p in matching if p.get("policy_id")])

    if validation_errors:
        decision = "REJECT"
    elif not matching and not can_make_decision(case_data, POLICY_JSON_PATH):
        decision = "NEED_MORE_INFO"
    else:
        if use_decision_node:
            decision = decision_node(
                {"validation_errors": validation_errors, "matching_policies": matching}
            ).get("decision", "APPROVED")
        else:
            # No decision node: first matching policy's action wins (no precedence override)
            if matching:
                first_action = matching[0].get("action", "REJECT")
                decision = first_action
            else:
                decision = "NEED_MORE_INFO"

    return {
        "json": case_data,
        "decision": _normalize_out_decision(decision),
        "policy": matched_ids,
        "explanation": "",
        "memory_hints": hints if not validation_errors else [],
        "validation_errors": validation_errors,
        "skipped_policies": policy_result.get("skipped_policies", []),
    }


# ---------------------------------------------------------------------------
# ABLATION VARIANTS
# ---------------------------------------------------------------------------

def run_full_pipeline(
    request: str, memory_store_patch: Optional[Dict] = None
) -> Dict[str, Any]:
    """Baseline — full pipeline via run_system."""
    from run_system import run_system
    return run_system(request, memory_store_patch, skip_explanation=True)


def run_no_interpreter(
    request: str, memory_store_patch: Optional[Dict] = None
) -> Dict[str, Any]:
    """Ablation — use deterministic heuristic_extract only, no LLM interpreter."""
    case_data = heuristic_extract(request)
    return _pipeline_from_parts(case_data, request, memory_store_patch)


def run_no_validation(
    request: str, memory_store_patch: Optional[Dict] = None
) -> Dict[str, Any]:
    """Ablation — skip validation node."""
    from interpreter.llm_interpreter import llm_interpreter
    case_data = llm_interpreter(request).get("case_data", {})
    return _pipeline_from_parts(
        case_data, request, memory_store_patch, use_validation=False
    )


def _run_baseline_b_for_ablation(request: str) -> Dict[str, Any]:
    """Wrapper for baseline_b that returns standard dict."""
    from baseline_b import run_baseline_b
    decision = run_baseline_b(request)
    return {
        "json": {},
        "decision": decision,
        "policy": [],
        "explanation": "",
        "memory_hints": [],
        "validation_errors": [],
        "skipped_policies": [],
    }


def run_no_memory(
    request: str, memory_store_patch: Optional[Dict] = None
) -> Dict[str, Any]:
    """Ablation — skip memory agent entirely."""
    from interpreter.llm_interpreter import llm_interpreter
    case_data = llm_interpreter(request).get("case_data", {})
    return _pipeline_from_parts(
        case_data, request, memory_store_patch, use_memory=False
    )


def run_no_decision_node(
    request: str, memory_store_patch: Optional[Dict] = None
) -> Dict[str, Any]:
    """Ablation — no explicit precedence. First matching policy's action wins."""
    from interpreter.llm_interpreter import llm_interpreter
    case_data = llm_interpreter(request).get("case_data", {})
    return _pipeline_from_parts(
        case_data, request, memory_store_patch, use_decision_node=False
    )


def run_no_explanation(
    request: str, memory_store_patch: Optional[Dict] = None
) -> Dict[str, Any]:
    """Ablation — skip explanation node (same decision path, no LLM call).
    Identical to full pipeline since explanation is post-decision."""
    return run_full_pipeline(request, memory_store_patch)


def run_no_validation_no_memory(
    request: str, memory_store_patch: Optional[Dict] = None
) -> Dict[str, Any]:
    """Combo ablation — remove both validation and memory."""
    from interpreter.llm_interpreter import llm_interpreter
    case_data = llm_interpreter(request).get("case_data", {})
    return _pipeline_from_parts(
        case_data, request, memory_store_patch, use_validation=False, use_memory=False
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("COMPREHENSIVE NODE ABLATION STUDY")
    print("=" * 60)

    if not CSV_MAIN.exists():
        print(f"[ERROR] {CSV_MAIN} not found. Run batch1_eval.py first.")
        return

    BENCHMARK_CASES = _load_benchmark_cases()

    # -----------------------------------------------------------------------
    # CACHE: run LLM interpreter ONCE for all cases, reuse across ablations
    # This eliminates non-deterministic interpreter variance
    # -----------------------------------------------------------------------
    print("\n[Phase 1] Running LLM interpreter once and caching outputs...")
    from interpreter.llm_interpreter import llm_interpreter
    cached_case_data = {}
    for case in BENCHMARK_CASES:
        text = case.get("input_text", "")
        case_data = llm_interpreter(text).get("case_data", {})
        cached_case_data[case["id"]] = case_data
    print(f"  Cached {len(cached_case_data)} interpreter outputs.")

    # Ablations that need cached interpreter data receive it directly
    def _run_cached(name, fn, use_cache=True):
        print(f"\nRunning: {name} ...")
        y_true = []
        y_pred = []
        for case in BENCHMARK_CASES:
            text = case.get("input_text", "")
            if use_cache:
                case_data = cached_case_data[case["id"]]
                result = fn(text, case.get("_memory_store_patch"), case_data)
            else:
                result = fn(text, case.get("_memory_store_patch"))
            y_true.append(case["expected_decision"])
            y_pred.append(result["decision"])
        metrics = _compute_metrics(y_true, y_pred)
        print(f"  Accuracy: {metrics['accuracy']:.2f}%  |  Macro F1: {metrics['macro_f1']:.4f}")
        return {"configuration": name, **metrics}

    ablations = [
        # (name, function, use_cached_case_data)
        ("Full Pipeline", lambda t, m, cd: _pipeline_from_parts(cd, t, m), True),
        ("No Interpreter", lambda t, m, cd: _pipeline_from_parts(heuristic_extract(t), t, m), True),  # uses heuristic, not cache
        ("No Validation", lambda t, m, cd: _pipeline_from_parts(cd, t, m, use_validation=False), True),
        ("No Policy Engine", lambda t, m, cd: _run_baseline_b_for_ablation(t), True),  # special, doesn't use case_data
        ("No Memory Agent", lambda t, m, cd: _pipeline_from_parts(cd, t, m, use_memory=False), True),
        ("No Decision Node", lambda t, m, cd: _pipeline_from_parts(cd, t, m, use_decision_node=False), True),
        ("No Explanation", lambda t, m, cd: _pipeline_from_parts(cd, t, m), True),  # same as full, explanation is post-decision
        ("No Validation + No Memory", lambda t, m, cd: _pipeline_from_parts(cd, t, m, use_validation=False, use_memory=False), True),
    ]

    results = []
    for name, fn, use_cache in ablations:
        r = _run_cached(name, fn, use_cache)
        results.append(r)

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(CSV_ABLATION_FULL, index=False)
    print(f"\nSaved: {CSV_ABLATION_FULL}")

    # Summary table
    print("\n" + "=" * 60)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Configuration':<28} {'Accuracy':>10} {'Drop':>10} {'Macro F1':>10}")
    print("-" * 60)

    full_acc = results[0]["accuracy"]
    for r in results:
        drop = r["accuracy"] - full_acc
        print(
            f"{r['configuration']:<28} {r['accuracy']:>9.2f}% {drop:>+9.1f} {r['macro_f1']:>10.4f}"
        )

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    configs = [r["configuration"] for r in results]
    accs = [r["accuracy"] for r in results]
    colors = ["#3498db"] + ["#e74c3c"] * (len(results) - 1)

    bars = ax.barh(configs[::-1], accs[::-1], color=colors[::-1])
    ax.axvline(full_acc, color="#2ecc71", linestyle="--", linewidth=2, label="Full Pipeline")
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Node Ablation Study — Accuracy Drop per Removed Component")
    ax.set_xlim(0, 105)

    for bar, acc in zip(bars, accs[::-1]):
        width = bar.get_width()
        ax.text(
            width + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.1f}%",
            ha="left",
            va="center",
            fontsize=9,
        )

    ax.legend()
    fig.tight_layout()
    fig.savefig(PNG_ABLATION_FULL, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {PNG_ABLATION_FULL}")
    print("\n=== ablation_full.py complete ===")


if __name__ == "__main__":
    main()
