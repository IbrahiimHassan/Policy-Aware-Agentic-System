"""
Run ONLY the decision node ablation (first-match-wins vs precedence).
Compares against the saved full pipeline results from main_evaluation.csv.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.metrics import f1_score

from engine.memory_agent import apply_memory_to_matching_policies
from engine.policy_engine import can_make_decision, policy_checker
from interpreter.llm_interpreter import llm_interpreter
from run_system import _normalize_out_decision
from utils.config import POLICY_JSON_PATH

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
CSV_MAIN = RESULTS_DIR / "main_evaluation.csv"


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


def run_no_decision_node(request: str, memory_store_patch: Optional[Dict] = None) -> Dict[str, Any]:
    """First matching policy's action wins — no REJECT-first precedence."""
    case_data = llm_interpreter(request).get("case_data", {})

    validation_errors = []
    policy_result = policy_checker(case_data, POLICY_JSON_PATH)
    matching = policy_result.get("matching_policies", [])
    matching, _, hints = apply_memory_to_matching_policies(
        case_data, matching, store_patch=memory_store_patch
    )

    if not matching and not can_make_decision(case_data, POLICY_JSON_PATH):
        decision = "NEED_MORE_INFO"
    elif matching:
        # FIRST match wins, not precedence
        decision = matching[0].get("action", "REJECT")
    else:
        decision = "APPROVED"

    return {"decision": _normalize_out_decision(decision)}


def main():
    if not CSV_MAIN.exists():
        print(f"[ERROR] {CSV_MAIN} not found. Run batch1_eval.py first.")
        return

    # Load full pipeline accuracy from saved CSV
    df_full = pd.read_csv(CSV_MAIN)
    full_acc = (df_full["gold_label"] == df_full["pipeline_decision"]).mean() * 100

    cases = _load_cases()
    y_true = []
    y_pred = []

    print("Running: No Decision Node (first-match-wins) ...")
    for case in cases:
        result = run_no_decision_node(case["input_text"], case.get("_memory_store_patch"))
        y_true.append(case["expected_decision"])
        y_pred.append(result["decision"])

    labels = ["APPROVE", "REJECT", "ESCALATE", "NEED_MORE_INFO"]
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true) * 100
    macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    print(f"\n{'Configuration':<25} {'Accuracy':>10} {'Drop':>10} {'Macro F1':>10}")
    print("-" * 60)
    print(f"{'Full Pipeline (saved)':<25} {full_acc:>9.2f}% {'—':>10} {'—':>10}")
    print(f"{'No Decision Node':<25} {acc:>9.2f}% {acc - full_acc:>+9.1f} {macro:>10.4f}")


if __name__ == "__main__":
    main()
