"""
Batch 2 evaluation: Consistency Analysis, Ablation Study, and Error Analysis.
Assumes batch1_eval.py has been run and results/main_evaluation.csv exists.
Does NOT modify any existing source files.
"""
from __future__ import annotations

import importlib
import json
import os
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Imports from project (do not modify)
# ---------------------------------------------------------------------------
from engine.memory_agent import apply_memory_to_matching_policies
from engine.policy_engine import can_make_decision, policy_checker
from interpreter.llm_interpreter import llm_interpreter
from nodes.decision_node import decision_node
from nodes.explanation_node import explanation_node
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

CSV_BATCH1 = RESULTS_DIR / "main_evaluation.csv"
CSV_CONSISTENCY = RESULTS_DIR / "consistency_results.csv"
JSON_CONSISTENCY = RESULTS_DIR / "consistency_summary.json"
CSV_ABLATION = RESULTS_DIR / "ablation_results.csv"
PNG_ABLATION = FIGURES_DIR / "ablation_comparison.png"
CSV_ERRORS = RESULTS_DIR / "error_analysis.csv"
CSV_ERROR_COMP = RESULTS_DIR / "error_comparison.csv"

LABELS = ["APPROVE", "REJECT", "ESCALATE", "NEED_MORE_INFO"]

# ---------------------------------------------------------------------------
# Load benchmark cases
# ---------------------------------------------------------------------------
from book_benchmark_cases import CASES as BENCHMARK_CASES

CASE_MAP: Dict[str, Dict[str, Any]] = {}
for c in BENCHMARK_CASES:
    CASE_MAP[c["case_id"]] = {
        "case_id": c["case_id"],
        "input": c.get("input_text") or c.get("input") or c.get("request", ""),
        "category": c.get("category", ""),
        "expected_decision": c.get("expected_decision"),
        "_memory_store_patch": c.get("_memory_store_patch"),
        "is_noisy": bool(c.get("is_noisy", False)),
    }

# ---------------------------------------------------------------------------
# Helper: run pipeline with optional skip_explanation
# ---------------------------------------------------------------------------


def _run_pipeline(
    input_text: str, memory_patch: Optional[Dict] = None, skip_explanation: bool = True
) -> Dict[str, Any]:
    """Direct copy of run_system logic so we can tweak it for ablations."""
    import interpreter.llm_interpreter as _interp_mod

    interpreter_result = _interp_mod.llm_interpreter(input_text)
    case_data = interpreter_result.get("case_data", {})

    validation_result = validation_node({"case_data": case_data})
    validation_errors: List[str] = (
        [] if validation_result.get("is_valid", True) else validation_result.get("issues", [])
    )

    policy_result = policy_checker(case_data, POLICY_JSON_PATH)
    matching = policy_result.get("matching_policies", [])
    matching, _, hints = apply_memory_to_matching_policies(
        case_data, matching, store_patch=memory_patch
    )
    matched_ids = sorted([p.get("policy_id") for p in matching if p.get("policy_id")])

    if validation_errors:
        decision = "REJECT"
        explanation = "Request cannot proceed due to invalid input: " + "; ".join(validation_errors)
    elif not matching and not can_make_decision(case_data, POLICY_JSON_PATH):
        decision = "NEED_MORE_INFO"
        explanation = "More information is required to evaluate all relevant policies."
    else:
        decision = decision_node(
            {"validation_errors": validation_errors, "matching_policies": matching}
        ).get("decision", "APPROVED")
        explanation = ""
        if not skip_explanation:
            explanation = explanation_node(
                {
                    "input": input_text,
                    "case_data": case_data,
                    "validation_errors": validation_errors,
                    "matching_policies": matching,
                    "memory_hints": hints,
                }
            ).get("explanation", "")

    return {
        "json": case_data,
        "decision": _normalize_out_decision(decision),
        "policy": matched_ids,
        "explanation": explanation,
        "memory_hints": hints if not validation_errors else [],
        "validation_errors": validation_errors,
        "skipped_policies": policy_result.get("skipped_policies", []),
    }


# ---------------------------------------------------------------------------
# TASK 4 — Consistency Analysis
# ---------------------------------------------------------------------------


def task4_consistency():
    print("\n=== TASK 4 ===")
    print("Running consistency analysis (10 runs per case, skip explanations)...")

    rows = []
    for case in BENCHMARK_CASES:
        cid = case["case_id"]
        text = case["input_text"] if "input_text" in case else case.get("input", "")
        patch = case.get("_memory_store_patch")
        gold = case["expected_decision"]

        decisions = []
        for _ in range(10):
            result = _run_pipeline(text, memory_patch=patch, skip_explanation=True)
            decisions.append(result["decision"])

        strict_consistent = len(set(decisions)) == 1
        majority_decision = Counter(decisions).most_common(1)[0][0]
        soft_consistency_rate = decisions.count(majority_decision) / 10.0
        is_correct_majority = majority_decision == gold

        scenario = "noisy" if case.get("is_noisy", False) else ("memory" if patch else "clean")

        rows.append(
            {
                "case_id": cid,
                "scenario": scenario,
                "gold_label": gold,
                "strict_consistent": strict_consistent,
                "majority_decision": majority_decision,
                "soft_consistency_rate": soft_consistency_rate,
                "is_correct_majority": is_correct_majority,
                "decisions_json": json.dumps(decisions),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(CSV_CONSISTENCY, index=False)

    strict_pct = df["strict_consistent"].mean() * 100
    mean_soft = df["soft_consistency_rate"].mean() * 100

    strict_by_scenario = df.groupby("scenario")["strict_consistent"].mean() * 100
    strict_by_label = df.groupby("gold_label")["strict_consistent"].mean() * 100

    summary = {
        "total_cases": len(df),
        "strict_consistency_percent": round(strict_pct, 2),
        "mean_soft_consistency_rate": round(mean_soft, 2),
        "strict_by_scenario": strict_by_scenario.to_dict(),
        "strict_by_gold_label": strict_by_label.to_dict(),
    }
    with open(JSON_CONSISTENCY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Strict consistency: {strict_pct:.1f}%")
    print(f"  Mean soft consistency rate: {mean_soft:.1f}%")
    print("  Strict consistency by scenario:")
    for k, v in strict_by_scenario.items():
        print(f"    {k}: {v:.1f}%")
    print("  Strict consistency by gold label:")
    for k, v in strict_by_label.items():
        print(f"    {k}: {v:.1f}%")
    print(f"  Saved: {CSV_CONSISTENCY.name}, {JSON_CONSISTENCY.name}")


# ---------------------------------------------------------------------------
# TASK 5 — Ablation Study
# ---------------------------------------------------------------------------


def _compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """Compute accuracy, REJECT F1, NMI F1, and macro F1."""
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    y_true_reject = [1 if y == "REJECT" else 0 for y in y_true]
    y_pred_reject = [1 if y == "REJECT" else 0 for y in y_pred]
    reject_f1 = f1_score(y_true_reject, y_pred_reject, zero_division=0)
    y_true_nmi = [1 if y == "NEED_MORE_INFO" else 0 for y in y_true]
    y_pred_nmi = [1 if y == "NEED_MORE_INFO" else 0 for y in y_pred]
    nmi_f1 = f1_score(y_true_nmi, y_pred_nmi, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    return {
        "accuracy": round(acc * 100, 2),
        "reject_f1": round(reject_f1, 4),
        "nmi_f1": round(nmi_f1, 4),
        "macro_f1": round(macro_f1, 4),
    }


def run_ablation_no_memory(
    request: str, memory_store_patch: Optional[Dict] = None
) -> Dict[str, Any]:
    """Ablation 1 — skip memory node entirely."""
    import interpreter.llm_interpreter as _interp_mod

    interpreter_result = _interp_mod.llm_interpreter(request)
    case_data = interpreter_result.get("case_data", {})

    validation_result = validation_node({"case_data": case_data})
    validation_errors: List[str] = (
        [] if validation_result.get("is_valid", True) else validation_result.get("issues", [])
    )

    policy_result = policy_checker(case_data, POLICY_JSON_PATH)
    matching = policy_result.get("matching_policies", [])
    matched_ids = sorted([p.get("policy_id") for p in matching if p.get("policy_id")])

    if validation_errors:
        decision = "REJECT"
    elif not matching and not can_make_decision(case_data, POLICY_JSON_PATH):
        decision = "NEED_MORE_INFO"
    else:
        decision = decision_node(
            {"validation_errors": validation_errors, "matching_policies": matching}
        ).get("decision", "APPROVED")

    return {
        "json": case_data,
        "decision": _normalize_out_decision(decision),
        "policy": matched_ids,
        "explanation": "",
        "memory_hints": [],
        "validation_errors": validation_errors,
        "skipped_policies": policy_result.get("skipped_policies", []),
    }


def run_ablation_no_validation(
    request: str, memory_store_patch: Optional[Dict] = None
) -> Dict[str, Any]:
    """Ablation 2 — skip validation node entirely."""
    import interpreter.llm_interpreter as _interp_mod

    interpreter_result = _interp_mod.llm_interpreter(request)
    case_data = interpreter_result.get("case_data", {})

    validation_errors: List[str] = []

    policy_result = policy_checker(case_data, POLICY_JSON_PATH)
    matching = policy_result.get("matching_policies", [])
    matching, _, hints = apply_memory_to_matching_policies(
        case_data, matching, store_patch=memory_store_patch
    )
    matched_ids = sorted([p.get("policy_id") for p in matching if p.get("policy_id")])

    if validation_errors:
        decision = "REJECT"
    elif not matching and not can_make_decision(case_data, POLICY_JSON_PATH):
        decision = "NEED_MORE_INFO"
    else:
        decision = decision_node(
            {"validation_errors": validation_errors, "matching_policies": matching}
        ).get("decision", "APPROVED")

    return {
        "json": case_data,
        "decision": _normalize_out_decision(decision),
        "policy": matched_ids,
        "explanation": "",
        "memory_hints": hints,
        "validation_errors": validation_errors,
        "skipped_policies": policy_result.get("skipped_policies", []),
    }


def _monkeypatch_no_retry():
    """Monkey-patch llm_interpreter to disable the single retry call."""
    import interpreter.llm_interpreter as interp_mod

    source_path = Path(interp_mod.__file__)
    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    old_block = """    data = run_once(prompt)
    # Retry if model returned empty object or missing keys
    if not data or set(data.keys()) != set(ALLOWED_FIELDS):
        print("[WARN] Retrying interpreter once (incomplete JSON keys).")
        data = run_once(retry_prompt)"""

    new_block = "    data = run_once(prompt)"

    if old_block not in source:
        print("[WARN] Retry block not found in source; no-retry ablation may be identical to full pipeline.")
        return False

    modified_source = source.replace(old_block, new_block)
    namespace = interp_mod.__dict__.copy()
    exec(compile(modified_source, str(source_path), "exec"), namespace)
    interp_mod.llm_interpreter = namespace["llm_interpreter"]
    return True


def _restore_llm_interpreter():
    import interpreter.llm_interpreter as interp_mod

    importlib.reload(interp_mod)


def run_ablation_no_retry(
    request: str, memory_store_patch: Optional[Dict] = None
) -> Dict[str, Any]:
    """Ablation 3 — disable interpreter retry."""
    return _run_pipeline(request, memory_patch=memory_store_patch, skip_explanation=True)


def run_ablation_no_explanation(
    request: str, memory_store_patch: Optional[Dict] = None
) -> Dict[str, Any]:
    """Ablation 4 — skip explanation node entirely (same decision path, no LLM call)."""
    return _run_pipeline(request, memory_patch=memory_store_patch, skip_explanation=True)


def task5_ablation():
    print("\n=== TASK 5 ===")
    print("Running ablation study...")

    df_batch1 = pd.read_csv(CSV_BATCH1)
    full_y_true = df_batch1["gold_label"].tolist()
    full_y_pred = df_batch1["pipeline_decision"].tolist()
    full_metrics = _compute_metrics(full_y_true, full_y_pred)

    ablations = [
        ("No Memory", run_ablation_no_memory),
        ("No Validation", run_ablation_no_validation),
        ("No Explanation", run_ablation_no_explanation),
    ]

    results = [
        {
            "configuration": "Full Pipeline",
            **full_metrics,
        }
    ]

    for name, ab_fn in ablations:
        y_true = []
        y_pred = []
        for case in BENCHMARK_CASES:
            text = case["input_text"] if "input_text" in case else case.get("input", "")
            result = ab_fn(text, case.get("_memory_store_patch"))
            y_true.append(case["expected_decision"])
            y_pred.append(result["decision"])
        metrics = _compute_metrics(y_true, y_pred)
        results.append({"configuration": f"— {name}", **metrics})
        print(f"  {name}: accuracy={metrics['accuracy']:.2f}%, macro_f1={metrics['macro_f1']:.4f}")

    patched = _monkeypatch_no_retry()
    y_true = []
    y_pred = []
    for case in BENCHMARK_CASES:
        text = case["input_text"] if "input_text" in case else case.get("input", "")
        result = run_ablation_no_retry(text, case.get("_memory_store_patch"))
        y_true.append(case["expected_decision"])
        y_pred.append(result["decision"])
    metrics = _compute_metrics(y_true, y_pred)
    results.append({"configuration": "— No Retry", **metrics})
    print(f"  No Retry: accuracy={metrics['accuracy']:.2f}%, macro_f1={metrics['macro_f1']:.4f}")
    if patched:
        _restore_llm_interpreter()

    df_results = pd.DataFrame(results)
    df_results.to_csv(CSV_ABLATION, index=False)

    print("\nConfiguration        | Accuracy | REJECT F1 | NMI F1 | Macro F1")
    print("-" * 65)
    for r in results:
        print(
            f"{r['configuration']:<20} | "
            f"{r['accuracy']:>7.2f}% | "
            f"{r['reject_f1']:>9.4f} | "
            f"{r['nmi_f1']:>6.4f} | "
            f"{r['macro_f1']:>8.4f}"
        )

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(results))
    full_acc = results[0]["accuracy"]
    accs = [r["accuracy"] for r in results]
    drops = [full_acc - a for a in accs]

    colors = ["#2ecc71"]
    for drop in drops[1:]:
        if drop > 5:
            colors.append("#e74c3c")
        elif drop >= 2:
            colors.append("#f39c12")
        else:
            colors.append("#2ecc71")

    bars = ax.bar(x, drops, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([r["configuration"] for r in results], rotation=15, ha="right")
    ax.set_ylabel("Accuracy Drop vs Full Pipeline (%)")
    ax.set_title("Ablation Study: Accuracy Drop per Configuration")
    ax.axhline(0, color="black", linewidth=0.5)
    for bar, drop in zip(bars, drops):
        height = bar.get_height()
        ax.annotate(
            f"{drop:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(PNG_ABLATION, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {CSV_ABLATION.name}, {PNG_ABLATION.name}")


# ---------------------------------------------------------------------------
# TASK 6 — Error Analysis
# ---------------------------------------------------------------------------


def _classify_pipeline_error(row: pd.Series, case: Dict[str, Any]) -> str:
    """Classify a single pipeline error into one category."""
    result = _run_pipeline(case["input"], case.get("_memory_store_patch"), skip_explanation=True)
    case_data = result.get("json", {})
    skipped = result.get("skipped_policies", [])
    patch = case.get("_memory_store_patch")

    request_type = case_data.get("request_type")
    books_req = case_data.get("books_requested")
    days_req = case_data.get("days_requested")
    book_cat = case_data.get("book_category")

    # MEMORY_CAP_BOUNDARY: memory case where stored + requested is within 1 of cap
    if patch and patch.get("library"):
        stored = patch["library"].get("current_books_borrowed", 0)
        total = stored + (books_req or 0)
        cap = 2
        if abs(total - cap) <= 1:
            return "MEMORY_CAP_BOUNDARY"

    # NOISE_INDUCED
    if row.get("scenario") == "noisy" or case.get("is_noisy", False):
        return "NOISE_INDUCED"

    # MISSING_FIELD: required field is null causing a policy to be skipped
    if skipped:
        return "MISSING_FIELD"

    # EXTRACTION_ERROR: key fields wrong/null when clearly stated
    input_lower = case["input"].lower()
    if request_type is None and any(
        k in input_lower for k in ["borrow", "renew", "reserve", "lost", "damaged"]
    ):
        return "EXTRACTION_ERROR"
    if books_req is None and any(
        w in input_lower for w in ["book", "books", "one", "two", "three"]
    ):
        return "EXTRACTION_ERROR"
    if days_req is None and any(w in input_lower for w in ["day", "days", "week"]):
        return "EXTRACTION_ERROR"
    if book_cat is None and any(
        w in input_lower
        for w in [
            "regular",
            "reference",
            "thesis",
            "dvd",
            "cd",
            "encyclopedia",
            "dictionary",
            "newspaper",
            "magazine",
            "journal",
            "atlas",
            "illustrated",
        ]
    ):
        return "EXTRACTION_ERROR"
    if request_type == "borrow" and "renew" in input_lower:
        return "EXTRACTION_ERROR"
    if request_type == "renew" and "borrow" in input_lower:
        return "EXTRACTION_ERROR"

    # POLICY_MISMATCH: fields extracted correctly but wrong policy fired
    return "POLICY_MISMATCH"


def _classify_baseline_a_error(row: pd.Series, case: Dict[str, Any]) -> str:
    """Simplified error classification for Baseline A."""
    gold = row["gold_label"]
    pred = row["baselineA_decision"]
    patch = case.get("_memory_store_patch")

    if patch and gold == "REJECT" and pred == "APPROVE":
        return "MEMORY_BLIND"
    if gold == "NEED_MORE_INFO" and pred != "NEED_MORE_INFO":
        return "NMI_MISSED"
    if case["category"] in ["borrow_approve", "borrow_reject", "fines_reject"]:
        return "THRESHOLD_HALLUCINATION"
    return "OTHER"


def _classify_baseline_b_error(row: pd.Series, case: Dict[str, Any]) -> str:
    """Simplified error classification for Baseline B."""
    gold = row["gold_label"]
    pred = row["baselineB_decision"]
    patch = case.get("_memory_store_patch")

    if patch and gold == "REJECT" and pred == "APPROVE":
        return "MEMORY_BLIND"
    if gold == "NEED_MORE_INFO" and pred != "NEED_MORE_INFO":
        return "NMI_MISSED"
    if case["category"] in ["borrow_approve", "borrow_reject"]:
        return "COMPOUND_CONDITION"
    return "OTHER"


def task6_error_analysis():
    print("\n=== TASK 6 ===")
    print("Running error analysis...")

    df = pd.read_csv(CSV_BATCH1)

    # A) Pipeline errors
    pipe_errors = df[df["pipeline_decision"] != df["gold_label"]].copy()
    pipe_errors["error_category"] = pipe_errors.apply(
        lambda row: _classify_pipeline_error(row, CASE_MAP.get(row["case_id"], {})), axis=1
    )

    total_errors = len(pipe_errors)
    cat_counts = pipe_errors["error_category"].value_counts().to_dict()

    print("\nA) Pipeline Error Category Frequency:")
    print(f"{'Category':<25} | {'Count':>5} | {'% of errors':>10} | {'% of total':>10}")
    print("-" * 60)
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        pct_errors = cnt / total_errors * 100 if total_errors else 0
        pct_total = cnt / len(df) * 100
        print(f"{cat:<25} | {cnt:>5} | {pct_errors:>9.1f}% | {pct_total:>9.1f}%")

    # B) Baseline A errors
    ba_errors = df[df["baselineA_decision"] != df["gold_label"]].copy()
    ba_errors["error_category"] = ba_errors.apply(
        lambda row: _classify_baseline_a_error(row, CASE_MAP.get(row["case_id"], {})), axis=1
    )
    ba_cat_counts = ba_errors["error_category"].value_counts().to_dict()

    print("\nB) Baseline A Error Category Frequency:")
    print(f"{'Category':<25} | {'Count':>5} | {'% of errors':>10} | {'% of total':>10}")
    print("-" * 60)
    for cat, cnt in sorted(ba_cat_counts.items(), key=lambda x: -x[1]):
        pct_errors = cnt / len(ba_errors) * 100 if len(ba_errors) else 0
        pct_total = cnt / len(df) * 100
        print(f"{cat:<25} | {cnt:>5} | {pct_errors:>9.1f}% | {pct_total:>9.1f}%")

    # C) Side-by-side comparison
    all_cats = sorted(set(list(cat_counts.keys()) + list(ba_cat_counts.keys())))
    bb_errors = df[df["baselineB_decision"] != df["gold_label"]].copy()
    bb_errors["error_category"] = bb_errors.apply(
        lambda row: _classify_baseline_b_error(row, CASE_MAP.get(row["case_id"], {})), axis=1
    )
    bb_cat_counts = bb_errors["error_category"].value_counts().to_dict()

    print("\nC) Side-by-side Error Comparison:")
    print(f"{'Error Type':<25} | {'Pipeline':>8} | {'Baseline A':>10} | {'Baseline B':>10}")
    print("-" * 65)
    for cat in all_cats:
        p = cat_counts.get(cat, 0)
        a = ba_cat_counts.get(cat, 0)
        b = bb_cat_counts.get(cat, 0)
        print(f"{cat:<25} | {p:>8} | {a:>10} | {b:>10}")

    # Save error tables
    pipe_errors["input"] = pipe_errors["case_id"].apply(
        lambda cid: CASE_MAP.get(cid, {}).get("input", "")
    )
    pipe_errors[
        ["case_id", "category", "scenario", "gold_label", "pipeline_decision", "error_category", "input"]
    ].to_csv(CSV_ERRORS, index=False)

    comp_rows = []
    for cat in all_cats:
        comp_rows.append(
            {
                "error_type": cat,
                "pipeline_count": cat_counts.get(cat, 0),
                "baselineA_count": ba_cat_counts.get(cat, 0),
                "baselineB_count": bb_cat_counts.get(cat, 0),
            }
        )
    pd.DataFrame(comp_rows).to_csv(CSV_ERROR_COMP, index=False)

    # Illustrative failure cases
    print("\n--- Illustrative Failure Cases ---")

    pref_cats = ["EXTRACTION_ERROR", "MEMORY_CAP_BOUNDARY", "MISSING_FIELD", "POLICY_MISMATCH", "NOISE_INDUCED"]
    case1 = None
    for cat in pref_cats:
        subset = pipe_errors[pipe_errors["error_category"] == cat]
        if not subset.empty:
            case1 = subset.iloc[0]
            break

    if case1 is not None:
        cid = case1["case_id"]
        c = CASE_MAP[cid]
        result = _run_pipeline(c["input"], c.get("_memory_store_patch"), skip_explanation=True)
        print(f"\nCase 1 — Pipeline ({case1['error_category']})  [{cid}]")
        print(f"  raw_text: {c['input'][:120]}")
        print(f"  gold_label: {case1['gold_label']}  |  predicted: {case1['pipeline_decision']}")
        print(f"  extracted_fields: {json.dumps(result.get('json', {}))}")
        print(f"  matched_policies: {result.get('policy', [])}")
        expl = (case1.get("pipeline_explanation") or "")[:300]
        print(f"  explanation: {expl}...")

    ba_pref = ba_errors[ba_errors["error_category"] == "THRESHOLD_HALLUCINATION"]
    if ba_pref.empty:
        ba_pref = ba_errors
    case2 = ba_pref.iloc[0] if not ba_pref.empty else None

    if case2 is not None:
        cid = case2["case_id"]
        c = CASE_MAP[cid]
        print(f"\nCase 2 — Baseline A ({case2['error_category']})  [{cid}]")
        print(f"  raw_text: {c['input'][:120]}")
        print(f"  gold_label: {case2['gold_label']}  |  predicted: {case2['baselineA_decision']}")

    bb_pref = bb_errors[bb_errors["error_category"] == "COMPOUND_CONDITION"]
    if bb_pref.empty:
        bb_pref = bb_errors
    case3 = bb_pref.iloc[0] if not bb_pref.empty else None

    if case3 is not None:
        cid = case3["case_id"]
        c = CASE_MAP[cid]
        print(f"\nCase 3 — Baseline B ({case3['error_category']})  [{cid}]")
        print(f"  raw_text: {c['input'][:120]}")
        print(f"  gold_label: {case3['gold_label']}  |  predicted: {case3['baselineB_decision']}")

    print(f"\n  Saved: {CSV_ERRORS.name}, {CSV_ERROR_COMP.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch 2 evaluation")
    parser.add_argument("--ablation-only", action="store_true", help="Run only Task 5 (ablation study)")
    parser.add_argument("--consistency-only", action="store_true", help="Run only Task 4 (consistency)")
    parser.add_argument("--error-only", action="store_true", help="Run only Task 6 (error analysis)")
    args = parser.parse_args()

    if args.ablation_only:
        try:
            task5_ablation()
        except Exception:
            traceback.print_exc()
        print("\n=== Ablation-only run complete ===")
        return

    if args.consistency_only:
        try:
            task4_consistency()
        except Exception:
            traceback.print_exc()
        print("\n=== Consistency-only run complete ===")
        return

    if args.error_only:
        try:
            task6_error_analysis()
        except Exception:
            traceback.print_exc()
        print("\n=== Error-only run complete ===")
        return

    for task in [task4_consistency, task5_ablation, task6_error_analysis]:
        try:
            task()
        except Exception as e:
            print(f"\n[ERROR in {task.__name__}]")
            traceback.print_exc()
            print("Continuing...")
    print("\n=== batch2_eval.py complete ===")


if __name__ == "__main__":
    main()
