"""
Batch 1 evaluation: Tasks 1–3.
  TASK 1: Smoke tests (policy_checker)
  TASK 2: Dataset distribution tables and CSV
  TASK 3: Main evaluation run (pipeline + baselines)

Imports existing modules only — does not modify project source files.
"""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

LABELS: List[str] = ["APPROVE", "REJECT", "ESCALATE", "NEED_MORE_INFO"]


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def normalize_label(raw: Any) -> str:
    if raw is None:
        return "NEED_MORE_INFO"
    text = str(raw).strip().upper().replace(" ", "_").replace("-", "_")
    if "NEED_MORE" in text or text in ("NEED_MORE_INFO", "NMI"):
        return "NEED_MORE_INFO"
    mapping = {
        "APPROVED": "APPROVE",
        "REJECTED": "REJECT",
        "TERMINATED": "REJECT",
        "ESCALATED": "ESCALATE",
    }
    if text in mapping:
        return mapping[text]
    if text in LABELS:
        return text
    for token in LABELS:
        if token in text.split():
            return token
    return "NEED_MORE_INFO"


def scenario_type(case: Dict[str, Any]) -> str:
    if case.get("_memory_store_patch"):
        return "memory"
    if case.get("is_noisy"):
        return "noisy"
    return "clean"


def leave_type_from_category(category: str) -> str:
    """Map benchmark category to a coarse bucket for tables."""
    cat = (category or "").strip()
    if cat == "memory_reject":
        return "memory"
    if cat == "ambiguous":
        return "ambiguous_nmi"
    if cat.startswith("borrow_approve"):
        return "borrow_approve"
    if cat.startswith("borrow_reject"):
        return "borrow_reject"
    if cat.startswith("noncirculating_"):
        return "non_circulating"
    if cat.startswith("renew_"):
        return "renewal"
    if cat.startswith("reserve_"):
        return "reservation"
    if cat.startswith("report_"):
        return "lost_damaged"
    if cat.startswith("fines_"):
        return "fines"
    return cat or "unknown"


def adapt_case(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize book_benchmark_cases row to script field names."""
    return {
        "id": row.get("case_id") or row.get("id"),
        "input": row.get("input_text") or row.get("input") or row.get("request"),
        "request": row.get("input_text") or row.get("input") or row.get("request"),
        "expected_decision": row.get("expected_decision"),
        "leave_type": leave_type_from_category(str(row.get("category", ""))),
        "category": row.get("category", ""),
        "scenario": scenario_type(row),
        "is_noisy": bool(row.get("is_noisy", False)),
        "_memory_store_patch": row.get("_memory_store_patch"),
        "pair_id": row.get("pair_id", row.get("case_id")),
    }


def load_benchmark_cases() -> List[Dict[str, Any]]:
    from book_benchmark_cases import CASES as BENCHMARK_CASES
    return [adapt_case(c) for c in BENCHMARK_CASES]


def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    return float(
        f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    )


def weighted_f1(y_true: List[str], y_pred: List[str]) -> float:
    return float(
        f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
    )


def print_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(df.to_string(index=False))


def save_distribution_csv(tables: Dict[str, pd.DataFrame]) -> None:
    parts: List[pd.DataFrame] = []
    for name, frame in tables.items():
        header = pd.DataFrame([{"section": name}])
        parts.append(header)
        parts.append(frame)
        parts.append(pd.DataFrame([{}]))
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(RESULTS / "dataset_distribution.csv", index=False, encoding="utf-8-sig")


def plot_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    title: str,
    csv_path: Path,
    png_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(csv_path, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(title)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, facecolor="white")
    plt.close(fig)


def eval_category_bucket(category: str) -> str:
    """Aggregate benchmark categories into spec buckets for Task 3E."""
    cat = str(category or "")
    if cat in ("borrow_approve", "borrow_reject", "ambiguous", "memory_reject"):
        return {"ambiguous": "ambiguous_nmi", "memory_reject": "memory"}.get(cat, cat)
    if cat.startswith("noncirculating_"):
        return "non_circulating"
    if cat.startswith("renew_"):
        return "renewal"
    if cat.startswith("reserve_"):
        return "reservation"
    if cat.startswith("report_"):
        return "lost_damaged"
    if cat.startswith("fines_"):
        return "fines"
    return cat or "unknown"


EVAL_BUCKET_ORDER = [
    "borrow_approve",
    "borrow_reject",
    "non_circulating",
    "renewal",
    "reservation",
    "lost_damaged",
    "fines",
    "ambiguous_nmi",
    "memory",
]


def task1_smoke_tests() -> None:
    print("\n=== TASK 1 — Smoke Tests (policy_checker) ===")
    from engine.policy_engine import policy_checker
    from test_policies import TESTS
    from utils.config import POLICY_JSON_PATH

    passed = 0
    failed = 0
    for test in TESTS:
        result = policy_checker(test["case_data"], POLICY_JSON_PATH)
        got = sorted(p["policy_id"] for p in result.get("matching_policies", []))
        exp = sorted(test["expected_ids"])
        case_id = test.get("label", "unknown")
        if got == exp:
            passed += 1
        else:
            failed += 1
            print(f"\nFAIL  case id: {case_id}")
            print(f"  expected policy IDs: {exp}")
            print(f"  actual policy IDs:   {got}")
    total = len(TESTS)
    print(f"\nTotal tests: {total}")
    print(f"Passed:      {passed}")
    print(f"Failed:      {failed}")


def task2_dataset_distribution(cases: List[Dict[str, Any]]) -> None:
    print("\n=== TASK 2 — Dataset Distribution ===")
    n = len(cases)
    gold = [normalize_label(c["expected_decision"]) for c in cases]
    scenarios = [c["scenario"] for c in cases]
    leave_types = [c["leave_type"] for c in cases]

    table_a = (
        pd.Series(gold, name="Outcome")
        .value_counts()
        .reindex(LABELS, fill_value=0)
        .reset_index()
    )
    table_a.columns = ["Outcome", "Count"]
    table_a["Percent"] = (table_a["Count"] / n * 100).round(2)

    table_b = (
        pd.Series(scenarios, name="Scenario")
        .value_counts()
        .reindex(["clean", "memory", "noisy"], fill_value=0)
        .reset_index()
    )
    table_b.columns = ["Scenario", "Count"]
    table_b["Percent"] = (table_b["Count"] / n * 100).round(2)

    table_c = (
        pd.Series(leave_types, name="Category")
        .value_counts()
        .reset_index()
    )
    table_c.columns = ["Category", "Count"]
    table_c["Percent"] = (table_c["Count"] / n * 100).round(2)

    table_d = pd.crosstab(
        pd.Series(scenarios, name="scenario"),
        pd.Series(gold, name="expected_decision"),
        dropna=False,
    )
    for label in LABELS:
        if label not in table_d.columns:
            table_d[label] = 0
    table_d = table_d[LABELS]
    table_d.index.name = "Scenario"

    print_table("Table A: Distribution by gold-standard label", table_a)
    print_table("Table B: Distribution by scenario type", table_b)
    print_table("Table C: Distribution by category", table_c)
    print("\nTable D: scenario_type × expected_decision")
    print(table_d.to_string())

    save_distribution_csv(
        {
            "Table_A_gold_label": table_a,
            "Table_B_scenario": table_b,
            "Table_C_category": table_c,
            "Table_D_crosstab": table_d.reset_index(),
        }
    )
    print(f"\nSaved: {RESULTS / 'dataset_distribution.csv'}")


def task3_main_evaluation(cases: List[Dict[str, Any]]) -> None:
    print("\n=== TASK 3 — Main Evaluation Run ===")
    from run_system import run_system

    try:
        from baseline_a import run_baseline_a as baseline_a_fn
    except ImportError:
        from baseline_a import baseline_a_llm as baseline_a_fn

    try:
        from baseline_b import run_baseline_b as baseline_b_fn
    except ImportError:
        from baseline_b import baseline_b_llm as baseline_b_fn
    from engine.policy_engine import policy_checker
    from utils.config import POLICY_JSON_PATH
    from utils.openai_client import load_project_env

    load_project_env()

    with open(POLICY_JSON_PATH, "r", encoding="utf-8") as f:
        policies_doc = json.load(f)

    rows: List[Dict[str, Any]] = []
    total = len(cases)

    for idx, case in enumerate(cases, start=1):
        cid = case["id"]
        req = case["request"]
        patch = case.get("_memory_store_patch")
        gold = normalize_label(case["expected_decision"])

        # --- Pipeline ---
        try:
            out = run_system(req, memory_store_patch=patch)
            pipe_dec = normalize_label(out.get("decision"))
            pipe_policies = out.get("policy") or []
            if not isinstance(pipe_policies, list):
                pipe_policies = []
            pipe_expl = (out.get("explanation") or "")[:200]
            case_data = out.get("json") or {}
            pol_res = policy_checker(case_data, POLICY_JSON_PATH)
            skipped = pol_res.get("skipped_policies", [])
            val_errors: List[str] = []
        except Exception as e:
            print(f"  [pipeline error] {cid}: {e}")
            pipe_dec = "PIPELINE_ERROR"
            pipe_policies = []
            pipe_expl = str(e)[:200]
            skipped = []
            val_errors = [str(e)]

        # --- Baseline A ---
        try:
            base_a = normalize_label(baseline_a_fn(req))
        except Exception as e:
            print(f"  [baseline A error] {cid}: {e}")
            base_a = "BASELINE_A_ERROR"

        # --- Baseline B ---
        try:
            base_b = normalize_label(baseline_b_fn(req))
        except Exception as e:
            print(f"  [baseline B error] {cid}: {e}")
            base_b = "BASELINE_B_ERROR"

        rows.append(
            {
                "case_id": cid,
                "leave_type": case["leave_type"],
                "category": case["category"],
                "scenario": case["scenario"],
                "gold_label": gold,
                "pipeline_decision": pipe_dec,
                "baselineA_decision": base_a,
                "baselineB_decision": base_b,
                "pipeline_policies": pipe_policies,
                "pipeline_explanation": pipe_expl,
                "validation_errors": val_errors,
                "skipped_policies": skipped,
            }
        )
        if idx % 25 == 0 or idx == total:
            print(f"  ... {idx}/{total}")

    df = pd.DataFrame(rows)
    export_df = df.copy()
    export_df["pipeline_policies"] = export_df["pipeline_policies"].apply(
        lambda x: ",".join(x) if isinstance(x, list) else str(x)
    )
    export_df["skipped_policies"] = export_df["skipped_policies"].apply(
        lambda x: json.dumps(x, ensure_ascii=False) if x else ""
    )
    export_df.to_csv(RESULTS / "main_evaluation.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved: {RESULTS / 'main_evaluation.csv'}")

    gold = df["gold_label"].tolist()

    systems = {
        "pipeline": df["pipeline_decision"].tolist(),
        "baseline_a": df["baselineA_decision"].tolist(),
        "baseline_b": df["baselineB_decision"].tolist(),
    }

    # --- A) Overall accuracy ---
    print("\n--- A) Overall accuracy ---")
    acc_rows = []
    for name, preds in systems.items():
        correct = sum(1 for g, p in zip(gold, preds) if g == p)
        acc = correct / len(gold) if gold else 0.0
        acc_rows.append({"System": name, "Accuracy": f"{acc:.2%}", "Correct": correct, "n": len(gold)})
        print(f"  {name}: {acc:.2%} ({correct}/{len(gold)} correct)")
    print_table("", pd.DataFrame(acc_rows))

    metrics: Dict[str, Any] = {
        "overall_accuracy": {},
        "per_class": {},
        "macro_f1": {},
        "weighted_f1": {},
        "macro_avg": {},
        "weighted_avg": {},
    }

    # --- B) Classification reports ---
    print("\n--- B) classification_report (per system) ---")
    for name, preds in systems.items():
        y_t = gold
        y_p = [p if p in LABELS else "NONE" for p in preds]
        print(f"\n>>> {name}")
        print(classification_report(y_t, y_p, labels=LABELS, zero_division=0))
        report = classification_report(
            y_t, y_p, labels=LABELS, zero_division=0, output_dict=True
        )
        metrics["overall_accuracy"][name] = sum(1 for g, p in zip(y_t, y_p) if g == p) / len(y_t)
        metrics["macro_f1"][name] = macro_f1(y_t, y_p)
        metrics["weighted_f1"][name] = weighted_f1(y_t, y_p)
        metrics["per_class"][name] = {lbl: report.get(lbl, {}) for lbl in LABELS}
        metrics["macro_avg"][name] = report.get("macro avg", {})
        metrics["weighted_avg"][name] = report.get("weighted avg", {})

    # --- C) Confusion matrices ---
    print("\n--- C) Confusion matrices ---")
    for name, preds in systems.items():
        mask = [p != "ERROR" for p in preds]
        y_t = [g for g, m in zip(gold, mask) if m]
        y_p = [p for p, m in zip(preds, mask) if m]
        slug = name.replace(" ", "_")
        csv_p = FIGURES / f"confusion_matrix_{slug}.csv"
        png_p = FIGURES / f"confusion_matrix_{slug}.png"
        plot_confusion_matrix(y_t, y_p, f"Confusion matrix — {name}", csv_p, png_p)
        print(f"  Saved {csv_p.name} and {png_p.name}")

    # --- D) Per-scenario accuracy ---
    print("\n--- D) Per-scenario accuracy ---")
    scenario_rows = []
    for name, preds in zip(
        ["Pipeline", "Baseline A", "Baseline B"],
        [systems["pipeline"], systems["baseline_a"], systems["baseline_b"]],
    ):
        sub = df.copy()
        sub["pred"] = preds
        sub = sub[sub["pred"] != "ERROR"]
        def scen_acc(scen: str) -> str:
            s = sub[sub["scenario"] == scen]
            if s.empty:
                return "n/a"
            return f"{(s['gold_label'] == s['pred']).mean():.2%}"

        scenario_rows.append(
            {
                "System": name,
                "Clean Acc": scen_acc("clean"),
                "Memory Acc": scen_acc("memory"),
                "Noisy Acc": scen_acc("noisy"),
            }
        )
    print_table("", pd.DataFrame(scenario_rows))

    # --- E) Per-category accuracy (pipeline) ---
    print("\n--- E) Per-category accuracy (pipeline only) ---")
    pipe_df = df[df["pipeline_decision"] != "ERROR"].copy()
    pipe_df["bucket"] = pipe_df["category"].map(eval_category_bucket)
    cat_rows = []
    for bucket in EVAL_BUCKET_ORDER:
        grp = pipe_df[pipe_df["bucket"] == bucket]
        if grp.empty:
            cat_rows.append({"Category": bucket, "Count": 0, "Accuracy": "n/a"})
            continue
        acc = (grp["gold_label"] == grp["pipeline_decision"]).mean()
        cat_rows.append(
            {
                "Category": bucket,
                "Count": len(grp),
                "Accuracy": f"{acc:.2%}",
            }
        )
    print_table("", pd.DataFrame(cat_rows))

    metrics["per_scenario_accuracy"] = scenario_rows
    metrics["per_category_pipeline"] = cat_rows

    # --- Paired noise robustness ---
    try:
        from book_benchmark_cases import CASES as _RAW_CASES
    except Exception:
        _RAW_CASES = []

    clean_paired = df[df["case_id"].isin([f"TC_{i:03d}" for i in range(1, 26)])].copy()
    noisy_paired = df[df["scenario"] == "noisy"].copy()

    if _RAW_CASES:
        pair_map = {c.get("case_id"): c.get("pair_id") for c in _RAW_CASES if c.get("pair_id") is not None}
        noisy_paired["pair_id"] = noisy_paired["case_id"].map(pair_map)
        clean_paired["pair_id"] = clean_paired["case_id"].map(pair_map)

    noisy_paired = noisy_paired[noisy_paired["pair_id"].notna()]
    clean_paired = clean_paired[clean_paired["pair_id"].notna()]

    if not clean_paired.empty and not noisy_paired.empty:
        pairs = clean_paired.merge(
            noisy_paired,
            on="pair_id",
            suffixes=("_clean", "_noisy"),
            how="inner",
        )
        if len(pairs):
            nr = (pairs["pipeline_decision_clean"] == pairs["pipeline_decision_noisy"]).mean()
            print(
                f"\n--- Paired noise robustness (pipeline, same pair_id): {nr:.2%} ({len(pairs)} pairs) ---"
            )
            metrics["noise_robustness_paired"] = {"accuracy": float(nr), "n_pairs": int(len(pairs))}

    with open(RESULTS / "main_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {RESULTS / 'main_metrics.json'}")

    # Summary table
    summary = pd.DataFrame(
        [
            {
                "System": "Pipeline",
                "Accuracy": f"{metrics['overall_accuracy'].get('pipeline', 0):.2%}",
                "Macro F1": f"{metrics['macro_f1'].get('pipeline', 0):.4f}",
            },
            {
                "System": "Baseline A",
                "Accuracy": f"{metrics['overall_accuracy'].get('baseline_a', 0):.2%}",
                "Macro F1": f"{metrics['macro_f1'].get('baseline_a', 0):.4f}",
            },
            {
                "System": "Baseline B",
                "Accuracy": f"{metrics['overall_accuracy'].get('baseline_b', 0):.2%}",
                "Macro F1": f"{metrics['macro_f1'].get('baseline_b', 0):.4f}",
            },
        ]
    )
    print("\n=== Summary ===")
    print_table("System | Accuracy | Macro F1", summary)


def main() -> None:
    ensure_dirs()
    cases = load_benchmark_cases()
    print(
        f"Loaded {len(cases)} benchmark cases from book_benchmark_cases"
    )

    for task_fn, name in (
        (task1_smoke_tests, "TASK 1"),
        (lambda: task2_dataset_distribution(cases), "TASK 2"),
        (lambda: task3_main_evaluation(cases), "TASK 3"),
    ):
        try:
            task_fn()
        except Exception as e:
            print(f"\n!!! {name} failed: {e}")
            traceback.print_exc()

    print("\n=== batch1_eval.py complete ===")


if __name__ == "__main__":
    main()
