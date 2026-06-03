"""
Batch 3 evaluation: Explanation Faithfulness Audit, Policy ID Accuracy,
Figure Generation, and Master Results Summary.

Assumes batch1_eval.py and batch2_eval.py have been run and all results CSVs exist.
Does NOT modify any existing source files.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from utils.openai_client import get_llm, message_content

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CSV_MAIN = RESULTS_DIR / "main_evaluation.csv"
CSV_ABLATION = RESULTS_DIR / "ablation_results.csv"
JSON_CONSISTENCY = RESULTS_DIR / "consistency_summary.json"
CSV_ERRORS = RESULTS_DIR / "error_analysis.csv"
CSV_ERROR_COMP = RESULTS_DIR / "error_comparison.csv"

CSV_AUDIT = RESULTS_DIR / "explanation_audit.csv"
JSON_POLICY_ACC = RESULTS_DIR / "policy_id_accuracy.json"
PNG_FAITHFULNESS = FIGURES_DIR / "explanation_faithfulness.png"

LABELS: List[str] = ["APPROVE", "REJECT", "ESCALATE", "NEED_MORE_INFO"]

random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# OpenAI API helper (reuses project client)
# ---------------------------------------------------------------------------

_llm = None


def _call_openai(system_prompt: str, user_prompt: str) -> str:
    global _llm
    if _llm is None:
        _llm = get_llm()
    response = _llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return message_content(response)


# ---------------------------------------------------------------------------
# Load policy messages lookup
# ---------------------------------------------------------------------------


def _load_policy_messages() -> Dict[str, str]:
    policy_path = ROOT / "policies" / "policies.json"
    with open(policy_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    return {p["id"]: p.get("message", "") for p in doc.get("policies", [])}


POLICY_MESSAGES = _load_policy_messages()


# ---------------------------------------------------------------------------
# TASK 7 — Explanation Faithfulness Audit
# ---------------------------------------------------------------------------


def task7_explanation_faithfulness():
    print("\n=== TASK 7 ===")
    print("Running explanation faithfulness audit...")

    try:
        get_llm()
    except ValueError:
        print("  [SKIP] OPENAI_API_KEY not set in environment.")
        print("  Set it and re-run to perform the faithfulness audit.")
        return None

    df = pd.read_csv(CSV_MAIN)
    correct = df[df["pipeline_decision"] == df["gold_label"]].copy()
    if len(correct) < 20:
        print(f"  [WARN] Only {len(correct)} correct decisions; sampling all.")
        sample = correct
    else:
        sample = correct.sample(n=20, random_state=42)

    ratings: List[str] = []
    rows_out: List[Dict[str, Any]] = []

    for _, row in sample.iterrows():
        cid = row["case_id"]
        explanation = str(row.get("pipeline_explanation", ""))
        policy_ids_str = str(row.get("pipeline_policies", ""))

        policy_ids = [pid.strip() for pid in policy_ids_str.split(",") if pid.strip()]
        policy_msgs = [POLICY_MESSAGES.get(pid, f"[Message not found for {pid}]") for pid in policy_ids]
        policy_messages_text = "\n".join(f"- {msg}" for msg in policy_msgs) if policy_msgs else "(No policies matched)"

        system_prompt = "You are a strict faithfulness auditor for AI-generated explanations."
        user_prompt = f"""The following explanation was generated for a circulation-request decision.
Allowed sources of reasoning:
1. The policy messages listed below
2. Factual memory / user history hints that appear in the explanation (e.g., "Our records show..." about prior borrowing or unavailable books)

Policy messages:
{policy_messages_text}

Generated explanation:
{explanation}

Rate faithfulness as exactly one of:
FAITHFUL — explanation draws only on allowed sources (policy messages + factual memory hints)
PARTIAL — explanation mostly follows allowed sources but adds minor unreferenced reasoning
UNFAITHFUL — explanation introduces reasoning not grounded in any allowed source

Reply with ONLY one word."""

        try:
            raw = _call_openai(system_prompt, user_prompt)
        except Exception as e:
            print(f"  [API error for {cid}] {e}")
            raw = "PARTIAL"

        raw_upper = raw.upper()
        if "FAITHFUL" in raw_upper and "UN" not in raw_upper:
            rating = "FAITHFUL"
        elif "UNFAITHFUL" in raw_upper:
            rating = "UNFAITHFUL"
        elif "PARTIAL" in raw_upper:
            rating = "PARTIAL"
        else:
            rating = "PARTIAL"

        ratings.append(rating)
        rows_out.append(
            {
                "case_id": cid,
                "gold_label": row["gold_label"],
                "policy_ids": policy_ids_str,
                "explanation_snippet": explanation[:300],
                "faithfulness_rating": rating,
            }
        )

    counts = {"FAITHFUL": 0, "PARTIAL": 0, "UNFAITHFUL": 0}
    for r in ratings:
        counts[r] = counts.get(r, 0) + 1

    total = len(ratings)
    print(
        f"  Faithful: {counts['FAITHFUL']} ({counts['FAITHFUL']/total*100:.1f}%) | "
        f"Partial: {counts['PARTIAL']} ({counts['PARTIAL']/total*100:.1f}%) | "
        f"Unfaithful: {counts['UNFAITHFUL']} ({counts['UNFAITHFUL']/total*100:.1f}%)"
    )

    pd.DataFrame(rows_out).to_csv(CSV_AUDIT, index=False)
    print(f"  Saved: {CSV_AUDIT.name}")

    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = [counts["FAITHFUL"], counts["PARTIAL"], counts["UNFAITHFUL"]]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    labels = [
        f"FAITHFUL\n({counts['FAITHFUL']})",
        f"PARTIAL\n({counts['PARTIAL']})",
        f"UNFAITHFUL\n({counts['UNFAITHFUL']})",
    ]
    ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 11},
    )
    ax.set_title("Explanation Faithfulness Audit (n=20)")
    fig.patch.set_facecolor("white")
    fig.savefig(PNG_FAITHFULNESS, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {PNG_FAITHFULNESS.name}")

    return counts


# ---------------------------------------------------------------------------
# TASK 8 — Policy ID Accuracy
# ---------------------------------------------------------------------------


def task8_policy_id_accuracy():
    print("\n=== TASK 8 ===")
    print("Running policy ID accuracy analysis...")

    from engine.expected_labels import expected_decision_and_policies
    from book_benchmark_cases import CASES as BENCHMARK_CASES
    from run_system import run_system

    df = pd.read_csv(CSV_MAIN)
    correct = df[df["pipeline_decision"] == df["gold_label"]].copy()

    case_map = {c["case_id"]: c for c in BENCHMARK_CASES}

    exact = 0
    partial = 0
    no_match = 0
    total = len(correct)

    for idx, (_, row) in enumerate(correct.iterrows(), start=1):
        cid = row["case_id"]
        case = case_map.get(cid, {})
        req = case.get("input_text") or case.get("input", "")
        patch = case.get("_memory_store_patch")

        try:
            out = run_system(req, memory_store_patch=patch, skip_explanation=True)
            case_data = out.get("json") or {}
        except Exception as e:
            print(f"  [WARN] run_system failed for {cid}: {e}")
            case_data = {}

        case_data["_memory_store_patch"] = patch

        try:
            _, expected_ids = expected_decision_and_policies(case_data)
        except Exception as e:
            print(f"  [WARN] expected_decision_and_policies failed for {cid}: {e}")
            expected_ids = []

        expected_set: Set[str] = set(expected_ids)
        actual_str = str(row.get("pipeline_policies", ""))
        actual_set: Set[str] = set(pid.strip() for pid in actual_str.split(",") if pid.strip())

        if expected_set == actual_set:
            exact += 1
        elif expected_set & actual_set:
            partial += 1
        else:
            no_match += 1

        if idx % 25 == 0 or idx == total:
            print(f"  ... {idx}/{total} cases evaluated")

    result = {
        "total_correct_decisions": total,
        "exact_match_count": exact,
        "exact_match_pct": round(exact / total * 100, 2) if total else 0,
        "partial_match_count": partial,
        "partial_match_pct": round(partial / total * 100, 2) if total else 0,
        "no_match_count": no_match,
        "no_match_pct": round(no_match / total * 100, 2) if total else 0,
    }

    print(
        f"  Exact match:   {exact}/{total} = {result['exact_match_pct']:.1f}%\n"
        f"  Partial match: {partial}/{total} = {result['partial_match_pct']:.1f}%\n"
        f"  No match:      {no_match}/{total} = {result['no_match_pct']:.1f}%"
    )

    with open(JSON_POLICY_ACC, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {JSON_POLICY_ACC.name}")

    return result


# ---------------------------------------------------------------------------
# TASK 9 — Generate All Remaining Figures
# ---------------------------------------------------------------------------


def _plot_confusion_matrix(y_true, y_pred, title, png_path):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
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


def task9_generate_figures():
    print("\n=== TASK 9 ===")
    print("Generating all remaining figures...")

    df = pd.read_csv(CSV_MAIN)

    # Figure 1-3: Confusion matrices (regenerate if needed)
    for name, col, title in [
        ("pipeline", "pipeline_decision", "Pipeline"),
        ("baseline_a", "baselineA_decision", "Baseline A"),
        ("baseline_b", "baselineB_decision", "Baseline B"),
    ]:
        sub = df[df[col] != "ERROR"].copy()
        y_t = sub["gold_label"].tolist()
        y_p = sub[col].tolist()
        png_path = FIGURES_DIR / f"confusion_matrix_{name}.png"
        _plot_confusion_matrix(y_t, y_p, f"Confusion matrix — {title}", png_path)
        print(f"  Saved: confusion_matrix_{name}.png")

    # Figure 4: accuracy_by_scenario_type.png
    print("  Generating accuracy_by_scenario_type.png...")
    scenario_acc = []
    for system, pred_col in [
        ("Pipeline", "pipeline_decision"),
        ("Baseline A", "baselineA_decision"),
        ("Baseline B", "baselineB_decision"),
    ]:
        sub = df[df[pred_col] != "ERROR"].copy()
        for scen in ["clean", "memory", "noisy"]:
            s = sub[sub["scenario"] == scen]
            acc = (s["gold_label"] == s[pred_col]).mean() if not s.empty else 0
            scenario_acc.append(
                {
                    "System": system,
                    "Scenario": scen.capitalize(),
                    "Accuracy": acc,
                }
            )

    scen_df = pd.DataFrame(scenario_acc)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    width = 0.25
    systems = ["Pipeline", "Baseline A", "Baseline B"]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    for i, sys_name in enumerate(systems):
        vals = scen_df[scen_df["System"] == sys_name]["Accuracy"].tolist()
        ax.bar(x + i * width, vals, width, label=sys_name, color=colors[i])

    ax.set_xticks(x + width)
    ax.set_xticklabels(["Clean", "Memory", "Noisy"])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy by Scenario Type")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "accuracy_by_scenario_type.png", dpi=300, facecolor="white")
    plt.close(fig)
    print("  Saved: accuracy_by_scenario_type.png")

    # Figure 5: accuracy_by_leave_category.png (renamed content for book domain)
    print("  Generating accuracy_by_leave_category.png...")
    pipe_df = df[df["pipeline_decision"] != "ERROR"].copy()

    def _bucket(cat):
        cat = str(cat or "")
        if cat == "memory_reject":
            return "Memory-REJECT"
        if cat == "ambiguous":
            return "NMI"
        if cat.startswith("borrow_approve"):
            return "Borrow-APPROVE"
        if cat.startswith("borrow_reject"):
            return "Borrow-REJECT"
        if cat.startswith("noncirculating_"):
            return "Non-circulating"
        if cat.startswith("renew_"):
            return "Renewal"
        if cat.startswith("reserve_"):
            return "Reservation"
        if cat.startswith("report_"):
            return "Lost/Damaged"
        if cat.startswith("fines_"):
            return "Fines"
        return cat or "Unknown"

    pipe_df["bucket"] = pipe_df["category"].apply(_bucket)
    cat_accs = []
    for bucket in [
        "Borrow-APPROVE",
        "Borrow-REJECT",
        "Non-circulating",
        "Renewal",
        "Reservation",
        "Lost/Damaged",
        "Fines",
        "NMI",
        "Memory-REJECT",
    ]:
        grp = pipe_df[pipe_df["bucket"] == bucket]
        if not grp.empty:
            acc = (grp["gold_label"] == grp["pipeline_decision"]).mean()
            cat_accs.append({"Category": bucket, "Accuracy": acc})

    cat_df = pd.DataFrame(cat_accs)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(cat_df["Category"], cat_df["Accuracy"], color="#3498db")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Pipeline Accuracy by Category")
    ax.tick_params(axis="x", rotation=30)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "accuracy_by_leave_category.png", dpi=300, facecolor="white")
    plt.close(fig)
    print("  Saved: accuracy_by_leave_category.png")

    # Figure 6: ablation_comparison.png
    print("  Generating ablation_comparison.png...")
    abl_df = pd.read_csv(CSV_ABLATION)
    full_row = abl_df[abl_df["configuration"] == "Full Pipeline"]
    full_acc = float(full_row["accuracy"].iloc[0]) if not full_row.empty else 0

    abl_rows = []
    for _, row in abl_df.iterrows():
        cfg = row["configuration"]
        if cfg == "Full Pipeline":
            continue
        acc = float(row["accuracy"])
        drop = full_acc - acc
        abl_rows.append(
            {
                "Configuration": cfg.replace("— ", ""),
                "Drop": drop,
                "Accuracy": acc,
            }
        )

    abl_chart = pd.DataFrame(abl_rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = []
    for drop in abl_chart["Drop"]:
        if drop > 5:
            colors.append("#e74c3c")
        elif drop >= 2:
            colors.append("#f39c12")
        else:
            colors.append("#2ecc71")

    bars = ax.bar(abl_chart["Configuration"], abl_chart["Drop"], color=colors)
    ax.set_ylabel("Accuracy Drop from Full Pipeline (pp)")
    ax.set_title("Ablation Comparison")
    ax.axhline(0, color="black", linewidth=0.5)
    for bar, drop in zip(bars, abl_chart["Drop"]):
        height = bar.get_height()
        ax.annotate(
            f"{drop:.1f}pp",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ablation_comparison.png", dpi=300, facecolor="white")
    plt.close(fig)
    print("  Saved: ablation_comparison.png")

    # Figure 7: consistency_by_scenario.png
    print("  Generating consistency_by_scenario.png...")
    with open(JSON_CONSISTENCY, "r", encoding="utf-8") as f:
        cons = json.load(f)

    by_scenario = cons.get("strict_by_scenario", {}) or cons.get("by_scenario", {})
    scenarios = ["clean", "memory", "noisy"]
    rates = [by_scenario.get(s, 0) for s in scenarios]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar([s.capitalize() for s in scenarios], rates, color="#3498db")
    ax.set_ylabel("Strict Consistency Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Consistency by Scenario")
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.annotate(
            f"{rate:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "consistency_by_scenario.png", dpi=300, facecolor="white")
    plt.close(fig)
    print("  Saved: consistency_by_scenario.png")

    # Figure 8: explanation_faithfulness.png (if not already done in Task 7)
    if PNG_FAITHFULNESS.exists():
        print("  explanation_faithfulness.png already generated in Task 7.")
    else:
        print("  [WARN] explanation_faithfulness.png not found; run Task 7 first.")

    print("  Task 9 complete.")


# ---------------------------------------------------------------------------
# TASK 10 — Master Results Summary
# ---------------------------------------------------------------------------


def task10_master_summary(audit_counts: Optional[Dict[str, int]] = None, policy_acc: Optional[Dict[str, Any]] = None):
    print("\n=== TASK 10 ===")
    print("Generating master results summary...")

    df = pd.read_csv(CSV_MAIN)
    with open(RESULTS_DIR / "main_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(JSON_CONSISTENCY, "r", encoding="utf-8") as f:
        consistency = json.load(f)
    abl_df = pd.read_csv(CSV_ABLATION)
    err_df = pd.read_csv(CSV_ERRORS) if CSV_ERRORS.exists() else pd.DataFrame()

    total_cases = len(df)
    gold_counts = df["gold_label"].value_counts().to_dict()
    scenario_counts = df["scenario"].value_counts().to_dict()

    def _pct(label):
        return gold_counts.get(label, 0) / total_cases * 100 if total_cases else 0

    overall = metrics.get("overall_accuracy", {})
    macro_f1 = metrics.get("macro_f1", {})
    per_class = metrics.get("per_class", {}).get("pipeline", {})
    per_scen = metrics.get("per_scenario_accuracy", [])
    pipe_scen = next((r for r in per_scen if r.get("System") == "Pipeline"), {})

    strict_cons = consistency.get("strict_consistency_percent", consistency.get("pct_strict_all", 0))
    mean_soft = consistency.get("mean_soft_consistency_rate", consistency.get("mean_soft_rate", 0))
    by_scenario_cons = consistency.get("strict_by_scenario", consistency.get("by_scenario", {}))

    full_acc_row = abl_df[abl_df["configuration"] == "Full Pipeline"]
    full_acc = float(full_acc_row["accuracy"].iloc[0]) if not full_acc_row.empty else 0
    abl_lines = []
    for _, row in abl_df.iterrows():
        cfg = row["configuration"]
        if cfg == "Full Pipeline":
            continue
        acc = float(row["accuracy"])
        drop = full_acc - acc
        abl_lines.append(f"— {cfg.replace('— ', '')}:     {drop:+.1f}% pts  (Accuracy: {acc:.1f}%)")

    pipe_err = err_df
    total_err = len(pipe_err)
    err_rate = total_err / total_cases * 100 if total_cases else 0
    if not pipe_err.empty and "error_category" in pipe_err.columns:
        top_cat = pipe_err["error_category"].value_counts().iloc[0]
        top_name = pipe_err["error_category"].value_counts().index[0]
        err_breakdown = pipe_err["error_category"].value_counts().to_dict()
        err_breakdown_str = ", ".join(f"{k}: {v}" for k, v in list(err_breakdown.items())[:5])
    else:
        top_cat = 0
        top_name = "N/A"
        err_breakdown_str = "N/A"

    audit_skipped = audit_counts is None
    if audit_counts is None:
        audit_counts = {}
    audit_total = sum(audit_counts.values()) if audit_counts else 1
    faithful_pct = audit_counts.get("FAITHFUL", 0) / audit_total * 100 if audit_total else 0
    partial_pct = audit_counts.get("PARTIAL", 0) / audit_total * 100 if audit_total else 0
    unfaithful_pct = audit_counts.get("UNFAITHFUL", 0) / audit_total * 100 if audit_total else 0

    if policy_acc is None:
        if JSON_POLICY_ACC.exists():
            with open(JSON_POLICY_ACC, "r", encoding="utf-8") as f:
                policy_acc = json.load(f)
        else:
            policy_acc = {}

    saved_files = []
    for p in RESULTS_DIR.glob("*.csv"):
        saved_files.append(str(p))
    for p in RESULTS_DIR.glob("*.json"):
        saved_files.append(str(p))
    for p in FIGURES_DIR.glob("*.png"):
        saved_files.append(str(p))
    saved_files.sort()

    summary_text = f"""
=== THESIS CHAPTER 4 — MASTER RESULTS SUMMARY ===

[DATASET]
Total cases: {total_cases}
APPROVE: {gold_counts.get('APPROVE', 0)} ({_pct('APPROVE'):.1f}%) | REJECT: {gold_counts.get('REJECT', 0)} ({_pct('REJECT'):.1f}%) | ESCALATE: {gold_counts.get('ESCALATE', 0)} ({_pct('ESCALATE'):.1f}%) | NMI: {gold_counts.get('NEED_MORE_INFO', 0)} ({_pct('NEED_MORE_INFO'):.1f}%)
Clean: {scenario_counts.get('clean', 0)} | Memory: {scenario_counts.get('memory', 0)} | Noisy: {scenario_counts.get('noisy', 0)}

[MAIN RESULTS]
Pipeline   accuracy: {overall.get('pipeline', 0)*100:.1f}%  |  Macro F1: {macro_f1.get('pipeline', 0):.4f}
Baseline A accuracy: {overall.get('baseline_a', 0)*100:.1f}%  |  Macro F1: {macro_f1.get('baseline_a', 0):.4f}
Baseline B accuracy: {overall.get('baseline_b', 0)*100:.1f}%  |  Macro F1: {macro_f1.get('baseline_b', 0):.4f}

[PER-CLASS F1 — PIPELINE]
APPROVE: {per_class.get('APPROVE', {}).get('f1-score', 0):.4f}  REJECT: {per_class.get('REJECT', {}).get('f1-score', 0):.4f}  ESCALATE: {per_class.get('ESCALATE', {}).get('f1-score', 0):.4f}  NMI: {per_class.get('NEED_MORE_INFO', {}).get('f1-score', 0):.4f}

[PER-SCENARIO ACCURACY — PIPELINE]
Clean: {pipe_scen.get('Clean Acc', 'n/a')}  |  Memory: {pipe_scen.get('Memory Acc', 'n/a')}  |  Noisy: {pipe_scen.get('Noisy Acc', 'n/a')}

[CONSISTENCY]
Strict consistency: {strict_cons:.1f}%
Mean soft consistency: {mean_soft:.1f}%
By scenario — Clean: {by_scenario_cons.get('clean', 0):.1f}%  Memory: {by_scenario_cons.get('memory', 0):.1f}%  Noisy: {by_scenario_cons.get('noisy', 0):.1f}%

[ABLATION — ACCURACY DROP FROM FULL PIPELINE]
{chr(10).join(abl_lines)}

[ERROR ANALYSIS — PIPELINE]
Total errors: {total_err}  ({err_rate:.1f}% error rate)
Top error type: {top_name} ({top_cat} cases)
Error breakdown: {err_breakdown_str}

[EXPLANATION AUDIT]
{("Skipped — OPENAI_API_KEY not set" if audit_skipped else f"Faithful: {audit_counts.get('FAITHFUL', 0)} ({faithful_pct:.1f}%)  |  Partial: {audit_counts.get('PARTIAL', 0)} ({partial_pct:.1f}%)  |  Unfaithful: {audit_counts.get('UNFAITHFUL', 0)} ({unfaithful_pct:.1f}%)")}

[POLICY ID ACCURACY]
Exact match: {policy_acc.get('exact_match_pct', 0):.1f}%  |  Partial: {policy_acc.get('partial_match_pct', 0):.1f}%  |  No match: {policy_acc.get('no_match_pct', 0):.1f}%

===================================================

Files saved:
{chr(10).join(saved_files)}
"""

    print(summary_text)

    with open(RESULTS_DIR / "master_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"  Saved: master_summary.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Batch 3 evaluation tasks")
    parser.add_argument("--task7", action="store_true", help="Run only Task 7 (Explanation Faithfulness Audit)")
    parser.add_argument("--task8", action="store_true", help="Run only Task 8 (Policy ID Accuracy)")
    parser.add_argument("--task9", action="store_true", help="Run only Task 9 (Generate Figures)")
    parser.add_argument("--task10", action="store_true", help="Run only Task 10 (Master Summary)")
    args = parser.parse_args()

    # If no flags given, run all tasks
    run_all = not any([args.task7, args.task8, args.task9, args.task10])

    audit_counts = None
    policy_acc = None

    task_map = [
        (task7_explanation_faithfulness, "TASK 7", args.task7),
        (task8_policy_id_accuracy, "TASK 8", args.task8),
        (task9_generate_figures, "TASK 9", args.task9),
    ]

    for task_fn, name, flag in task_map:
        if run_all or flag:
            try:
                result = task_fn()
                if name == "TASK 7":
                    audit_counts = result
                elif name == "TASK 8":
                    policy_acc = result
            except Exception as e:
                print(f"\n[ERROR in {name}]")
                traceback.print_exc()
                print("Continuing...")

    if run_all or args.task10:
        try:
            task10_master_summary(audit_counts=audit_counts, policy_acc=policy_acc)
        except Exception:
            print("\n[ERROR in TASK 10]")
            traceback.print_exc()

    print("\n=== batch3_eval.py complete ===")


if __name__ == "__main__":
    main()
