"""
Evaluation: agent (run_system) vs baseline_a vs baseline_b against authored expected_decision labels.

The benchmark has no structured gold_case_data: each row is natural language + expected_decision (+ notes).
Optional: compare agent outputs to deterministic engine on the agent's own extraction (sanity checks).

Usage:
  python evaluation.py

Environment: OPENAI_API_KEY (see .env). Optional: POLICY_JSON_PATH.
Optional: EVAL_CONSISTENCY_RUNS (default 10, minimum 10) — per-case stability: strict (all runs identical)
  and soft rate (fraction of runs matching the first run); both printed and exported.
Optional: EVAL_BENCHMARK_PART — unset or \"all\" = full 125 cases; \"1\" / \"2\" / \"3\" = row slices 1–50, 51–100, 101–125 (quick run).
Prints per-category agent accuracy (annual approve/reject, sick, maternity, hajj, casual, ambiguous/NMI; plus memory and contact/unpaid when present).
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from baseline_a import baseline_a_llm
from baseline_b import baseline_b_llm
from engine.expected_labels import expected_decision_and_policies
from evaluation_dataset import build_leave_evaluation_dataset
from run_system import run_system
from utils.config import POLICY_JSON_PATH


def normalize_decision(decision: Optional[str]) -> Optional[str]:
    if decision is None:
        return None
    value = str(decision).strip().upper()
    mapping = {
        "APPROVED": "APPROVE",
        "REJECTED": "REJECT",
        "TERMINATED": "REJECT",
        "ESCALATED": "ESCALATE",
    }
    return mapping.get(value, value)


def normalize_policy_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return sorted(str(x).strip() for x in val if str(x).strip())
    if isinstance(val, str) and val.strip():
        return sorted(p.strip() for p in val.split(",") if p.strip())
    return []


def _row_memory_patch(row: Any) -> Optional[Dict[str, Any]]:
    """Return per-row memory patch dict or None (handles NaN from CSV round-trips)."""
    if isinstance(row, pd.Series):
        v = row.get("_memory_store_patch", None)
    elif isinstance(row, dict):
        v = row.get("_memory_store_patch")
    else:
        v = getattr(row, "_memory_store_patch", None)
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if isinstance(v, dict):
        return v
    return None


def _case_for_engine(agent_json: Dict[str, Any], patch: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    case = dict(agent_json) if isinstance(agent_json, dict) else {}
    if patch:
        case["_memory_store_patch"] = patch
    return case


def print_per_category_agent_accuracy(df: pd.DataFrame) -> None:
    """Break down agent decision accuracy by leave type (more informative than overall accuracy alone)."""
    if "agent_decision_correct" not in df.columns:
        return
    cat = df["category"].fillna("").astype(str)
    # Primary breakdown (requested labels)
    groups: List[tuple[str, pd.Series]] = [
        ("Annual APPROVE:", cat == "annual_approve"),
        ("Annual REJECT:", cat == "annual_reject"),
        ("Sick ESCALATE:", cat == "sick_escalate"),
        ("Maternity:", cat.str.startswith("maternity", na=False)),
        ("Hajj:", cat.str.startswith("hajj", na=False)),
        ("Casual:", cat.str.startswith("casual", na=False)),
        ("Ambiguous/NMI:", cat == "ambiguous"),
    ]
    # Supplemental rows when the benchmark includes these categories
    extra: List[tuple[str, pd.Series]] = [
        ("Memory (MEM-RECORD):", cat == "memory_reject"),
        ("Contact / Unpaid:", cat.isin(["contact_escalate", "unpaid_escalate"])),
    ]
    print("\n=== Per-category accuracy (agent vs authored label) ===")
    for label, mask in groups + extra:
        sub = df[mask]
        n = len(sub)
        if n == 0:
            print(f"{label:24s}  —  (n=0)")
            continue
        acc = float(sub["agent_decision_correct"].mean())
        print(f"{label:24s}  {acc:6.2%}  (n={n})")


def save_evaluation_csv(export: pd.DataFrame, primary: str = "evaluation_results.csv") -> str:
    candidates = [
        primary,
        f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    ]
    last_err: Optional[BaseException] = None
    for path in candidates:
        try:
            export.to_csv(path, index=False, encoding="utf-8-sig")
            return path
        except PermissionError as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError("save_evaluation_csv: no candidates")


def save_evaluation_excel_colored(
    export: pd.DataFrame,
    csv_path: str,
    pass_col: str = "agent_decision_correct",
) -> Optional[str]:
    try:
        from openpyxl.styles import PatternFill
    except ImportError:
        return None

    if pass_col not in export.columns:
        return None

    green = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    red = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    ncols = len(export.columns)
    pi = list(export.columns).index(pass_col) + 1

    def fill_rows(ws):
        for r in range(2, len(export) + 2):
            ok = bool(ws.cell(row=r, column=pi).value)
            fill = green if ok else red
            for c in range(1, ncols + 1):
                ws.cell(row=r, column=c).fill = fill

    xlsx_path = str(Path(csv_path).with_suffix(".xlsx"))
    for path_try in (
        xlsx_path,
        str(
            Path(csv_path).parent
            / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        ),
    ):
        try:
            with pd.ExcelWriter(path_try, engine="openpyxl") as writer:
                export.to_excel(writer, index=False, sheet_name="results")
                fill_rows(writer.sheets["results"])
            return path_try
        except Exception:
            continue
    return None


def main() -> None:
    from utils.openai_client import load_project_env

    load_project_env()

    print("\n=== Employee Leave Evaluation ===")
    print(f"OPENAI_API_KEY: {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")

    with open(POLICY_JSON_PATH, "r", encoding="utf-8") as f:
        policies_doc: Dict[str, Any] = json.load(f)

    df = build_leave_evaluation_dataset()
    print(f"Loaded {len(df)} test cases (authored labels; no structured gold).")

    agent_results: List[Dict[str, Any]] = []
    baseline_a_results: List[str] = []
    baseline_b_results: List[str] = []

    print("\nRunning agent (run_system), baseline A, and baseline B on each input...")
    for idx, row in df.iterrows():
        inp = row["input"]
        patch = _row_memory_patch(row)
        out = run_system(inp, memory_store_patch=patch)
        agent_results.append(
            {
                "agent_decision": normalize_decision(out.get("decision")),
                "agent_policy": normalize_policy_list(out.get("policy")),
                "agent_json": out.get("json") or {},
                "agent_explanation": out.get("explanation") or "",
            }
        )

        try:
            baseline_a_results.append(normalize_decision(baseline_a_llm(inp)))
        except Exception as e:
            baseline_a_results.append(f"ERROR: {e}")

        try:
            baseline_b_results.append(normalize_decision(baseline_b_llm(inp, policies_doc)))
        except Exception as e:
            baseline_b_results.append(f"ERROR: {e}")

        if (idx + 1) % 25 == 0:
            print(f"  ... {idx + 1}/{len(df)}")

    agent_df = pd.DataFrame(agent_results)
    out_df = pd.concat([df.reset_index(drop=True), agent_df], axis=1)

    out_df["expected_decision_norm"] = out_df["expected_decision"].apply(normalize_decision)

    eng_decisions: List[Optional[str]] = []
    eng_policies: List[List[str]] = []
    for _, r in out_df.iterrows():
        patch = _row_memory_patch(r)
        case = _case_for_engine(r["agent_json"], patch)
        ed, ep = expected_decision_and_policies(case)
        eng_decisions.append(normalize_decision(ed))
        eng_policies.append(normalize_policy_list(ep))
    out_df["engine_decision_on_agent_json"] = eng_decisions
    out_df["engine_policy_on_agent_json"] = eng_policies

    out_df["baseline_a_decision"] = baseline_a_results
    out_df["baseline_b_decision"] = baseline_b_results

    ok_a = ~out_df["baseline_a_decision"].astype(str).str.startswith("ERROR")
    ok_b = ~out_df["baseline_b_decision"].astype(str).str.startswith("ERROR")
    out_df["baseline_a_correct"] = ok_a & (
        out_df["baseline_a_decision"] == out_df["expected_decision_norm"]
    )
    out_df["baseline_b_correct"] = ok_b & (
        out_df["baseline_b_decision"] == out_df["expected_decision_norm"]
    )

    out_df["agent_decision_correct"] = out_df["agent_decision"] == out_df["expected_decision_norm"]
    out_df["agent_decision_matches_engine_on_extraction"] = (
        out_df["agent_decision"] == out_df["engine_decision_on_agent_json"]
    )
    out_df["agent_policy_matches_engine_on_extraction"] = out_df.apply(
        lambda r: r["agent_policy"] == r["engine_policy_on_agent_json"], axis=1
    )

    agent_accuracy = float(out_df["agent_decision_correct"].mean())
    baseline_a_accuracy = float(out_df["baseline_a_correct"].mean())
    baseline_b_accuracy = float(out_df["baseline_b_correct"].mean())
    engine_decision_agree = float(out_df["agent_decision_matches_engine_on_extraction"].mean())
    engine_policy_agree = float(out_df["agent_policy_matches_engine_on_extraction"].mean())

    try:
        _env_runs = int(os.getenv("EVAL_CONSISTENCY_RUNS", "10"))
    except ValueError:
        _env_runs = 10
    consistency_runs = max(10, _env_runs)

    consistency_strict: List[bool] = []
    consistency_rates: List[float] = []
    consistency_n_agree: List[int] = []

    n_consistency = len(df)
    print(
        f"\nConsistency pass: {n_consistency} cases × {consistency_runs} runs "
        f"(decision-only; explanations skipped for speed)…"
    )
    for cidx, (_, row) in enumerate(df.iterrows(), start=1):
        inp = row["input"]
        patch = _row_memory_patch(row)
        decisions: List[Optional[str]] = []
        for _ in range(consistency_runs):
            d = normalize_decision(
                run_system(inp, memory_store_patch=patch, skip_explanation=True).get(
                    "decision"
                )
            )
            decisions.append(d)
        if cidx == 1 or cidx % 10 == 0 or cidx == n_consistency:
            print(f"  … consistency progress {cidx}/{n_consistency}")
        first = decisions[0] if decisions else None
        n_agree = sum(1 for d in decisions if d == first) if decisions else 0
        rate = (n_agree / len(decisions)) if decisions else 0.0
        consistency_n_agree.append(n_agree)
        consistency_rates.append(rate)
        consistency_strict.append(len(set(decisions)) == 1)

    consistency_score = sum(consistency_strict) / len(consistency_strict) if consistency_strict else 0.0
    mean_consistency_rate = (
        sum(consistency_rates) / len(consistency_rates) if consistency_rates else 0.0
    )

    out_df["agent_consistency_strict"] = consistency_strict
    out_df["agent_consistency_rate"] = consistency_rates
    out_df["agent_consistency_n_agree"] = consistency_n_agree
    out_df["agent_consistency_n_runs"] = consistency_runs
    out_df["agent_consistency_label"] = [f"{n}/{consistency_runs}" for n in consistency_n_agree]

    clean_df = out_df[out_df["is_noisy"] == False][["pair_id", "agent_decision"]].rename(
        columns={"agent_decision": "clean_decision"}
    )
    noisy_df = out_df[out_df["is_noisy"] == True][["pair_id", "agent_decision"]].rename(
        columns={"agent_decision": "noisy_decision"}
    )
    noise_df = clean_df.merge(noisy_df, on="pair_id", how="inner")
    noise_robustness = (
        float((noise_df["clean_decision"] == noise_df["noisy_decision"]).mean())
        if len(noise_df)
        else 0.0
    )

    diff_a = agent_accuracy - baseline_a_accuracy
    diff_b = agent_accuracy - baseline_b_accuracy

    print("\n=== Metrics ===")
    print(f"Agent decision accuracy (vs authored label): {agent_accuracy:.2%}")
    print(f"Baseline A (raw LLM):                        {baseline_a_accuracy:.2%}")
    print(f"Baseline B (policies JSON + request, 1 call): {baseline_b_accuracy:.2%}")
    print(
        f"Agent decision vs engine(agent extraction):   {engine_decision_agree:.2%} "
        "(sanity: pipeline vs policy_checker+memory on same JSON + patch)"
    )
    print(
        f"Agent policy IDs vs engine(agent extraction): {engine_policy_agree:.2%} "
        "(sanity: should be ~100% when extraction matches engine inputs)"
    )
    print_per_category_agent_accuracy(out_df)
    print(
        f"Agent consistency — strict ({consistency_runs} runs all identical): {consistency_score:.2%}"
    )
    print(
        f"Agent consistency — mean agreement with 1st run: {mean_consistency_rate:.2%} "
        f"(per-row: agent_consistency_rate, agent_consistency_n_agree / {consistency_runs}, agent_consistency_strict)"
    )
    if len(noise_df):
        print(f"Noise robustness (paired): {noise_robustness:.2%} ({len(noise_df)} pairs)")
    else:
        print("Noise robustness (paired): N/A — no paired noisy/clean rows in this benchmark.")

    mis_agent = out_df[~out_df["agent_decision_correct"]]
    mis_a = out_df[~out_df["baseline_a_correct"]]
    mis_b = out_df[~out_df["baseline_b_correct"]]

    print("\n=== Misclassified — Agent (show up to 20) ===")
    if mis_agent.empty:
        print("None.")
    else:
        for _, r in mis_agent.head(20).iterrows():
            print("-" * 70)
            print(f"id={r['case_id']}  expected={r['expected_decision_norm']}  got={r['agent_decision']}")
            print(f"input: {r['input'][:200]}...")

    print("\n=== Misclassified — Baseline A (show up to 20) ===")
    if mis_a.empty:
        print("None.")
    else:
        for _, r in mis_a.head(20).iterrows():
            print("-" * 70)
            print(f"id={r['case_id']}  expected={r['expected_decision_norm']}  got={r['baseline_a_decision']}")
            print(f"input: {r['input'][:200]}...")

    print("\n=== Misclassified — Baseline B (show up to 20) ===")
    if mis_b.empty:
        print("None.")
    else:
        for _, r in mis_b.head(20).iterrows():
            print("-" * 70)
            print(f"id={r['case_id']}  expected={r['expected_decision_norm']}  got={r['baseline_b_decision']}")
            print(f"input: {r['input'][:200]}...")

    export = out_df.copy()
    export["agent_json"] = export["agent_json"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    export["engine_policy_on_agent_json"] = export["engine_policy_on_agent_json"].apply(lambda x: ",".join(x))
    export["agent_policy"] = export["agent_policy"].apply(lambda x: ",".join(x))
    if "_memory_store_patch" in export.columns:
        export["_memory_store_patch"] = export["_memory_store_patch"].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else ("")
        )

    saved_path = save_evaluation_csv(export)
    print(f"\nSaved results to {saved_path}")
    if saved_path != "evaluation_results.csv":
        print("(Primary evaluation_results.csv was not writable — close Excel or other locks and retry.)")

    xlsx_path = save_evaluation_excel_colored(export, saved_path)
    if xlsx_path:
        print(f"Saved colored workbook (green=agent correct, red=agent wrong): {xlsx_path}")
    else:
        print("\nSkipped colored Excel export — install openpyxl: pip install openpyxl")


if __name__ == "__main__":
    main()
