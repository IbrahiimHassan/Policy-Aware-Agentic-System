from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
EVAL_CSV = RESULTS / "main_evaluation.csv"


def load_memory_patches():
    try:
        from leave_benchmark_cases import CASES as RAW
    except Exception:
        RAW = []
    mp = {c.get("case_id"): c.get("_memory_store_patch") for c in RAW}
    return mp


def main():
    if not EVAL_CSV.exists():
        print("Evaluation CSV not found:", EVAL_CSV)
        return
    df = pd.read_csv(EVAL_CSV, dtype=str)
    df.fillna("", inplace=True)

    mem_map = load_memory_patches()
    df["_memory_store_patch"] = df["case_id"].map(mem_map)

    # Filter: Baseline B correct, pipeline incorrect
    mask = (df["baselineB_decision"] == df["gold_label"]) & (df["pipeline_decision"] != df["gold_label"])
    out = df[mask].copy()

    if out.empty:
        print("No disagreements where Baseline B is correct and pipeline is wrong.")
        return

    cols = [
        "case_id",
        "request",
        "category",
        "gold_label",
        "pipeline_decision",
        "baselineB_decision",
        "pipeline_policies",
        "skipped_policies",
        "validation_errors",
        "_memory_store_patch",
    ]

    # Attempt to include the original request text: if not present, try reading from leave_benchmark_cases
    if "request" not in out.columns:
        try:
            from leave_benchmark_cases import CASES as RAW
            req_map = {c.get("case_id"): c.get("input") for c in RAW}
            out["request"] = out["case_id"].map(req_map)
        except Exception:
            out["request"] = ""

    out = out[[c for c in cols if c in out.columns]]
    out_path = RESULTS / "disagreement_baselineB_vs_pipeline.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out)} disagreement rows to {out_path}")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
