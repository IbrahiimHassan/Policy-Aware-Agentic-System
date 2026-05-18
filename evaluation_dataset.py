"""
125 benchmark rows from ``leave_benchmark_cases.CASES`` (or a slice via EVAL_BENCHMARK_PART).

Set EVAL_BENCHMARK_PART: unset / ``all`` → 125 rows; ``1`` / ``2`` / ``3`` → first 50, next 50,
last 25 (quick runs).

Optional per-case keys in each case dict:
  - ``is_noisy`` — noisy robustness slice
  - ``_memory_store_patch`` — merged into memory for that row only (run_system / consistency)

There is no structured gold_case_data: ``expected_decision`` is the authored label only.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd

from leave_benchmark_cases import CASES as _ALL


def _all_cases() -> List[Dict[str, Any]]:
    part = os.environ.get("EVAL_BENCHMARK_PART", "all").strip().lower()
    if part == "1":
        return list(_ALL[:50])
    if part == "2":
        return list(_ALL[50:100])
    if part == "3":
        return list(_ALL[100:125])
    return list(_ALL)


def build_leave_evaluation_dataset() -> pd.DataFrame:
    cases = _all_cases()
    rows: List[Dict[str, Any]] = []
    for tc in cases:
        patch = tc.get("_memory_store_patch")
        rows.append(
            {
                "case_id": tc["case_id"],
                "pair_id": tc.get("pair_id", tc["case_id"]),
                "is_noisy": bool(tc.get("is_noisy", False)),
                "category": tc["category"],
                "input": tc["input"],
                "expected_decision": tc["expected_decision"],
                "notes": tc.get("notes", ""),
                "_memory_store_patch": patch if patch is not None else None,
            }
        )
    df = pd.DataFrame(rows)
    if len(df) != len(cases):
        raise RuntimeError(f"Dataset row count mismatch: built {len(df)}, cases {len(cases)}")
    return df
