"""
Smoke tests for policy_checker against policies/policies.json (Egypt leave policy).
Uses structured case_data only — no LLM — so runs offline.
"""
import json

from engine.policy_engine import policy_checker
from utils.config import POLICY_JSON_PATH

TESTS = [
    {
        "label": "AL-01 annual under 6 months",
        "case_data": {"leave_type": "annual", "employment_duration_months": 3, "days_requested": 5},
        "expected_ids": ["AL-01"],
    },
    {
        "label": "AL-02 annual 6–11 months within 15 days",
        "case_data": {"leave_type": "annual", "employment_duration_months": 8, "days_requested": 5},
        "expected_ids": ["AL-02"],
    },
    {
        "label": "AL-03B annual 12+ months over 21 days",
        "case_data": {"leave_type": "annual", "employment_duration_months": 24, "days_requested": 31},
        "expected_ids": ["AL-03B"],
    },
    {
        "label": "AL-03B twelve months 30 days over mid cap",
        "case_data": {"leave_type": "annual", "employment_duration_months": 12, "days_requested": 30},
        "expected_ids": ["AL-03B"],
    },
    {
        "label": "AL-03 twelve months 21 days OK",
        "case_data": {"leave_type": "annual", "employment_duration_months": 12, "days_requested": 21},
        "expected_ids": ["AL-03"],
    },
    {
        "label": "SP-02 casual over 3 days",
        "case_data": {"leave_type": "casual", "employment_duration_months": 0, "days_requested": 4},
        "expected_ids": ["SP-02"],
    },
    {
        "label": "SP-01 casual 3 days OK",
        "case_data": {"leave_type": "casual", "employment_duration_months": 0, "days_requested": 3},
        "expected_ids": ["SP-01"],
    },
    {
        "label": "SP-01 casual 2 days OK",
        "case_data": {"leave_type": "casual", "employment_duration_months": 0, "days_requested": 2},
        "expected_ids": ["SP-01"],
    },
    {
        "label": "SL-01 sick escalates",
        "case_data": {"leave_type": "sick", "employment_duration_months": 0, "days_requested": 1},
        "expected_ids": ["SL-01"],
    },
    {
        "label": "HJ-01 Hajj before 5 years",
        "case_data": {"leave_type": "hajj", "employment_duration_months": 48, "days_requested": 20},
        "expected_ids": ["HJ-01"],
    },
]

for test in TESTS:
    result = policy_checker(test["case_data"], POLICY_JSON_PATH)
    got = sorted(p["policy_id"] for p in result["matching_policies"])
    exp = sorted(test["expected_ids"])
    ok = got == exp
    status = "PASS" if ok else "FAIL"
    print(f"\n[{status}] {test['label']}")
    print(f"  case_data: {json.dumps(test['case_data'])}")
    print(f"  expected: {exp}")
    print(f"  got:      {got}")
    if not ok:
        print(f"  skipped: {result['skipped_policies']}")
