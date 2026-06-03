"""
Smoke tests for policy_checker against policies/policies.json (BA book circulation policy).
Uses structured case_data only — no LLM — so runs offline.
"""
import json

from engine.policy_engine import policy_checker
from utils.config import POLICY_JSON_PATH

TESTS = [
    {
        "label": "BR-01 non-researcher reject",
        "case_data": {"request_type": "borrow", "user_type": "visitor", "has_membership_card": True, "current_books_borrowed": 0, "books_requested": 1, "days_requested": 5, "book_category": "regular"},
        "expected_ids": ["BR-01"],
    },
    {
        "label": "BR-02 no membership card reject",
        "case_data": {"request_type": "borrow", "user_type": "researcher", "has_membership_card": False, "current_books_borrowed": 0, "books_requested": 1, "days_requested": 5, "book_category": "regular"},
        "expected_ids": ["BR-02"],
    },
    {
        "label": "BR-03 current books >= 2 reject",
        "case_data": {"request_type": "borrow", "user_type": "researcher", "has_membership_card": True, "current_books_borrowed": 2, "books_requested": 1, "days_requested": 5, "book_category": "regular"},
        "expected_ids": ["BR-03", "BR-07"],
    },
    {
        "label": "BR-04 books requested > 2 reject",
        "case_data": {"request_type": "borrow", "user_type": "researcher", "has_membership_card": True, "current_books_borrowed": 0, "books_requested": 3, "days_requested": 5, "book_category": "regular"},
        "expected_ids": ["BR-04"],
    },
    {
        "label": "BR-05 days > 14 reject",
        "case_data": {"request_type": "borrow", "user_type": "researcher", "has_membership_card": True, "current_books_borrowed": 0, "books_requested": 1, "days_requested": 15, "book_category": "regular"},
        "expected_ids": ["BR-05"],
    },
    {
        "label": "BR-07 borrow approve",
        "case_data": {"request_type": "borrow", "user_type": "researcher", "has_membership_card": True, "current_books_borrowed": 0, "books_requested": 2, "days_requested": 14, "book_category": "regular"},
        "expected_ids": ["BR-07"],
    },
    {
        "label": "NC-01 non-circulating reference reject",
        "case_data": {"request_type": "borrow", "user_type": "researcher", "has_membership_card": True, "current_books_borrowed": 0, "books_requested": 1, "days_requested": 5, "book_category": "reference"},
        "expected_ids": ["BR-06", "NC-01", "BR-07"],
    },
    {
        "label": "NC-01 non-circulating thesis reject",
        "case_data": {"request_type": "borrow", "user_type": "researcher", "has_membership_card": True, "current_books_borrowed": 0, "books_requested": 1, "days_requested": 5, "book_category": "thesis"},
        "expected_ids": ["BR-06", "NC-01", "BR-07"],
    },
    {
        "label": "RN-01 renewal escalate",
        "case_data": {"request_type": "renew", "user_type": "researcher", "has_membership_card": True, "current_books_borrowed": 1, "books_requested": 0, "days_requested": 0, "book_category": "regular"},
        "expected_ids": ["RN-01"],
    },
    {
        "label": "FN-01 fines reject",
        "case_data": {"request_type": "borrow", "user_type": "researcher", "has_membership_card": True, "current_books_borrowed": 0, "books_requested": 1, "days_requested": 5, "book_category": "regular", "has_fines": True},
        "expected_ids": ["FN-01", "BR-07"],
    },
    {
        "label": "BR-07 borrow approve with missing optional fields",
        "case_data": {"request_type": "borrow", "user_type": "researcher", "has_membership_card": True, "books_requested": 1, "days_requested": 10},
        "expected_ids": ["BR-07"],
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
