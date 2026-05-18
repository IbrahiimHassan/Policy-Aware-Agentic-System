"""
Sanity validation only: types and non-negative numeric fields.
Does NOT enforce policies — missing fields are allowed (policy layer handles NEED_MORE_INFO).
"""
from typing import Any, Dict, List

from utils.config import SCHEMA


def validation_node(state):
    print("\n [NODE] Validation")

    case_data = state.get("case_data") or {}
    issues: List[str] = []

    for field_name, meta in SCHEMA.items():
        value = case_data.get(field_name)
        if value is None:
            continue

        expected_type = meta.get("type", "string")

        if expected_type == "number":
            if not isinstance(value, (int, float)):
                issues.append(f"{field_name} must be a number.")
                continue
            if isinstance(value, float) and value != int(value):
                issues.append(f"{field_name} must be a whole number.")
                continue
            iv = int(value)
            if iv < 0:
                issues.append(f"{field_name} cannot be negative.")

        elif expected_type == "string":
            if not isinstance(value, str):
                issues.append(f"{field_name} must be a string.")

    if not issues:
        print("Data is valid.")
        return {"is_valid": True, "issues": [], "suggested_corrections": {}}

    print("Validation Issues:")
    for issue in issues:
        print(f"- {issue}")

    return {"is_valid": False, "issues": issues, "suggested_corrections": {}}
