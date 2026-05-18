"""
Gold labels from policies.json + the same memory layer as run_system (when enabled).
Keys starting with '_' in case_data are metadata only (e.g. _memory_store_patch) and are not
sent to the policy engine.
"""
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from engine.memory_agent import apply_memory_to_matching_policies
from engine.policy_engine import can_make_decision, policy_checker
from utils.config import POLICY_JSON_PATH


def _policy_case_data(case_data: Dict[str, Any]) -> Dict[str, Any]:
    """Strip evaluation-only keys before policy_checker / can_make_decision."""
    return {k: v for k, v in case_data.items() if not str(k).startswith("_")}


def expected_decision_and_policies(
    case_data: Dict[str, Any],
    policy_path: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """
    Returns (decision, sorted policy ids) where decision is one of:
    APPROVE, REJECT, ESCALATE, NEED_MORE_INFO
    """
    path = policy_path or POLICY_JSON_PATH
    policy_case = _policy_case_data(case_data)
    store_patch = (
        case_data.get("_memory_store_patch")
        if isinstance(case_data, dict)
        else None
    )

    result = policy_checker(policy_case, path)
    matching = deepcopy(result.get("matching_policies", []))
    matching, _, _ = apply_memory_to_matching_policies(
        policy_case, matching, store_patch=store_patch
    )
    ids = sorted([p["policy_id"] for p in matching if p.get("policy_id")])

    if matching:
        actions = {p.get("action", "REJECT") for p in matching}
        if "REJECT" in actions:
            dec = "REJECT"
        elif "ESCALATE" in actions:
            dec = "ESCALATE"
        elif "APPROVE" in actions:
            dec = "APPROVE"
        else:
            dec = "REJECT"
        return dec, ids

    if not can_make_decision(policy_case, path):
        return "NEED_MORE_INFO", []

    return "APPROVE", []
