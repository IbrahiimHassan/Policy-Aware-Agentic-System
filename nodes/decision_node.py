from engine.policy_engine import can_make_decision
from utils.config import POLICY_JSON_PATH


def decision_node(state):
    print("\n [NODE] Decision")

    validation_errors = state.get("validation_errors", [])
    matching_policies = state.get("matching_policies", [])
    case_data = state.get("case_data") or {}

    if validation_errors:
        decision = "TERMINATED"

    elif matching_policies:
        actions = {p.get("action", "REJECT") for p in matching_policies}
        if "REJECT" in actions:
            decision = "REJECTED"
        elif "ESCALATE" in actions:
            decision = "ESCALATE"
        elif "APPROVE" in actions:
            decision = "APPROVED"
        else:
            decision = "REJECTED"

    elif not can_make_decision(case_data, POLICY_JSON_PATH):
        decision = "NEED_MORE_INFO"

    else:
        decision = "APPROVED"

    print(f"Decision: {decision}")
    return {"decision": decision}