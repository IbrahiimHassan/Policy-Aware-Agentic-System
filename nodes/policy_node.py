from engine.policy_engine import policy_checker, can_make_decision, get_fields_needed_for_decision, field_to_question
from utils.config import POLICY_JSON_PATH


def policy_node(state):
    print("\n [NODE] Policy Checker")
    result = policy_checker(
        state["case_data"],
        POLICY_JSON_PATH
    )

    print("Matching Policies:")
    for policy in result["matching_policies"]:
        print(f"  - {policy['policy_id']}: {policy['message']}")
    
    print("Skipped Policies (missing fields):")
    for policy in result["skipped_policies"]:
        print(f"  - {policy['policy_id']}: {policy['missing_fields']}")

    # Check if decision can be made
    can_decide = can_make_decision(state["case_data"], POLICY_JSON_PATH)
    
    return {
        "matching_policies": result["matching_policies"],
        "skipped_policies": result["skipped_policies"],
        "can_decide": can_decide
    }