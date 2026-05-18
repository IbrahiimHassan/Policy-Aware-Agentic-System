import json
import re

from utils.config import SCHEMA


# -----------------------------
# CONDITION EVALUATION
# -----------------------------
def evaluate_condition(condition, data):
    """
    Recursively evaluate a condition against input data.
    Supports: and, or, not, and simple comparisons.
    Also supports string conditions like "field != 'value'"
    """

    # Handle string conditions
    if isinstance(condition, str):
        return evaluate_string_condition(condition, data)

    # AND condition
    if "and" in condition:
        return all(evaluate_condition(c, data) for c in condition["and"])

    # OR condition
    if "or" in condition:
        return any(evaluate_condition(c, data) for c in condition["or"])

    # NOT condition
    if "not" in condition:
        return not evaluate_condition(condition["not"], data)

    # Simple condition
    field = condition.get("field")
    operator = condition.get("operator")
    value = condition.get("value")

    actual = data.get(field)

    # If field missing → treat as False
    if actual is None:
        return False

    try:
        if operator == "==":
            return actual == value

        elif operator == "!=":
            return actual != value

        elif operator == ">":
            return actual > value

        elif operator == "<":
            return actual < value

        elif operator == ">=":
            return actual >= value

        elif operator == "<=":
            return actual <= value

        elif operator == "in":
            return actual in value

        elif operator == "not_in":
            return actual not in value

    except Exception:
        return False

    return False


# -----------------------------
# EXTRACT FIELDS FROM CONDITION
# -----------------------------
def extract_fields_from_condition(condition_str):
    """
    Extract all field names from a condition string.
    Returns a set of field names used in the condition.
    """
    fields = set()
    
    # Pattern for != with string value
    for match in re.finditer(r"(\w+)\s*!=\s*['\"](.+?)['\"]", condition_str):
        fields.add(match.group(1))
    
    # Pattern for == with string value
    for match in re.finditer(r"(\w+)\s*==\s*['\"](.+?)['\"]", condition_str):
        fields.add(match.group(1))
    
    # Pattern for == with boolean
    for match in re.finditer(r"(\w+)\s*==\s*(true|false)", condition_str, re.IGNORECASE):
        fields.add(match.group(1))
    
    # Pattern for != with boolean
    for match in re.finditer(r"(\w+)\s*!=\s*(true|false)", condition_str, re.IGNORECASE):
        fields.add(match.group(1))
    
    # Pattern for > with number
    for match in re.finditer(r"(\w+)\s*>\s*(\d+)", condition_str):
        fields.add(match.group(1))
    
    # Pattern for < with number
    for match in re.finditer(r"(\w+)\s*<\s*(\d+)", condition_str):
        fields.add(match.group(1))
    
    # Pattern for >= with number
    for match in re.finditer(r"(\w+)\s*>=\s*(\d+)", condition_str):
        fields.add(match.group(1))
    
    # Pattern for <= with number
    for match in re.finditer(r"(\w+)\s*<=\s*(\d+)", condition_str):
        fields.add(match.group(1))
    
    # Pattern for in with list
    for match in re.finditer(r"(\w+)\s+in\s+\[(.+?)\]", condition_str):
        fields.add(match.group(1))

    # Pattern for simple arithmetic like "a + b > 10"
    for match in re.finditer(r"(\w+)\s*\+\s*(\w+)\s*(?:>=|<=|==|!=|>|<)\s*\d+", condition_str):
        fields.add(match.group(1))
        fields.add(match.group(2))
    
    # Handle 'and' and 'or' by recursively extracting
    if " and " in condition_str:
        for part in condition_str.split(" and "):
            fields.update(extract_fields_from_condition(part))
    
    if " or " in condition_str:
        for part in condition_str.split(" or "):
            fields.update(extract_fields_from_condition(part))
    
    return fields


# -----------------------------
# CHECK IF POLICY CAN BE EVALUATED
# -----------------------------
def _missing_fields_for_atom(atom: str, data) -> list:
    """Fields referenced by one atomic clause that are missing or null."""
    fields = extract_fields_from_condition(atom.strip())
    return [f for f in fields if f not in data or data[f] is None]


def _can_evaluate_string_policy(condition_str: str, data):
    """
    Whether a string condition can be evaluated without guessing missing values.

    For AND chains, uses short-circuit: if an earlier clause is false, later
    clauses need not be evaluated (so missing fields there do not block the policy).
    """
    condition_str = condition_str.strip()

    if " or " in condition_str:
        parts = [p.strip() for p in condition_str.split(" or ")]
        pending_missing = []
        saw_possible_true = False
        for part in parts:
            missing = _missing_fields_for_atom(part, data)
            if missing:
                pending_missing.extend(missing)
                continue
            if evaluate_string_condition(part, data):
                saw_possible_true = True
                break
        if saw_possible_true:
            return True, []
        if pending_missing:
            return False, list(dict.fromkeys(pending_missing))
        return True, []

    if " and " in condition_str:
        parts = [p.strip() for p in condition_str.split(" and ")]
        for part in parts:
            missing = _missing_fields_for_atom(part, data)
            if missing:
                return False, missing
            if not evaluate_string_condition(part, data):
                return True, []
        return True, []

    missing = _missing_fields_for_atom(condition_str, data)
    if missing:
        return False, missing
    return True, []


def can_evaluate_policy(condition, data):
    """
    Returns (True, []) if the condition can be evaluated to true/false.
    Returns (False, [missing_fields]) if unknown due to missing values.
    """
    if isinstance(condition, str):
        return _can_evaluate_string_policy(condition, data)

    fields = extract_fields_from_condition(condition)
    missing = []

    for field in fields:
        if field not in data or data[field] is None:
            missing.append(field)

    if missing:
        return False, missing
    return True, []


# -----------------------------
# STRING CONDITION EVALUATION
# -----------------------------
def evaluate_string_condition(condition_str, data):
    """
    Evaluate a string condition like "field != 'value'"
    or "count > 2" or "has_property == false".
    
    NOTE: This should only be called after verifying all fields are present.
    """
    condition_str = condition_str.strip()

    # Handle 'and' / 'or' first. Otherwise a prefix like "leave_type == 'annual'"
    # would match and return early, ignoring the rest of the condition.
    if " and " in condition_str:
        parts = condition_str.split(" and ")
        return all(evaluate_string_condition(p.strip(), data) for p in parts)

    if " or " in condition_str:
        parts = condition_str.split(" or ")
        return any(evaluate_string_condition(p.strip(), data) for p in parts)
    
    # Match patterns: field operator value
    # Supported operators: ==, !=, >, <, >=, <=, in
    
    # Pattern for arithmetic like "a + b > 10"
    match = re.match(r"(\w+)\s*\+\s*(\w+)\s*>\s*(\d+)", condition_str)
    if match:
        left, right, value = match.groups()
        return (data.get(left) + data.get(right)) > int(value)

    match = re.match(r"(\w+)\s*\+\s*(\w+)\s*>=\s*(\d+)", condition_str)
    if match:
        left, right, value = match.groups()
        return (data.get(left) + data.get(right)) >= int(value)

    match = re.match(r"(\w+)\s*\+\s*(\w+)\s*<\s*(\d+)", condition_str)
    if match:
        left, right, value = match.groups()
        return (data.get(left) + data.get(right)) < int(value)

    match = re.match(r"(\w+)\s*\+\s*(\w+)\s*<=\s*(\d+)", condition_str)
    if match:
        left, right, value = match.groups()
        return (data.get(left) + data.get(right)) <= int(value)

    match = re.match(r"(\w+)\s*\+\s*(\w+)\s*==\s*(\d+)", condition_str)
    if match:
        left, right, value = match.groups()
        return (data.get(left) + data.get(right)) == int(value)

    match = re.match(r"(\w+)\s*\+\s*(\w+)\s*!=\s*(\d+)", condition_str)
    if match:
        left, right, value = match.groups()
        return (data.get(left) + data.get(right)) != int(value)

    # Pattern for != with string value
    match = re.match(r"(\w+)\s*!=\s*['\"](.+?)['\"]", condition_str)
    if match:
        field, value = match.groups()
        return data.get(field) != value
    
    # Pattern for == with string value
    match = re.match(r"(\w+)\s*==\s*['\"](.+?)['\"]", condition_str)
    if match:
        field, value = match.groups()
        return data.get(field) == value
    
    # Pattern for == with boolean
    match = re.match(r"(\w+)\s*==\s*(true|false)", condition_str, re.IGNORECASE)
    if match:
        field, value = match.groups()
        return data.get(field) == (value.lower() == "true")
    
    # Pattern for != with boolean
    match = re.match(r"(\w+)\s*!=\s*(true|false)", condition_str, re.IGNORECASE)
    if match:
        field, value = match.groups()
        return data.get(field) != (value.lower() == "true")
    
    # Pattern for > with number
    match = re.match(r"(\w+)\s*>\s*(\d+)", condition_str)
    if match:
        field, value = match.groups()
        return data.get(field) > int(value)
    
    # Pattern for < with number
    match = re.match(r"(\w+)\s*<\s*(\d+)", condition_str)
    if match:
        field, value = match.groups()
        return data.get(field) < int(value)
    
    # Pattern for >= with number
    match = re.match(r"(\w+)\s*>=\s*(\d+)", condition_str)
    if match:
        field, value = match.groups()
        return data.get(field) >= int(value)
    
    # Pattern for <= with number
    match = re.match(r"(\w+)\s*<=\s*(\d+)", condition_str)
    if match:
        field, value = match.groups()
        return data.get(field) <= int(value)
    
    # Pattern for in with list
    match = re.match(r"(\w+)\s+in\s+\[(.+?)\]", condition_str)
    if match:
        field, value_list = match.groups()
        values = [v.strip().strip("'\"") for v in value_list.split(",")]
        return data.get(field) in values
    
    return False


# -----------------------------
# LOAD POLICIES
# -----------------------------
def load_policies(policy_path):
    """
    Load policies from JSON file.
    """
    with open(policy_path, "r") as f:
        return json.load(f)["policies"]


# -----------------------------
# SAFE POLICY EVALUATOR
# -----------------------------
def safe_evaluate_policy(policy, data):
    """
    Safely evaluate a policy, skipping it if any required field is missing.
    Returns (True, triggered) if policy was evaluated, (False, []) if skipped.
    """
    condition = policy.get("condition", "")
    
    # Check if all required fields are present
    can_eval, missing = can_evaluate_policy(condition, data)
    
    if not can_eval:
        # Skip this policy - cannot evaluate due to missing fields
        return False, []
    
    # All fields present - evaluate the condition
    if evaluate_condition(condition, data):
        return True, [policy]
    
    return False, []


# -----------------------------
# MAIN POLICY CHECKER
# -----------------------------
def policy_checker(data, policy_path):
    """
    Main function to evaluate policies against input data.
    Returns matching policies.
    
    IMPORTANT: Policies with missing required fields are SKIPPED.
    """

    policies = load_policies(policy_path)

    matching_policies = []
    skipped_policies = []

    for rule in policies:
        condition = rule.get("condition", "")
        
        # Check if we can evaluate this policy
        can_eval, missing = can_evaluate_policy(condition, data)
        
        if not can_eval:
            # Skip this policy - cannot evaluate due to missing fields
            skipped_policies.append({
                "policy_id": rule.get("id"),
                "condition": condition,
                "missing_fields": missing
            })
            continue
        
        # Evaluate the condition (all fields are present)
        if evaluate_condition(condition, data):

            policy_entry = {
                "policy_id": rule.get("id"),
                "violation": rule.get("violation"),
                "action": rule.get("action"),
                "message": rule.get("message"),
                "type": rule.get("type")
            }

            matching_policies.append(policy_entry)

    return {
        "matching_policies": matching_policies,
        "skipped_policies": skipped_policies
    }


# -----------------------------
# GET REQUIRED FIELDS FROM POLICIES
# -----------------------------
def get_required_fields_from_policies(policy_path):
    """
    Dynamically detect which fields are needed based on policies.
    Returns a set of field names that appear in any policy condition.
    """
    policies = load_policies(policy_path)
    required_fields = set()

    for rule in policies:
        condition = rule.get("condition", "")
        # Extract field names from condition
        # Simple extraction: find all field= patterns
        import re
        fields = re.findall(r'(\w+)\s*(?:==|!=|>|<|>=|<=|in|not_in)', condition)
        required_fields.update(fields)

    return required_fields


# -----------------------------
# CHECK IF DECISION CAN BE MADE
# -----------------------------
def can_make_decision(case_data, policy_file):
    """
    Determine if a decision can be made based on the current case data and policies.
    Returns True if a decision can be made, False otherwise.
    
    A decision can be made if:
    - At least one policy is triggered (clear violation/action from available data), OR
    - No policy requires missing fields (complete information for policy evaluation).
    
    A decision CANNOT be made if:
    - Any policy is skipped due to missing required fields,
    - AND no policy has already been triggered.
    """
    # First, check if any policy is triggered
    result = policy_checker(case_data, policy_file)
    matching = result.get("matching_policies", [])
    
    if matching:
        # Clear policy outcome exists even with partial data.
        return True

    # No policies triggered. If anything was skipped, information is incomplete.
    skipped = result.get("skipped_policies", [])
    if skipped:
        return False

    # No triggered policies and no skipped policies -> safe to approve.
    return True


# -----------------------------
# GET FIELDS NEEDED FOR DECISION
# -----------------------------
def get_fields_needed_for_decision(data, policy_path):
    """
    Get the minimal set of fields needed to make a decision.
    Only returns fields that are:
    1. Currently missing (null or not present)
    2. Required by policies that haven't been evaluated yet
    """
    policies = load_policies(policy_path)
    
    needed_fields = set()
    
    for policy in policies:
        condition = policy.get("condition", "")
        
        # Check if we can evaluate this policy
        can_eval, missing = can_evaluate_policy(condition, data)
        
        if not can_eval:
            # This policy cannot be evaluated - add its missing fields
            needed_fields.update(missing)
    
    return list(needed_fields)


# -----------------------------
# FORMAT QUESTIONS AS BULLET POINTS
# -----------------------------
def format_questions(questions):
    """
    Format questions as bullet points for UI display.
    """
    if not questions:
        return ""

    text = "We need a bit more information to proceed:\n"
    for q in questions:
        text += f"- {q}\n"
    return text


# -----------------------------
# MAP FIELD TO QUESTION
# -----------------------------
def field_to_question(field):
    """
    Map a missing field to a user-friendly question.
    Never exposes field names to the user.
    """
    field_meta = SCHEMA.get(field, {})
    if field_meta.get("question"):
        return field_meta["question"]

    # Fallback for unknown fields - describe by type, not by field name
    field_type = field_meta.get("type", "information")
    return f"Please provide the missing {field_type}."


# -----------------------------
# GENERATE EXPLANATION FROM TRIGGERED POLICIES
# -----------------------------
def generate_explanation(matching_policies, data):
    """
    Generate explanation ONLY from triggered policies.
    
    IMPORTANT:
    - Only includes triggered policies
    - Only uses known data
    - Never mentions missing fields
    - Never assumes values
    
    Args:
        matching_policies: List of triggered policy entries from policy_checker
        data: The original data dictionary (used to verify known values)
    
    Returns:
        List of explanation strings
    """
    explanations = []
    
    for policy in matching_policies:
        message = policy.get("message", "")
        if message:
            explanations.append(message)
    
    return explanations


# -----------------------------
# MAKE DECISION WITH FULL WORKFLOW
# -----------------------------
def make_decision(data, policy_path):
    """
    Complete decision workflow with non-assumptive approach.
    
    Returns:
        dict with:
        - decision: "REJECT", "APPROVE", "ESCALATE", or "NEED_MORE_INFO"
        - triggered_policies: list of triggered policy messages
        - questions: list of questions to ask (if NEED_MORE_INFO)
        - explanation: list of explanation strings
    """
    # Evaluate policies
    result = policy_checker(data, policy_path)
    matching = result.get("matching_policies", [])
    skipped = result.get("skipped_policies", [])
    
    # Determine decision
    if matching:
        # At least one policy triggered - aggregate all matched actions
        actions = {p.get("action", "REJECT") for p in matching}
        if "REJECT" in actions:
            action = "REJECT"
        elif "ESCALATE" in actions:
            action = "ESCALATE"
        elif "APPROVE" in actions:
            action = "APPROVE"
        else:
            action = "REJECT"
        
        # Generate explanation ONLY from triggered policies
        explanation = generate_explanation(matching, data)
        
        return {
            "decision": action,
            "triggered_policies": [p.get("policy_id") for p in matching],
            "explanation": explanation,
            "questions": []
        }
    
    if skipped:
        # Incomplete information and no triggered policy -> ask follow-up questions.
        needed_fields = get_fields_needed_for_decision(data, policy_path)
        questions = [field_to_question(f) for f in needed_fields]
        
        return {
            "decision": "NEED_MORE_INFO",
            "triggered_policies": [],
            "explanation": [],
            "questions": questions
        }
    
    # We could evaluate and none were triggered - APPROVE
    return {
        "decision": "APPROVE",
        "triggered_policies": [],
        "explanation": [],
        "questions": []
    }