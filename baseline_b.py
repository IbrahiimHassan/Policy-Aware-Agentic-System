"""
Baseline B — Single LLM call with structured JSON policies (conditions intact).
No agent pipeline, no memory state, no validation node.

The LLM must:
1. Extract structured facts from the employee request text
2. Evaluate each policy's machine-readable condition against those facts
3. Apply precedence rules and return a decision

This tests whether a single LLM call can replace the full pipeline
(interpreter + policy engine + decision node) when given the same
structured policy knowledge.
"""
from __future__ import annotations

import json
import re

from utils.config import DOMAIN_PROMPTS
from utils.openai_client import get_llm, message_content

_VALID = frozenset({"APPROVE", "REJECT", "ESCALATE", "NEED_MORE_INFO"})


def _format_policies(policies_json: dict) -> str:
    """
    Keep the FULL structured policy JSON — including machine-evaluable conditions.
    The LLM will read the conditions and evaluate them against extracted facts.
    """
    if not isinstance(policies_json, dict):
        return "{}"

    formatted_policies = []
    for policy in policies_json.get("policies", []):
        formatted_policies.append(
            {
                "id": policy.get("id"),
                "category": policy.get("category"),
                "condition": policy.get("condition"),
                "action": policy.get("action"),
                "message": policy.get("message"),
            }
        )

    output = {
        "policy_set": policies_json.get("policy_set", ""),
        "note": (
            "Evaluate each policy's CONDITION against the facts you extract "
            "from the employee request. The condition uses Python-like syntax. "
            "If all variables in the condition are available and the expression "
            "evaluates to True, the policy matches."
        ),
        "policies": formatted_policies,
    }
    return json.dumps(output, ensure_ascii=False, indent=2)


def _parse(raw: str) -> str:
    text = (raw or "").strip().upper()
    compact = text.replace(" ", "_").replace("-", "_")
    if "NEED_MORE_INFO" in compact or "NEED_MORE" in compact:
        return "NEED_MORE_INFO"
    for token in ("ESCALATE", "APPROVE", "REJECT"):
        if text == token or text.startswith(token + " ") or text.startswith(token + "."):
            return token
    m = re.search(r"\b(APPROVE|REJECT|ESCALATE|NEED_MORE_INFO)\b", text)
    if m and m.group(1) in _VALID:
        return m.group(1)
    return "NEED_MORE_INFO"


def baseline_b_llm(input_text: str, policies_json: dict) -> str:
    """
    Baseline B: single LLM call with full structured policies (conditions intact).
    No agent pipeline, no memory state, no validation node.

    The LLM must:
    1. Extract structured facts from the employee request text
    2. Evaluate each policy's machine-readable condition against those facts
    3. Apply precedence rules and return a decision

    Expected failure modes vs the pipeline:
    - Extraction errors: typos, abbreviations, or mixed-language input may cause
      the LLM to misread days_requested or employment_duration_months
    - Compound condition failures: multi-field conditions (tenure band AND day count)
      may be partially satisfied in the LLM's reasoning
    - Memory blindness: no cumulative state injected, so all memory-cap REJECT cases
      are incorrectly evaluated as if the employee has zero prior usage
    - JSON reasoning degradation: structured JSON in a prompt is harder for the LLM
      to reason over holistically than dedicated code (cf. StructGPT findings)
    - Precedence inconsistency: the reject > escalate > approve rule is stated in
      the prompt but not enforced deterministically

    Returns exactly one of: APPROVE, REJECT, ESCALATE, NEED_MORE_INFO
    """
    role = DOMAIN_PROMPTS.get("assistant_role", "HR assistant").strip()
    subject = DOMAIN_PROMPTS.get("request_subject", "leave request").strip()
    subject_title = subject[0].upper() + subject[1:] if subject else "Request"

    policy_text = _format_policies(policies_json)

    prompt = f"""You are a {role}.

You are given a set of company leave policies below. Each policy has an id,
a category, a machine-readable CONDITION, an intended ACTION, and a human-readable
MESSAGE describing when it applies.

Your task:
1. Read the employee request carefully.
2. Extract structured facts: leave_type, days_requested, employment_duration_months,
   and any other relevant fields mentioned.
   - If numbers and a leave type are present in the text, extract them even if
     phrasing is informal (e.g., "10 days off" → days_requested=10).
   - For tenure, extract any mention of employment length; if absent, mark it null.
3. Evaluate each policy's CONDITION against the extracted facts.
   The condition uses Python-like syntax (e.g., 'leave_type == "annual" and
   employment_duration_months >= 6 and days_requested <= 15').
   A variable is only "missing" if it is REQUIRED by the condition AND cannot be
   inferred from the text at all. If a condition does NOT mention a variable,
   that variable's absence does NOT prevent evaluation.
   - Example: for casual leave, the condition only checks leave_type and days_requested.
     Do NOT skip casual policies just because tenure is not mentioned.
4. If a condition evaluates to True, the policy matches.

PRECEDENCE RULE (apply strictly):
  If any matching policy says REJECT  → final decision is REJECT
  Else if any says ESCALATE           → final decision is ESCALATE
  Else if any says APPROVE            → final decision is APPROVE
  If no policy clearly matches and critical information is missing → NEED_MORE_INFO
  IMPORTANT: Only use NEED_MORE_INFO when the request truly lacks a leave type,
  day count, or other field that ALL candidate policies require.

POLICIES (full structured JSON with conditions):
{policy_text}

{subject_title}:
{input_text}

Return ONLY one word: APPROVE, REJECT, ESCALATE, or NEED_MORE_INFO.
Do not explain. Do not add punctuation. One word only."""

    llm = get_llm()
    response = llm.invoke(prompt)
    decision = _parse(message_content(response))
    return decision if decision in _VALID else "NEED_MORE_INFO"


def run_baseline_b(input_text: str, policies_json: dict | None = None) -> str:
    """
    Uniform interface for evaluation.py.
    If policies_json is not passed, load it from the default path.
    """
    if policies_json is None:
        from utils.config import POLICY_JSON_PATH
        with open(POLICY_JSON_PATH, "r", encoding="utf-8") as f:
            policies_json = json.load(f)
    return baseline_b_llm(input_text, policies_json)