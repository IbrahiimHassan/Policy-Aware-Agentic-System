"""
Baseline A — Single LLM call with Egyptian leave rules as vague plain English only.
No JSON policies, no agent pipeline, no memory, no field extraction, no validation.

Represents the naive approach: "describe the rules in the system prompt and ask the LLM to decide."
Intentionally vague — no exact numeric thresholds — to reflect realistic knowledge
a manager might summarise from memory rather than from the formal statute.
"""
from __future__ import annotations

import re

from utils.config import DOMAIN_PROMPTS
from utils.openai_client import get_llm, message_content

# Deliberately vague English summary — no exact day counts, no exact tenure thresholds.
# This is what a human manager might write from memory, not the formal policy document.
# The pipeline uses policies/policies.json with exact machine-evaluable conditions instead.
_EGYPT_LEAVE_RULES_VAGUE = """
Egyptian private-sector leave rules (informal summary):

ANNUAL LEAVE
Employees are entitled to paid annual leave based on how long they have worked
at the company. Newer employees with only a few months of service are entitled
to fewer days than employees who have been with the company for a year or more.
Very long-serving employees may be entitled to more.

SICK LEAVE
All sick leave requests must be escalated to HR for review. Sick leave involves
medical documentation requirements and cannot be auto-approved regardless of
the number of days requested.

MATERNITY LEAVE
Female employees are entitled to a period of paid maternity leave after giving
birth. Requests within the standard entitlement period can be approved.
Requests that appear to exceed the standard entitlement should be rejected.

HAJJ LEAVE
Employees who have been with the company for a sufficiently long period are
entitled to take Hajj leave once during their employment. Employees who have
not yet served long enough, or who request more than the allowed period, should
be rejected.

SHORT PERSONAL / CASUAL LEAVE
Employees may take a small number of days off for urgent personal matters
without formal HR review. Requests for more than a few days of personal leave
should be rejected for auto-approval and handled through HR instead.

CONTACT / CONTAGION LEAVE
Must be escalated. Requires medical authority involvement.

UNPAID LEAVE
Must be escalated. Requires employer approval and HR processing.

GENERAL RULES
- If the request does not clearly state what type of leave is needed, how many
  days are requested, or how long the employee has worked there (when relevant),
  respond NEED_MORE_INFO.
- If the situation is ambiguous or requires human judgment, respond ESCALATE.
- Do not invent numeric thresholds that are not stated above.
- Apply common sense only when the request does not match any clear policy rule.
- Do not assume prior usage or memory state — judge the request on its own text.
"""

_VALID = frozenset({"APPROVE", "REJECT", "ESCALATE", "NEED_MORE_INFO"})


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


def baseline_a_llm(input_text: str) -> str:
    """
    Baseline A: single LLM call, vague English rules only.
    No memory state, no field extraction, no validation, no policy JSON.

    Expected failure modes:
    - Threshold ambiguity: 'a few days' and 'reasonable tenure' cause wrong
      decisions on boundary cases (e.g. 15 vs 16 days, 6 vs 5 months tenure)
    - Memory blindness: cannot enforce cumulative caps — always sees fresh state
    - NMI underdetection: without a schema, cannot know which fields are required,
      so may guess a decision instead of asking for missing information
    - Inconsistency: same request may yield different decisions across runs
      because nothing constrains the model to a deterministic rule evaluation

    Returns exactly one of: APPROVE, REJECT, ESCALATE, NEED_MORE_INFO
    """
    role = DOMAIN_PROMPTS.get("assistant_role", "HR assistant").strip()
    subject = DOMAIN_PROMPTS.get("request_subject", "leave request").strip()
    subject_title = subject[0].upper() + subject[1:] if subject else "Request"

    prompt = f"""You are a {role}.

Apply ONLY the following rules when deciding. Do not use knowledge of specific
legal thresholds or numeric entitlements beyond what is stated below.

{_EGYPT_LEAVE_RULES_VAGUE}

Decide exactly one outcome for this {subject}:
  APPROVE        — the request clearly fits an approval path described above
  REJECT         — the request clearly violates a limit described above
  ESCALATE       — sick leave, contact leave, unpaid leave, or anything requiring
                   HR or medical review per the rules above
  NEED_MORE_INFO — leave type, number of days, or length of service is missing
                   or too vague to apply the rules above

{subject_title}:
{input_text}

Reply with ONLY one word: APPROVE, REJECT, ESCALATE, or NEED_MORE_INFO.
Do not explain. Do not add punctuation."""

    llm = get_llm()
    response = llm.invoke(prompt)
    return _parse(message_content(response))


# Allow evaluation.py to call run_baseline_a(request) as a uniform interface
def run_baseline_a(input_text: str) -> str:
    return baseline_a_llm(input_text)