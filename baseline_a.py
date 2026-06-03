"""
Baseline A — Single LLM call with Bibliotheca Alexandrina circulation rules as vague plain English only.
No JSON policies, no agent pipeline, no memory, no field extraction, no validation.

Represents the naive approach: "describe the rules in the system prompt and ask the LLM to decide."
Intentionally vague — no exact numeric thresholds — to reflect realistic knowledge
a desk assistant might summarise from memory rather than from the formal policy.
"""
from __future__ import annotations

import re

from utils.config import DOMAIN_PROMPTS
from utils.openai_client import get_llm, message_content

# Deliberately vague English summary — no exact book counts, no exact day counts.
# This is what a human desk assistant might write from memory, not the formal policy document.
# The pipeline uses policies/policies.json with exact machine-evaluable conditions instead.
_BOOK_CIRCULATION_RULES_VAGUE = """
Bibliotheca Alexandrina circulation rules (informal summary):

BORROWING
Only researchers can borrow books from the Main Reading Area, and they must use
their individual membership card. Members can typically borrow a small number of
books for a couple of weeks. Borrowing more than a few books or for longer than
a couple of weeks may not be allowed.

RENEWALS
Borrowed materials can sometimes be renewed for another period if they are not
needed by someone else. Renewals should be handled through the proper channels
like the Circulation Desk or the website.

RESERVATIONS
If a book is already borrowed by another member, it can be reserved through
the Circulation Desk. The member will be notified when it becomes available.
Books are borrowed on a first-come, first-served basis.

RETURNS AND RECEIPTS
Borrowed books must be handed over at the Circulation Desk, and members should
make sure they get a check-in receipt.

OVERDUES AND FINES
The library sends notifications for overdue books. Members must pay any
outstanding fines before borrowing other books. If a book is delayed and not
returned after notifications, the member may not be able to access library
services until the situation is resolved. The library may also suspend membership.

LOST OR DAMAGED BOOKS
If a book is lost or damaged, the member may need to pay for it plus some
administrative expenses. If a book is accidentally damaged, users must inform
the Library immediately and should not try to repair it themselves.

NON-CIRCULATING MATERIALS
Some types of materials are not available for borrowing, such as reference works,
periodicals, audio-visual materials, certain illustrated books, theses, and books
in closed stacks.

GENERAL RULES
- If the request does not clearly state what the member wants, which book,
  or their membership status, respond NEED_MORE_INFO.
- If the situation requires desk intervention (renewals, reservations,
  lost/damaged reports, fine payment), respond ESCALATE.
- Do not invent specific numeric thresholds that are not stated above.
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
    - Threshold ambiguity: 'a small number' and 'a couple of weeks' cause wrong
      decisions on boundary cases (e.g. 2 vs 3 books, 14 vs 15 days)
    - Memory blindness: cannot enforce cumulative caps or borrowed-by-others checks
    - NMI underdetection: without a schema, cannot know which fields are required,
      so may guess a decision instead of asking for missing information
    - Inconsistency: same request may yield different decisions across runs
      because nothing constrains the model to a deterministic rule evaluation

    Returns exactly one of: APPROVE, REJECT, ESCALATE, NEED_MORE_INFO
    """
    role = DOMAIN_PROMPTS.get("assistant_role", "circulation assistant").strip()
    subject = DOMAIN_PROMPTS.get("request_subject", "circulation request").strip()
    subject_title = subject[0].upper() + subject[1:] if subject else "Request"

    prompt = f"""You are a {role}.

Apply ONLY the following rules when deciding. Do not use knowledge of specific
legal thresholds or numeric entitlements beyond what is stated below.

{_BOOK_CIRCULATION_RULES_VAGUE}

Decide exactly one outcome for this {subject}:
  APPROVE        — the request clearly fits an approval path described above
  REJECT         — the request clearly violates a limit described above
  ESCALATE       — renewals, reservations, lost/damaged reports, fine issues,
                   or anything requiring desk intervention per the rules above
  NEED_MORE_INFO — request type, book details, or membership status is missing
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