import json
from utils.openai_client import get_llm, message_content

llm = get_llm()


def explanation_node(state):
    print("\n [NODE] Explanation")

    matching_policies = state.get("matching_policies", [])
    case_data = state.get("case_data", {})
    input_text = state.get("input", "")
    validation_errors = state.get("validation_errors", [])
    human_reason = state.get("human_reason", "")  # only set on HITL rejection
    memory_hints = state.get("memory_hints") or []

    def with_memory(text: str) -> str:
        if validation_errors or human_reason or not memory_hints:
            return text
        return (text or "").rstrip() + "\n\n" + "\n\n".join(memory_hints)

    if validation_errors:
        error_text = " ".join(validation_errors)
        prompt = f"""
You are an assistant that summarizes validation failures clearly and professionally.

The user made this request:
{input_text}

The request details are:
{json.dumps(case_data, indent=2)}

The request cannot proceed because of the following validation error(s):
{error_text}

Respond in 2 sentences max. State what was invalid and that the request was terminated.
Do not use markdown or bullets. Do not apologize excessively.
"""
        response = llm.invoke(prompt)
        explanation = message_content(response)
        print("Explanation:", explanation)
        return {"explanation": with_memory(explanation)}

    # HITL rejection with human-written reason
    if human_reason:
        prompt = f"""
You are an assistant writing a formal rejection notice.

The user made this request:
"{input_text}"

Their details:
{json.dumps(case_data, indent=2)}

A human reviewer has rejected this request for the following reason:
"{human_reason}"

Write a clear, professional explanation (3-4 sentences) that:
1. States the request was rejected following human review
2. Elaborates on the reviewer's reason with more context and detail
3. Mentions what the user could do next to resolve the issue
4. Is factual and professional — not cold, but not overly sympathetic either
Do not use bullet points or markdown.
"""
        response = llm.invoke(prompt)
        explanation = message_content(response)
        print("Explanation:", explanation)
        return {"explanation": with_memory(explanation)}

    if state.get("decision") == "NEED_MORE_INFO":
        prompt = f"""
You are an assistant.

The user made this request:
"{input_text}"

Their details:
{json.dumps(case_data, indent=2)}

We cannot decide yet because important information is missing (leave type, days, and/or tenure).
Write a brief, professional message (2-3 sentences) asking what is still needed.
Do not use bullet points or markdown.
"""
        response = llm.invoke(prompt)
        explanation = message_content(response)
        print("Explanation:", explanation)
        return {"explanation": with_memory(explanation)}

    if not matching_policies:
        prompt = f"""
You are an assistant.

The user made this request:
"{input_text}"

Their details:
{json.dumps(case_data, indent=2)}

The request has been approved. Write a brief, professional confirmation (2 sentences).
State what was approved. Do not use bullet points or markdown. Do not be overly enthusiastic.
"""
        response = llm.invoke(prompt)
        explanation = message_content(response)
        print("Explanation:", explanation)
        return {"explanation": with_memory(explanation)}

    # Policy-driven decision (aggregated across all matches)
    actions = {p.get("action", "REJECT") for p in matching_policies}
    if "REJECT" in actions:
        action = "REJECT"
    elif "ESCALATE" in actions:
        action = "ESCALATE"
    else:
        action = "APPROVE"
    user_role = case_data.get("user_type")

    policy_summary = "\n".join(
        f"- {p.get('message') or p.get('violation', '')}"
        for p in matching_policies
        if p.get('message') or p.get('violation')
    )

    if action == "ESCALATE":
        outcome_phrase = "has been flagged for human review"
        tone_instruction = "Be neutral and factual. Explain why the case needs manual review without pre-judging the outcome."
    elif action == "APPROVE":
        outcome_phrase = "has been approved"
        tone_instruction = "Be professional and concise."
    else:
        outcome_phrase = "has been rejected"
        tone_instruction = "Be direct and professional. Briefly mention what the user can do to resolve the situation."

    # Only mention user role if it was explicitly provided
    role_context = ""
    if user_role is not None:
        role_context = f"The user role is: {user_role}.\n\n"

    prompt = f"""
You are an assistant explaining a policy decision.

The user made this request:
"{input_text}"

Their details:
{json.dumps(case_data, indent=2)}

{role_context}
The request {outcome_phrase} due to the following triggered policy reason(s):
{policy_summary}

{tone_instruction}

Write a clear, professional explanation (3-4 sentences) that:
1. States the outcome directly
2. If a role is provided, reference it; otherwise focus on the triggered policies and relevant details
3. Mentions all triggered policy reasons in plain language
4. Provides a clear, professional explanation without excessive sympathy
"""
    response = llm.invoke(prompt)
    explanation = message_content(response)
    print("Explanation:", explanation)
    return {"explanation": with_memory(explanation)}