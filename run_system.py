"""
End-to-end pipeline wrapper for evaluation and scripts.
Returns normalized decisions: APPROVE | REJECT | ESCALATE | NEED_MORE_INFO
and policy as a list of policy ids.

Optional ``skip_explanation=True`` skips the explanation LLM call (decision path unchanged);
evaluation uses this during the consistency pass to avoid ~1 LLM call per re-run.
"""
from typing import Any, Dict, List, Optional

from engine.memory_agent import apply_memory_to_matching_policies
from engine.policy_engine import can_make_decision, policy_checker
from interpreter.llm_interpreter import llm_interpreter
from nodes.decision_node import decision_node
from nodes.explanation_node import explanation_node
from nodes.validation_node import validation_node
from utils.config import POLICY_JSON_PATH


def _normalize_out_decision(raw: str) -> str:
    m = {
        "APPROVED": "APPROVE",
        "REJECTED": "REJECT",
        "TERMINATED": "REJECT",
        "ESCALATED": "ESCALATE",
    }
    x = (raw or "").strip().upper()
    return m.get(x, x)


def run_system(
    input_text: str,
    memory_store_patch: Optional[Dict[str, Any]] = None,
    *,
    skip_explanation: bool = False,
) -> Dict[str, Any]:
    """
    skip_explanation: when True, skip the explanation LLM (same decision path). Use for
    evaluation consistency re-runs that only need the decision — much faster.
    """
    interpreter_result = llm_interpreter(input_text)
    case_data = interpreter_result.get("case_data", {})

    validation_result = validation_node({"case_data": case_data})
    validation_errors: List[str] = (
        [] if validation_result.get("is_valid", True) else validation_result.get("issues", [])
    )

    policy_result = policy_checker(case_data, POLICY_JSON_PATH)
    matching = policy_result.get("matching_policies", [])
    skipped = policy_result.get("skipped_policies", [])
    matching, _, hints = apply_memory_to_matching_policies(
        case_data, matching, store_patch=memory_store_patch
    )
    matched_ids = sorted([p.get("policy_id") for p in matching if p.get("policy_id")])

    if validation_errors:
        decision = "REJECT"
        explanation = (
            "Request cannot proceed due to invalid input: "
            + "; ".join(validation_errors)
        )
    elif not matching and not can_make_decision(case_data, POLICY_JSON_PATH):
        decision = "NEED_MORE_INFO"
        if skip_explanation:
            explanation = ""
        else:
            explanation = explanation_node(
                {
                    "input": input_text,
                    "case_data": case_data,
                    "validation_errors": validation_errors,
                    "matching_policies": matching,
                    "skipped_policies": skipped,
                    "memory_hints": hints,
                    "decision": "NEED_MORE_INFO",
                }
            ).get("explanation", "")
    else:
        decision = decision_node(
            {"validation_errors": validation_errors, "matching_policies": matching}
        ).get("decision", "APPROVED")
        if skip_explanation:
            explanation = ""
        else:
            explanation = explanation_node(
                {
                    "input": input_text,
                    "case_data": case_data,
                    "validation_errors": validation_errors,
                    "matching_policies": matching,
                    "skipped_policies": skipped,
                    "memory_hints": hints,
                }
            ).get("explanation", "")

    return {
        "json": case_data,
        "decision": _normalize_out_decision(decision),
        "policy": matched_ids,
        "explanation": explanation,
        "memory_hints": hints if not validation_errors else [],
    }
