import streamlit as st
import json
import time
from interpreter.llm_interpreter import llm_interpreter, _extract_followup_answers
from utils.config import DEFAULT_CASE, POLICY_JSON_PATH
from utils.validator import validate_case
from nodes.validation_node import validation_node
from nodes.policy_node import policy_node
from nodes.decision_node import decision_node
from nodes.explanation_node import explanation_node
from engine.memory_agent import apply_memory_to_matching_policies
from engine.policy_engine import (
    can_make_decision,
    get_fields_needed_for_decision,
    field_to_question,
    format_questions,
    policy_checker
)

# ── Session state initialization ─────────────────────────────
for key, default in {
    'step': 0,
    'results': {},
    'validation_issues': [],
    'corrected_data': {},
    'human_decision': None,
    'clarification_needed': False,
    'correction_needed': False,
    'awaiting_rejection_reason': False,
    'memory_hints': [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _merge_memory_into_matching():
    """Align UI with graph: stored facts may append MEM-RECORD before decision."""
    case_data = st.session_state.get("case_data") or {}
    mp = st.session_state.get("matching_policies") or []
    updated, _, hints = apply_memory_to_matching_policies(case_data, mp)
    st.session_state["matching_policies"] = updated
    st.session_state["memory_hints"] = hints


def _proceed_to_decision():
    """Helper to proceed to decision and explanation steps."""
    validation_result = validation_node({"case_data": st.session_state['case_data']})
    st.session_state['validation_result'] = validation_result

    if not validation_result["is_valid"]:
        st.session_state['correction_needed'] = True
        st.rerun()

    decision_result = decision_node({
        "validation_errors": st.session_state.get('validation_errors', []),
        "matching_policies": st.session_state['matching_policies']
    })
    st.session_state['decision'] = decision_result["decision"]

    with st.spinner("✍️ Generating explanation..."):
        explanation_result = explanation_node({
            "input": st.session_state['user_query'],
            "case_data": st.session_state['case_data'],
            "validation_errors": st.session_state.get('validation_errors', []),
            "matching_policies": st.session_state['matching_policies'],
            "memory_hints": st.session_state.get("memory_hints", []),
        })
        explanation_text = explanation_result.get("explanation", "")
    print("Explanation returned to UI:", explanation_text)
    st.session_state['explanation'] = explanation_text

    if decision_result["decision"] == "ESCALATE":
        st.session_state.step = 1
    else:
        st.session_state.step = 2

    st.rerun()


# ============================================================
# MAIN UI
# ============================================================

st.title("Policy-Aware Decision System")

# ── 1. Input Panel ──────────────────────────────────────────
st.header("1. Input Panel")
user_query = st.text_area(
    "Enter your request (e.g., 'I need help processing a policy request'):",
    height=100
)
submit_button = st.button("🚀 Submit Request")

if submit_button and user_query:
    for key in ['results', 'validation_issues', 'corrected_data', 'human_decision',
                'clarification_needed', 'correction_needed', 'awaiting_rejection_reason', 'memory_hints']:
        st.session_state[key] = {} if key == 'results' else [] if key in ('validation_issues', 'memory_hints') else \
                                  {} if key == 'corrected_data' else None if key == 'human_decision' else False

    st.session_state.step = 1
    st.session_state['user_query'] = user_query

    with st.spinner("🔍 Interpreting request..."):
        interpreter_result = llm_interpreter(user_query)
        time.sleep(1)

    st.session_state['case_data'] = interpreter_result["case_data"]
    st.session_state['validation_errors'] = interpreter_result["validation_errors"]

    with st.spinner("📋 Checking policies..."):
        policy_result = policy_node({"case_data": st.session_state['case_data']})
        st.session_state['matching_policies'] = policy_result["matching_policies"]
        _merge_memory_into_matching()
        time.sleep(1)

    if can_make_decision(st.session_state['case_data'], POLICY_JSON_PATH):
        _proceed_to_decision()
    else:
        if any(p.get('action') == 'REJECT' for p in st.session_state['matching_policies']):
            _proceed_to_decision()
        else:
            missing_fields = get_fields_needed_for_decision(
                st.session_state['case_data'], POLICY_JSON_PATH
            )
            if missing_fields:
                st.session_state['clarification_needed'] = True
                st.session_state['questions'] = format_questions([field_to_question(f) for f in missing_fields])
                st.session_state['missing_fields'] = missing_fields
                st.rerun()
            else:
                _proceed_to_decision()


# ── 2. Clarification Section ────────────────────────────────
if st.session_state.get('clarification_needed'):
    st.header("❓ Clarification Needed")
    st.write(st.session_state['questions'])
    clarification_answer = st.text_area("Your answer:", key="clarification")

    if st.button("Submit Clarification") and clarification_answer:
        followup_data = _extract_followup_answers(
            st.session_state['missing_fields'], clarification_answer
        )
        data = st.session_state.get('case_data', {})
        data.update(followup_data)

        validated_data, validation_errors = validate_case(data, DEFAULT_CASE)
        st.session_state['case_data'] = validated_data
        st.session_state['validation_errors'] = validation_errors
        st.session_state['clarification_needed'] = False

        if can_make_decision(st.session_state['case_data'], POLICY_JSON_PATH):
            _proceed_to_decision()
        else:
            missing_fields = get_fields_needed_for_decision(
                st.session_state['case_data'], POLICY_JSON_PATH
            )
            if missing_fields:
                st.session_state['questions'] = format_questions([field_to_question(f) for f in missing_fields])
                st.session_state['missing_fields'] = missing_fields
                st.rerun()
            else:
                _proceed_to_decision()


# ── 3. Validation Correction Section ────────────────────────
if st.session_state.get('correction_needed'):
    st.header("⚠️ Validation Issues")
    st.warning("Input seems invalid or unusual.")
    for issue in st.session_state['validation_result']["issues"]:
        st.write(f"- {issue}")
    if st.session_state['validation_result'].get("suggested_corrections"):
        st.write("Suggested Corrections:")
        st.json(st.session_state['validation_result']["suggested_corrections"])

    corrected_input = st.text_area(
        "Enter corrected JSON data:",
        value=json.dumps(st.session_state['case_data'], indent=2),
        height=200
    )
    if st.button("✅ Confirm Correction"):
        try:
            corrected_data = json.loads(corrected_input)
            st.session_state['case_data'] = corrected_data
            st.session_state['correction_needed'] = False

            policy_result = policy_node({"case_data": corrected_data})
            st.session_state['matching_policies'] = policy_result["matching_policies"]
            _merge_memory_into_matching()

            decision_result = decision_node({
                "validation_errors": [],
                "matching_policies": st.session_state['matching_policies']
            })
            st.session_state['decision'] = decision_result["decision"]

            with st.spinner("✍️ Generating explanation..."):
                explanation_result = explanation_node({
                    "input": st.session_state['user_query'],
                    "case_data": corrected_data,
                    "validation_errors": [],
                    "matching_policies": st.session_state['matching_policies'],
                    "memory_hints": st.session_state.get("memory_hints", []),
                })
            st.session_state['explanation'] = explanation_result.get("explanation", "")

            st.session_state['step'] = 1 if decision_result["decision"] == "ESCALATE" else 2
            st.rerun()
        except json.JSONDecodeError:
            st.error("Invalid JSON. Please correct and try again.")


# ── 4. Pipeline Visualization ────────────────────────────────
if st.session_state.step >= 1:
    st.header("2. Pipeline Visualization")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("🔍 Interpreter Output")
        if 'case_data' in st.session_state:
            st.json(st.session_state['case_data'])

    with col2:
        st.subheader("✅ Validation Result")
        if 'validation_result' in st.session_state:
            val = st.session_state['validation_result']
            if val['is_valid']:
                st.success("Valid")
            else:
                st.warning("Issues Found")
                for issue in val['issues']:
                    st.write(f"- {issue}")

    with col3:
        st.subheader("📋 Policy Matches")
        if 'matching_policies' in st.session_state:
            policies = st.session_state['matching_policies']
            if policies:
                for p in policies:
                    st.write(f"- **{p['policy_id']}:** {p['message']}")
            else:
                st.write("No policies triggered")

    with col4:
        st.subheader("⚖️ Decision")
        if 'decision' in st.session_state:
            decision = st.session_state['decision']
            if decision == "APPROVED":
                st.success("✅ APPROVED")
            elif decision == "REJECTED":
                st.error("❌ REJECTED")
            elif decision == "ESCALATE":
                st.warning("⚠️ ESCALATE")
            else:
                st.info(decision)


# ── 5. Human-in-the-Loop Panel ───────────────────────────────
if st.session_state.get('decision') == "ESCALATE" and st.session_state.step == 1:
    st.header("4. Human-in-the-Loop Panel")
    st.info("⚠️ This request has been escalated for human review.")

    st.write("**Original Query:**", st.session_state.get('user_query', ''))
    st.write("**Structured Data:**")
    st.json(st.session_state.get('case_data', {}))
    st.write("**Triggered Policies:**")
    for p in st.session_state.get('matching_policies', []):
        st.write(f"- **{p['policy_id']}:** {p['message']}")

    if st.session_state.get('awaiting_rejection_reason'):
        st.warning("Please provide a reason for rejection before confirming.")
        rejection_reason = st.text_area(
            "Rejection Reason (required):",
            placeholder="e.g. This request does not meet current policy requirements.",
            key="rejection_reason_input"
        )
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("🚫 Confirm Rejection"):
                if not rejection_reason.strip():
                    st.error("A reason is required to reject.")
                else:
                    st.session_state['human_decision'] = "REJECTED"
                    st.session_state['decision'] = "REJECTED"
                    st.session_state['awaiting_rejection_reason'] = False

                    # LLM elaborates on the human's reason
                    with st.spinner("✍️ Generating explanation..."):
                        explanation_result = explanation_node({
                            "input": st.session_state['user_query'],
                            "case_data": st.session_state['case_data'],
                            "validation_errors": [],
                            "matching_policies": st.session_state['matching_policies'],
                            "human_reason": rejection_reason.strip(),
                            "memory_hints": st.session_state.get("memory_hints", []),
                        })
                    st.session_state['explanation'] = explanation_result.get("explanation", "")
                    st.session_state.step = 2
                    st.rerun()
        with col_cancel:
            if st.button("↩️ Go Back"):
                st.session_state['awaiting_rejection_reason'] = False
                st.rerun()
    else:
        col_approve, col_reject = st.columns(2)
        with col_approve:
            if st.button("✅ Approve"):
                st.session_state['human_decision'] = "APPROVED"
                st.session_state['decision'] = "APPROVED"
                with st.spinner("✍️ Generating explanation..."):
                    explanation_result = explanation_node({
                        "input": st.session_state['user_query'],
                        "case_data": st.session_state['case_data'],
                        "validation_errors": [],
                        "matching_policies": [],
                        "memory_hints": st.session_state.get("memory_hints", []),
                    })
                st.session_state['explanation'] = explanation_result.get("explanation", "Request approved by reviewer.")
                st.session_state.step = 2
                st.rerun()
        with col_reject:
            if st.button("❌ Reject"):
                st.session_state['awaiting_rejection_reason'] = True
                st.rerun()


# ── 6. Final Output Panel ────────────────────────────────────
if st.session_state.step >= 2 and 'explanation' in st.session_state:
    st.header("3. Final Output Panel")
    decision = st.session_state['decision']
    explanation = st.session_state['explanation']

    if decision == "APPROVED":
        st.success("**Final Decision: ✅ APPROVED**")
    elif decision == "REJECTED":
        st.error("**Final Decision: ❌ REJECTED**")
    elif decision == "ESCALATE":
        st.warning("**Final Decision: ⚠️ ESCALATE (Pending Review)**")
    else:
        st.info(f"**Final Decision:** {decision}")

    st.divider()
    st.subheader("📝 Explanation")
    st.write(explanation)