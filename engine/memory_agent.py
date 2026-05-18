"""
User memory: facts only (what already happened). Runs after policy matching, before decision.

Does not read policies.json. Reject copy must stay factual (no entitlement / policy language).
Rules and stored facts live in utils.config.MEMORY_AGENT — change store + reject_rules when you
switch domain (e.g. library: remember which book ids are unavailable).

Evaluation rows may pass `_memory_store_patch` (merged into the in-memory store for that call only)
so gold labels and run_system can match per-scenario remembered facts without changing global config.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from utils.config import MEMORY_AGENT


def _merge_store_patch(
    base: Optional[Dict[str, Any]], patch: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Deep-merge patch into base (top-level keys; nested dicts under same key merge shallowly)."""
    out = deepcopy(base) if isinstance(base, dict) else {}
    if not isinstance(patch, dict):
        return out
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            merged = {**out[k], **v}
            out[k] = merged
        else:
            out[k] = deepcopy(v) if isinstance(v, dict) else v
    return out


def effective_memory_store(store_patch: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Base MEMORY_AGENT store merged with optional per-call patch (e.g. evaluation gold)."""
    base = MEMORY_AGENT.get("store")
    if not isinstance(base, dict):
        base = {}
    return _merge_store_patch(base, store_patch)


def _get_at_path(root: Any, path: List[str]) -> Any:
    cur = root
    for key in path:
        if not isinstance(cur, dict):
            return None
        if key not in cur:
            return None
        cur = cur[key]
    return cur


def _when_matches(when: Dict[str, Any], case_data: Dict[str, Any]) -> bool:
    for key, expected in when.items():
        if case_data.get(key) != expected:
            return False
    return True


def _as_int(val: Any) -> Optional[int]:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _evaluate_rule(
    rule: Dict[str, Any], case_data: Dict[str, Any], store: Dict[str, Any]
) -> Optional[str]:
    """Return user-facing rejection message if rule fires, else None."""
    when = rule.get("when") or {}
    if not _when_matches(when, case_data):
        return None

    rtype = rule.get("type")
    if rtype == "sum_exceeds_cap":
        req_field = str(rule.get("request_field", "days_requested"))
        raw_req = case_data.get(req_field)
        requested = _as_int(raw_req)
        if requested is None or requested < 1:
            return None
        path = rule.get("fact_path")
        if not isinstance(path, list) or not path:
            return None
        stored = _as_int(_get_at_path(store, path))
        if stored is None:
            return None
        cap = _as_int(rule.get("cap"))
        if cap is None:
            return None
        if stored + requested <= cap:
            return None
        tpl = str(
            rule.get(
                "message",
                "Our records show you already took {stored} unit(s); this request cannot be added.",
            )
        )
        return tpl.format(stored=stored, requested=requested, cap=cap)

    if rtype == "case_member_of_fact_list":
        case_field = str(rule.get("case_field", ""))
        if not case_field:
            return None
        value = case_data.get(case_field)
        if value is None or value == "":
            return None
        path = rule.get("fact_path")
        if not isinstance(path, list) or not path:
            return None
        fact = _get_at_path(store, path)
        if not isinstance(fact, list):
            return None
        if value not in fact:
            return None
        return str(rule.get("message", "Our records show that item is not available right now."))

    return None


def build_memory_rejection(
    case_data: Dict[str, Any], store_patch: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    If any configured memory rule fires, return a synthetic matching-policy-shaped dict.
    """
    if not MEMORY_AGENT.get("enabled", False):
        return None

    store = effective_memory_store(store_patch)
    rules = MEMORY_AGENT.get("reject_rules") or []
    if not isinstance(rules, list):
        return None

    for rule in rules:
        if not isinstance(rule, dict):
            continue
        msg = _evaluate_rule(rule, case_data, store)
        if msg:
            return {
                "policy_id": str(MEMORY_AGENT.get("policy_id", "MEM-RECORD")),
                "violation": None,
                "action": "REJECT",
                "message": msg,
                "type": "memory",
            }
    return None


def collect_memory_remaining_hints(
    case_data: Dict[str, Any], store_patch: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    When memory does not block the request, add a short factual note about the informal
    period total: what was already on file before this request, what this approval adds, and
    how many days would remain only for separate future requests after this one is counted.
    """
    if not MEMORY_AGENT.get("enabled", False):
        return []

    store = effective_memory_store(store_patch)
    hints: List[str] = []
    seen: set[str] = set()

    for rule in MEMORY_AGENT.get("reject_rules") or []:
        if not isinstance(rule, dict):
            continue
        if rule.get("type") != "sum_exceeds_cap":
            continue

        when = rule.get("when") or {}
        if not _when_matches(when, case_data):
            continue

        path = rule.get("fact_path")
        if not isinstance(path, list) or not path:
            continue
        stored = _as_int(_get_at_path(store, path))
        cap = _as_int(rule.get("cap"))
        if stored is None or cap is None:
            continue

        req_field = str(rule.get("request_field", "days_requested"))
        req = _as_int(case_data.get(req_field))
        if req is None or req < 1:
            continue

        if stored + req > cap:
            continue

        label = str(rule.get("kind_label", "casual")).strip() or "casual"
        after_this = max(0, cap - stored - req)
        key = f"{label}:{cap}:{stored}:{req}:{after_this}"
        if key in seen:
            continue
        seen.add(key)

        if after_this == 0:
            text = (
                f"Memory note (informal planning count for this period): we keep a running total of "
                f"up to {cap} {label} day(s) combined. Our records already showed {stored} taken before this request. "
                f"This approval covers {req} day(s). Once that is included in the same total, you would have "
                f"no {label} days left to schedule for any further leave in this period under that same informal total."
            )
        elif after_this == 1:
            text = (
                f"Memory note (informal planning count for this period): we keep a running total of "
                f"up to {cap} {label} day(s) combined. Our records showed {stored} taken before this request, "
                f"and this approval adds {req}. After that total is updated, you would still have 1 {label} day "
                f"available only for a separate future request — not as an extra day on top of the leave you "
                f"were just granted."
            )
        else:
            text = (
                f"Memory note (informal planning count for this period): we keep a running total of "
                f"up to {cap} {label} day(s) combined. Our records showed {stored} taken before this request, "
                f"and this approval adds {req}. After that total is updated, you would still have {after_this} "
                f"{label} days available only for further separate requests this period — not stacked on top of "
                f"the days in this approval."
            )
        hints.append(text)

    return hints


def apply_memory_to_matching_policies(
    case_data: Dict[str, Any],
    matching_policies: List[Dict[str, Any]],
    store_patch: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], bool, List[str]]:
    """
    Returns (updated matching_policies, memory_rejected, memory_hints).
    Policies have already been evaluated; this appends a factual memory block if needed,
    and supplies optional remaining-day hints when the request is not blocked by memory.
    """
    hints = collect_memory_remaining_hints(case_data, store_patch)

    block = build_memory_rejection(case_data, store_patch)
    if not block:
        return matching_policies, False, hints

    out = deepcopy(matching_policies) if matching_policies else []
    out.append(block)
    return out, True, []
