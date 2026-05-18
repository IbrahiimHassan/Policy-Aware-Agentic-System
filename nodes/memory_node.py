"""
Graph node: applies remembered facts after policy matching, before decision.
"""
from engine.memory_agent import apply_memory_to_matching_policies


def memory_node(state):
    print("\n [NODE] Memory (stored facts)")
    case_data = state.get("case_data") or {}
    current = state.get("matching_policies") or []
    updated, blocked, hints = apply_memory_to_matching_policies(case_data, current)
    if blocked:
        print("  Memory: a stored fact blocks this request (see MEM-RECORD message).")
    else:
        print("  Memory: no blocking fact.")
    return {"matching_policies": updated, "memory_hints": hints}
