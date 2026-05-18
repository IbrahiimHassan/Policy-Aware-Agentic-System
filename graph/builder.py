from typing import Literal

from langgraph.graph import StateGraph

from graph.state import AgentState

from nodes.interpreter_node import interpreter_node
from nodes.validation_node import validation_node
from nodes.policy_node import policy_node
from nodes.decision_node import decision_node
from nodes.explanation_node import explanation_node
from nodes.human_review_node import human_review_node
from nodes.memory_node import memory_node


def route_decision(
    state,
) -> Literal["Accepted", "Rejected", "NEED_MORE_INFO", "Escalate"]:
    """Edge labels shown on graph.png (Mermaid)."""
    d = state.get("decision", "")
    if d in ("REJECTED", "TERMINATED"):
        return "Rejected"
    if d == "NEED_MORE_INFO":
        return "NEED_MORE_INFO"
    if d == "ESCALATE":
        return "Escalate"
    return "Accepted"


def build_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("interpreter", interpreter_node)
    graph.add_node("validation", validation_node)
    graph.add_node("policy_matcher", policy_node)
    graph.add_node("memory", memory_node)
    graph.add_node("decision", decision_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("explanation", explanation_node)

    # Flow
    graph.set_entry_point("interpreter")

    graph.add_edge("interpreter", "validation")
    graph.add_edge("validation", "policy_matcher")
    graph.add_edge("policy_matcher", "memory")
    graph.add_edge("memory", "decision")
    graph.add_conditional_edges(
        "decision", 
        route_decision, 
        path_map={
            "Accepted": "explanation",
            "Rejected": "explanation",
            # LangGraph cannot target START; interpreter is the first node after __start__
            "NEED_MORE_INFO": "interpreter",
            "Escalate": "human_review",
        },
    )
    graph.add_edge("human_review", "explanation")

    graph.set_finish_point("explanation")

    return graph.compile()


def build_documentation_mermaid() -> str:
    """
    Full decision-branch labels for graph.png.

    LangGraph's auto draw_mermaid() collapses multiple edges to the same node — use this
    for documentation instead. NEED_MORE_INFO loops back to __start__ (graph entry).
    """
    return """---
config:
  flowchart:
    curve: linear
---
graph TD;
\t__start__([<p>__start__</p>]):::first
\tinterpreter(interpreter)
\tvalidation(validation)
\tpolicy_matcher(policy_matcher)
\tmemory(memory)
\tdecision(decision)
\thuman_review(human_review)
\texplanation(explanation)
\t__end__([<p>__end__</p>]):::last
\t__start__ --> interpreter;
\tinterpreter --> validation;
\tvalidation --> policy_matcher;
\tpolicy_matcher --> memory;
\tmemory --> decision;
\tdecision -. Accepted .-> explanation;
\tdecision -. Rejected .-> explanation;
\tdecision -. NEED_MORE_INFO .-> __start__;
\tdecision -. Escalate .-> human_review;
\thuman_review --> explanation;
\texplanation --> __end__;
\tclassDef default fill:#f2f0ff,line-height:1.2
\tclassDef first fill-opacity:0
\tclassDef last fill:#bfb6fc
"""


def save_graph_png(path: str = "graph.png") -> None:
    """Write graph.png with all four decision edge labels (see build_documentation_mermaid)."""
    from pathlib import Path

    from langchain_core.runnables.graph_mermaid import draw_mermaid_png

    png_bytes = draw_mermaid_png(mermaid_syntax=build_documentation_mermaid())
    Path(path).write_bytes(png_bytes)