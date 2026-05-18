from typing import TypedDict, Dict, List, Optional, NotRequired


class AgentState(TypedDict):
    input: str
    case_data: Dict
    decision: str
    matching_policies: List
    explanation: Optional[str]
    memory_hints: NotRequired[List[str]]