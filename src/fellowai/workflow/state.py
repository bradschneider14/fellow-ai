from typing import TypedDict, Optional
from fellowai.models.paper import ResearchProject

class GraphState(TypedDict):
    """Represents the state of our LangGraph workflow."""
    raw_text: str
    project: Optional[ResearchProject]
    error: Optional[str]
