from typing import TypedDict, Optional
from fellowai.models.paper import ResearchProject

class GraphState(TypedDict):
    """Represents the state of our LangGraph workflow."""
    pdf_source: str
    local_pdf_path: Optional[str]
    project: Optional[ResearchProject]
    error: Optional[str]
    human_decision: Optional[str]
    skip_research: Optional[bool]
    engineering_output: Optional[dict]
    revision_count: int
    engineering_feedback: Optional[str]
