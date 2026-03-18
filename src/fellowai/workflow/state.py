from typing import TypedDict, Optional
from fellowai.models.paper import ResearchProject

class GraphState(TypedDict):
    """Represents the state of our LangGraph workflow."""
    pdf_source: str
    local_pdf_path: Optional[str]
    project: Optional[ResearchProject]
    error: Optional[str]
