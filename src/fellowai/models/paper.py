from typing import List, Optional
from pydantic import BaseModel, Field

class Citation(BaseModel):
    text: str
    source: str
    relevance: str

class PaperSummary(BaseModel):
    title: str
    abstract_summary: str
    key_findings: List[str]
    citations: List[Citation] = Field(default_factory=list)

class PaperMetadata(BaseModel):
    title: str
    authors: List[str]
    publication_date: Optional[str] = None
    url: Optional[str] = None

class ResearchProject(BaseModel):
    metadata: PaperMetadata
    summary: Optional[PaperSummary] = None
    recommendation: Optional[str] = None
