from typing import List, Optional
from pydantic import BaseModel, Field

class Citation(BaseModel):
    text: str
    source: str
    relevance: str

class PaperSummary(BaseModel):
    title: str
    abstract_summary: str
    architecture_details: List[str]
    performance_metrics: List[str]
    citations: List[Citation] = Field(default_factory=list)

class PaperMetadata(BaseModel):
    title: str
    authors: List[str]
    publication_date: Optional[str] = None
    url: Optional[str] = None

class FinalReport(BaseModel):
    feasibility: str
    performance_metrics: List[str]
    implementation_details: List[str]
    pitfalls_and_ambiguities: List[str]

class ArchitecturalPlan(BaseModel):
    data_strategy: str
    model_implementation_plan: str

class ReviewStatus(BaseModel):
    is_approved: bool
    feedback: str

class EngineeringStatus(BaseModel):
    project_slug: str
    directory: str
    status_message: str

class ResearchProject(BaseModel):
    metadata: PaperMetadata
    summary: Optional[PaperSummary] = None
    report: Optional[FinalReport] = None
    architectural_plan: Optional[ArchitecturalPlan] = None
    engineering_status: Optional[EngineeringStatus] = None
