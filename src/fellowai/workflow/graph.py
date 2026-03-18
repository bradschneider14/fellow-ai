from langgraph.graph import StateGraph, START, END
import os
import tempfile
import requests
from fellowai.workflow.state import GraphState
from fellowai.agents.lab_director import LabDirector
from fellowai.agents.research_analyst import ResearchAnalyst
from fellowai.agents.librarian import Librarian
from fellowai.models.paper import ResearchProject

# Initialize the agents (these hold the CrewAI configurations)
lab_director = LabDirector()
research_analyst = ResearchAnalyst()
librarian = Librarian()

def initiate_project_node(state: GraphState) -> GraphState:
    print("--- Node: Lab Director Initiating Project ---")
    pdf_source = state["pdf_source"]
    local_path = pdf_source
    
    if pdf_source.startswith("http://") or pdf_source.startswith("https://"):
        print(f"Downloading PDF from {pdf_source}...")
        response = requests.get(pdf_source)
        response.raise_for_status()
        fd, local_path = tempfile.mkstemp(suffix=".pdf")
        with os.fdopen(fd, 'wb') as f:
            f.write(response.content)
            
    metadata = lab_director.extract_metadata(local_path)
    project = ResearchProject(metadata=metadata)
    return {"project": project, "local_pdf_path": local_path}

def summarize_paper_node(state: GraphState) -> GraphState:
    print("--- Node: Research Analyst Summarizing Paper ---")
    project = state["project"]
    summary = research_analyst.summarize_paper(project.metadata.title, state["local_pdf_path"])
    project.summary = summary
    return {"project": project}

def extract_citations_node(state: GraphState) -> GraphState:
    print("--- Node: Librarian Extracting Citations ---")
    project = state["project"]
    # Provide the summary findings and the raw text so Librarian knows what citations to look for
    citations = librarian.extract_citations(
        state["local_pdf_path"],
        project.summary.key_findings if project.summary else []
    )
    if project.summary:
        project.summary.citations.extend(citations)
    return {"project": project}

def recommend_node(state: GraphState) -> GraphState:
    print("--- Node: Lab Director Making Final Recommendation ---")
    project = state["project"]
    summary_text = (
        f"Title: {project.summary.title}\\n"
        f"Abstract: {project.summary.abstract_summary}\\n"
        f"Findings: {project.summary.key_findings}"
    ) if project.summary else "No summary available."
    
    recommendation = lab_director.make_recommendation(summary_text)
    project.recommendation = recommendation
    return {"project": project}

# Construct the graph
workflow = StateGraph(GraphState)

workflow.add_node("initiate", initiate_project_node)
workflow.add_node("summarize", summarize_paper_node)
workflow.add_node("citations", extract_citations_node)
workflow.add_node("recommend", recommend_node)

# Define the flow (linear for now)
workflow.add_edge(START, "initiate")
workflow.add_edge("initiate", "summarize")
workflow.add_edge("summarize", "citations")
workflow.add_edge("citations", "recommend")
workflow.add_edge("recommend", END)

# Compile into a runnable
app = workflow.compile()
