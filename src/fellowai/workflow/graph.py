from langgraph.graph import StateGraph, START, END
import os
import tempfile
import requests
import re
from fellowai.workflow.state import GraphState
from fellowai.agents.lab_director import LabDirector
from fellowai.agents.research_analyst import ResearchAnalyst
from fellowai.agents.librarian import Librarian
from fellowai.agents.research_architect import ResearchArchitect
from fellowai.agents.engineering_team import EngineeringTeam
from fellowai.models.paper import ResearchProject

NODE_INITIATE = "initiate"
NODE_SUMMARIZE = "summarize"
NODE_CITATIONS = "citations"
NODE_RECOMMEND = "recommend"
NODE_HUMAN_DECISION = "human_decision"
NODE_ARCHITECTURAL_PLAN = "architectural_plan"
NODE_ENGINEERING_TEAM = "engineering_team"
NODE_REVIEW_TEAM = "review_team"

MAX_REVISIONS = 3

# Initialize the agents (these hold the CrewAI configurations)
lab_director = LabDirector()
research_analyst = ResearchAnalyst()
librarian = Librarian()
research_architect = ResearchArchitect()
engineering_team = EngineeringTeam()

def slugify(text: str) -> str:
    return re.sub(r'[\W_]+', '_', text.lower()).strip('_')

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
    
    reports_dir = os.path.join(os.getcwd(), ".reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, f"{slugify(metadata.title)}.json")
    
    if os.path.exists(report_path):
        choice = input(f"\n[CACHE] Existing report found for '{metadata.title}'. Skip to decision point? (y/n/skip): ").strip().lower()
        if choice in ['y', 'yes', 'skip']:
            print("Loading cached report from disk...")
            with open(report_path, 'r') as f:
                json_data = f.read()
                project = ResearchProject.model_validate_json(json_data)
            return {"project": project, "local_pdf_path": local_path, "skip_research": True}
    
    project = ResearchProject(metadata=metadata)
    return {"project": project, "local_pdf_path": local_path, "skip_research": False, "revision_count": 0}

def route_after_initiate(state: GraphState) -> str:
    if state.get("skip_research"):
        return NODE_HUMAN_DECISION
    return NODE_SUMMARIZE

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
    extraction_context = []
    if project.summary:
        extraction_context.extend(project.summary.architecture_details)
        extraction_context.extend(project.summary.performance_metrics)
        
    citations = librarian.extract_citations(
        state["local_pdf_path"],
        extraction_context
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
        f"Architecture Details: {project.summary.architecture_details}\\n"
        f"Performance Metrics: {project.summary.performance_metrics}"
    ) if project.summary else "No summary available."
    
    report = lab_director.make_recommendation(summary_text)
    project.report = report
    
    reports_dir = os.path.join(os.getcwd(), ".reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, f"{slugify(project.metadata.title)}.json")
    with open(report_path, 'w') as f:
        f.write(project.model_dump_json(indent=2))
    print(f"Saved project state to {report_path}")
    
    return {"project": project}

def human_decision_node(state: GraphState) -> GraphState:
    print("--- Node: Human-in-the-loop Decision ---")
    choice = input("\nDo we proceed with a prototype? (y/n): ").strip().lower()
    if choice in ['y', 'yes']:
        return {"human_decision": "proceed"}
    else:
        return {"human_decision": "reject"}

def route_after_human(state: GraphState) -> str:
    if state.get("human_decision") == "proceed":
        return NODE_ARCHITECTURAL_PLAN
    return END

def architectural_plan_node(state: GraphState) -> GraphState:
    print("--- Node: Research Architect Drafting Plan ---")
    project = state.get("project")
    plan = research_architect.create_plan(project.summary, project.report)
    project.architectural_plan = plan
    return {"project": project}

def engineering_team_node(state: GraphState) -> GraphState:
    print("--- Node: Engineering Team Building Prototype ---")
    project = state.get("project")
    slug = slugify(project.metadata.title)
    feedback = state.get("engineering_feedback", None)
    
    # Run the engineering team
    result = engineering_team.generate_prototype(
        project_slug=slug,
        title=project.metadata.title,
        plan=project.architectural_plan,
        feedback=feedback
    )
    
    from fellowai.models.paper import EngineeringStatus
    project.engineering_status = EngineeringStatus(
        project_slug=slug,
        directory=os.path.join(os.getcwd(), ".projects", slug),
        status_message=result
    )
    
    reports_dir = os.path.join(os.getcwd(), ".reports")
    report_path = os.path.join(reports_dir, f"{slug}.json")
    with open(report_path, 'w') as f:
        f.write(project.model_dump_json(indent=2))
        
    return {"project": project, "engineering_output": {"status": "success"}}

def review_team_node(state: GraphState) -> GraphState:
    print("--- Node: Engineering Code Review ---")
    project = state.get("project")
    slug = slugify(project.metadata.title)
    
    # Review code
    review_status = research_architect.review_code(slug, project.architectural_plan)
    
    print(f"Review decision: {'APPROVED' if review_status.is_approved else 'REJECTED'}")
    
    # Increment revision count
    current_revisions = state.get("revision_count", 0) + 1
    
    return {
        "revision_count": current_revisions, 
        "engineering_feedback": review_status.feedback,
        "engineering_output": {"status": "approved" if review_status.is_approved else "rejected"}
    }

def route_after_review(state: GraphState) -> str:
    status = state.get("engineering_output", {}).get("status")
    revision_count = state.get("revision_count", 0)
    
    if status == "approved":
        print("Code approved! Ending workflow.")
        return END
    elif revision_count >= MAX_REVISIONS:
        print(f"Max revisions ({MAX_REVISIONS}) reached. Ending workflow despite rejection.")
        return END
    else:
        print(f"Code rejected. Routing back to Engineering Team (Revision {revision_count}/{MAX_REVISIONS}).")
        return NODE_ENGINEERING_TEAM

# Construct the graph
workflow = StateGraph(GraphState)

workflow.add_node(NODE_INITIATE, initiate_project_node)
workflow.add_node(NODE_SUMMARIZE, summarize_paper_node)
workflow.add_node(NODE_CITATIONS, extract_citations_node)
workflow.add_node(NODE_RECOMMEND, recommend_node)
workflow.add_node(NODE_HUMAN_DECISION, human_decision_node)
workflow.add_node(NODE_ARCHITECTURAL_PLAN, architectural_plan_node)
workflow.add_node(NODE_ENGINEERING_TEAM, engineering_team_node)
workflow.add_node(NODE_REVIEW_TEAM, review_team_node)

# Define the flow
workflow.add_edge(START, NODE_INITIATE)
workflow.add_conditional_edges(NODE_INITIATE, route_after_initiate)
workflow.add_edge(NODE_SUMMARIZE, NODE_CITATIONS)
workflow.add_edge(NODE_CITATIONS, NODE_RECOMMEND)
workflow.add_edge(NODE_RECOMMEND, NODE_HUMAN_DECISION)
workflow.add_conditional_edges(NODE_HUMAN_DECISION, route_after_human)
workflow.add_edge(NODE_ARCHITECTURAL_PLAN, NODE_ENGINEERING_TEAM)
workflow.add_edge(NODE_ENGINEERING_TEAM, NODE_REVIEW_TEAM)
workflow.add_conditional_edges(NODE_REVIEW_TEAM, route_after_review)

# Compile into a runnable
app = workflow.compile()
