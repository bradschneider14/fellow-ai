from crewai.events.utils.console_formatter import set_suppress_console_output
import logging
from fellowai.workflow.graph import app
import os
import sys
import datetime
from dotenv import load_dotenv

# Load environment variables from .env file FIRST; Must happen before importing app!
# This is because app imports llm, which imports os.environ
load_dotenv()

class TeeStream:
    def __init__(self, stream, log_filename):
        self.stream = stream
        self.file = open(log_filename, 'a', encoding='utf-8')

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()
        
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

def setup_logging():
    os.makedirs(".runs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(".runs", f"run_{timestamp}.log")
    
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    print(f"Logging initialized. Output is being saved to {log_file}")

def main():
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("XAI_API_KEY"):
        print("Please set the OPENAI_API_KEY or XAI_API_KEY environment variable to run the agents.")
        print("Example: export XAI_API_KEY='xai-...'")
        return

    setup_logging()

    print("--- FellowAI: LangGraph + CrewAI Workflow ---")

    # Pass the URL to the famous "Attention Is All You Need" paper
    # sample_pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"
    sample_pdf_url = "https://arxiv.org/pdf/2506.06718"
    # sample_pdf_url = "https://ietresearch.onlinelibrary.wiley.com/doi/pdfdirect/10.1049/htl.2019.0015"

    initial_state = {
        "pdf_source": sample_pdf_url,
        "local_pdf_path": None,
        "project": None,
        "error": None
    }

    print("Invoking graph...")
    final_state = app.invoke(initial_state)

    project = final_state.get("project")
    if project:
        print("\\n=== FINAL RESULT ===")
        print(f"Title: {project.metadata.title}")
        print(f"Authors: {project.metadata.authors}")
        if project.summary:
            print(f"Summary: {project.summary.abstract_summary}")
            print("\\nArchitecture Details:")
            for detail in project.summary.architecture_details:
                print(f"  - {detail}")
            print("\\nPerformance Metrics:")
            for metric in project.summary.performance_metrics:
                print(f"  - {metric}")

            print(f"\\nCitations Extracted: {len(project.summary.citations)}")
            for c in project.summary.citations:
                print(f"  - [{c.source}]: {c.text} ({c.relevance})")

        if project.report:
            print("\\n=== ENGINEERING FEASIBILITY REPORT ===")
            print(f"Feasibility: {project.report.feasibility}")
            print("\\nImplementation Details (Synthesis):")
            for detail in project.report.implementation_details:
                print(f"  - {detail}")
            print("\\nPerformance Metrics (Synthesis):")
            for metric in project.report.performance_metrics:
                print(f"  - {metric}")
            print("\\nPitfalls & Ambiguities:")
            for pitfall in project.report.pitfalls_and_ambiguities:
                print(f"  - {pitfall}")

        if project.architectural_plan:
            print("\\n=== ARCHITECTURAL PLAN ===")
            print("\\n--- Data Strategy ---")
            print(project.architectural_plan.data_strategy)
            print("\\n--- Model Implementation Plan ---")
            print(project.architectural_plan.model_implementation_plan)
            
        if project.engineering_status:
            print("\\n=== ENGINEERING PROTOTYPE ===")
            print(f"Project Slug: {project.engineering_status.project_slug}")
            print(f"Directory: {project.engineering_status.directory}")
            print(f"\\nStatus Message:\\n{project.engineering_status.status_message}")


if __name__ == "__main__":
    main()
