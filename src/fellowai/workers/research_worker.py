from fellowai.workers.base_worker import BaseRMQWorker
from fellowai.agents.lab_director import LabDirector
from fellowai.agents.research_analyst import ResearchAnalyst
from fellowai.agents.librarian import Librarian
from fellowai.models.paper import ResearchProject
import os
import tempfile
import requests
import re


def slugify(text: str) -> str:
    return re.sub(r'[\W_]+', '_', text.lower()).strip('_')


class ResearchWorker(BaseRMQWorker):
    '''
    A worker that processes research papers. Consumes from the initiate queue and publishes to the architect queue. Produces a research project file in the .reports directory.
    '''

    def __init__(self):
        '''
        Initialize the worker.
        '''
        super().__init__(listen_queue="initiate_queue", publish_queue="architect_queue")
        self.lab_director = LabDirector()
        self.research_analyst = ResearchAnalyst()
        self.librarian = Librarian()

    def process_message(self, payload: dict):
        '''
        Process a message. Expects a payload with a pdf_source key.

        Args:
            payload (dict): The payload to process.
        '''
        pdf_source = payload.get("pdf_source")
        if not pdf_source:
            raise ValueError("No pdf_source provided in payload")

        local_path = pdf_source
        if pdf_source.startswith("http://") or pdf_source.startswith("https://"):
            print(f"Downloading PDF from {pdf_source}...")
            response = requests.get(pdf_source)
            response.raise_for_status()
            fd, local_path = tempfile.mkstemp(suffix=".pdf")
            with os.fdopen(fd, 'wb') as f:
                f.write(response.content)

        # Initiate
        metadata = self.lab_director.extract_metadata(local_path)
        slug = slugify(metadata.title)

        project = ResearchProject(metadata=metadata)

        # Summarize
        summary = self.research_analyst.summarize_paper(
            project.metadata.title, local_path)
        project.summary = summary

        # Extract Citations
        extraction_context = []
        if project.summary:
            extraction_context.extend(project.summary.architecture_details)
            extraction_context.extend(project.summary.performance_metrics)

        citations = self.librarian.extract_citations(
            local_path, extraction_context)
        if project.summary:
            project.summary.citations.extend(citations)

        # Recommend
        summary_text = (
            f"Title: {project.summary.title}\n"
            f"Abstract: {project.summary.abstract_summary}\n"
            f"Architecture Details: {project.summary.architecture_details}\n"
            f"Performance Metrics: {project.summary.performance_metrics}"
        ) if project.summary else "No summary available."
        report = self.lab_director.make_recommendation(summary_text)
        project.report = report

        # Write to state (file-based)
        reports_dir = os.path.join(os.getcwd(), ".reports")
        os.makedirs(reports_dir, exist_ok=True)
        report_path = os.path.join(reports_dir, f"{slug}.json")
        with open(report_path, 'w') as f:
            f.write(project.model_dump_json(indent=2))

        print(f"Research done for {slug}. Passing to Architect.")
        next_payload = {
            "project_slug": slug,
            "static_retry_count": payload.get("static_retry_count", 0),
            "runtime_retry_count": payload.get("runtime_retry_count", 0)
        }
        self.publish_output_to_queue(next_payload)


if __name__ == "__main__":
    worker = ResearchWorker()
    worker.start_listening()
