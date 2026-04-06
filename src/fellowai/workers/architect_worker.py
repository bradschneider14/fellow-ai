from fellowai.workers.base_worker import BaseRMQWorker
from fellowai.agents.research_architect import ResearchArchitect
from fellowai.models.paper import ResearchProject
import os


class ArchitectWorker(BaseRMQWorker):
    '''
    A worker that processes research papers. Consumes from the architect queue and publishes to the engineering queue. Updates the research project file in the .reports directory with a plan for implementation.
    '''

    def __init__(self):
        super().__init__(listen_queue="architect_queue", publish_queue="engineering_queue")
        self.research_architect = ResearchArchitect()

    def process_message(self, payload: dict):
        '''
        Process a message. Expects a payload with a project_slug key. The payload should contain the summary and report from the research worker.

        Args:
            payload (dict): The payload to process.
        '''
        slug = payload.get("project_slug")
        report_path = os.path.join(os.getcwd(), ".reports", f"{slug}.json")

        with open(report_path, 'r') as f:
            project = ResearchProject.model_validate_json(f.read())

        print(f"Drafting architecture plan for {slug}...")
        plan = self.research_architect.create_plan(
            project.summary, project.report)
        project.architectural_plan = plan

        with open(report_path, 'w') as f:
            f.write(project.model_dump_json(indent=2))

        self.publish_output_to_queue(payload)


if __name__ == "__main__":
    worker = ArchitectWorker()
    worker.start_listening()
