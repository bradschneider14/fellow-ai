from fellowai.workers.base_worker import BaseRMQWorker
from fellowai.agents.engineering_team import EngineeringTeam
from fellowai.models.paper import ResearchProject, EngineeringStatus
import os


class EngineeringWorker(BaseRMQWorker):
    '''
    A worker that processes research papers. Consumes from the engineering queue and publishes to the static review queue. Updates the research project file in the .reports directory with a plan for implementation.
    '''

    def __init__(self):
        super().__init__(listen_queue="engineering_queue",
                         publish_queue="static_review_queue")
        self.engineering_team = EngineeringTeam()

    def process_message(self, payload: dict):
        '''
        Process a message. Expects a payload with a project_slug key. The payload should contain the summary and report from the research worker as well as the implementation plan from the architect worker.

        Args:
            payload (dict): The payload to process.
        '''
        slug = payload.get("project_slug")
        feedback = payload.get("engineering_feedback")

        report_path = os.path.join(os.getcwd(), ".reports", f"{slug}.json")
        with open(report_path, 'r') as f:
            project = ResearchProject.model_validate_json(f.read())

        print(f"Engineering Team running for {slug}...")
        result = self.engineering_team.generate_prototype(
            project_slug=slug,
            title=project.metadata.title,
            plan=project.architectural_plan,
            feedback=feedback
        )

        project.engineering_status = EngineeringStatus(
            project_slug=slug,
            directory=os.path.join(os.getcwd(), ".projects", slug),
            status_message=result
        )

        with open(report_path, 'w') as f:
            f.write(project.model_dump_json(indent=2))

        next_payload = {
            "project_slug": slug,
            "static_retry_count": payload.get("static_retry_count", 0),
            "runtime_retry_count": payload.get("runtime_retry_count", 0)
        }
        self.publish_output_to_queue(next_payload)


if __name__ == "__main__":
    worker = EngineeringWorker()
    worker.start_listening()
