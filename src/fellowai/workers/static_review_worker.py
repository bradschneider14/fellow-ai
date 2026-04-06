from fellowai.workers.base_worker import BaseRMQWorker
from fellowai.agents.research_architect import ResearchArchitect
from fellowai.models.paper import ResearchProject
import os

MAX_RETRIES = 3

class StaticReviewWorker(BaseRMQWorker):
    '''
    A worker that reviews the architectural plan for a research project against the research summary and report. Consumes from the static review queue and publishes to the execution queue if approved or the engineering queue if rejected.
    '''
    def __init__(self):
        super().__init__(listen_queue="static_review_queue")
        self.research_architect = ResearchArchitect()

    def process_message(self, payload: dict):
        '''
        Process a message. Expects a payload with a project_slug key. The payload should contain the summary and report from the research worker as well as the implementation plan from the architect worker.

        Args:
            payload (dict): The payload to process.
        '''
        slug = payload.get("project_slug")
        report_path = os.path.join(os.getcwd(), ".reports", f"{slug}.json")
        with open(report_path, 'r') as f:
            project = ResearchProject.model_validate_json(f.read())
            
        print(f"Static Review running for {slug}...")
        review_status = self.research_architect.review_code(slug, project.architectural_plan)
        
        static_retry_count = payload.get("static_retry_count", 0)
        
        if review_status.is_approved:
            print("Static review APPROVED. Sending to execution.")
            next_queue = "execution_queue"
        elif static_retry_count >= MAX_RETRIES:
            print(f"Static review REJECTED and MAX_RETRIES ({MAX_RETRIES}) reached. Force ending workflow.")
            next_queue = "completed_queue"
        else:
            print("Static review REJECTED. Returning to engineering.")
            static_retry_count += 1
            next_queue = "engineering_queue"
            
        next_payload = {
            "project_slug": slug,
            "static_retry_count": static_retry_count,
            "runtime_retry_count": payload.get("runtime_retry_count", 0),
            "engineering_feedback": review_status.feedback if not review_status.is_approved else None
        }
        self.publish_output_to_queue(next_payload, queue_name=next_queue)


if __name__ == "__main__":
    worker = StaticReviewWorker()
    worker.start_listening()
