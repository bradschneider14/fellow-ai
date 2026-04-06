from fellowai.workers.base_worker import BaseRMQWorker
from fellowai.agents.research_architect import ResearchArchitect
from fellowai.models.paper import ResearchProject
import os

MAX_RETRIES = 3

class RuntimeReviewWorker(BaseRMQWorker):
    '''
    A worker that reviews the execution results for a research project against the research summary and report. Consumes from the runtime review queue and publishes to the completed queue if approved or the engineering queue if rejected.
    '''
    def __init__(self):
        super().__init__(listen_queue="runtime_review_queue")
        self.research_architect = ResearchArchitect()

    def process_message(self, payload: dict):
        '''
        Process a message. Expects a payload with a project_slug key. The payload should contain the summary and report from the research worker as well as the execution results from the execution worker.

        Args:
            payload (dict): The payload to process.
        '''
        slug = payload.get("project_slug")
        report_path = os.path.join(os.getcwd(), ".reports", f"{slug}.json")
        with open(report_path, 'r') as f:
            project = ResearchProject.model_validate_json(f.read())
            
        print(f"Runtime Review running for {slug}...")
        exec_logs = project.execution_result
        plan_with_logs = project.architectural_plan.model_copy()
        
        if exec_logs:
            plan_with_logs.model_implementation_plan += \
            f"\n\n--- EXECUTION LOGS ---\nExit Code: {exec_logs.exit_code}\nSuccess: {exec_logs.success}\nLogs:\n{exec_logs.logs}"
            
        review_status = self.research_architect.review_code(slug, plan_with_logs)
        
        runtime_retry_count = payload.get("runtime_retry_count", 0)
        
        if review_status.is_approved:
            print("Runtime review APPROVED. Workflow completed.")
            next_queue = "completed_queue"
        elif runtime_retry_count >= MAX_RETRIES:
            print(f"Runtime review REJECTED and MAX_RETRIES ({MAX_RETRIES}) reached. Workflow aborted.")
            next_queue = "completed_queue"
        else:
            print("Runtime review REJECTED. Returning to engineering with runtime logs.")
            runtime_retry_count += 1
            next_queue = "engineering_queue"
            
        next_payload = {
            "project_slug": slug,
            "static_retry_count": payload.get("static_retry_count", 0),
            "runtime_retry_count": runtime_retry_count,
            "engineering_feedback": review_status.feedback if not review_status.is_approved else None
        }
        self.publish_output_to_queue(next_payload, queue_name=next_queue)


if __name__ == "__main__":
    worker = RuntimeReviewWorker()
    worker.start_listening()
