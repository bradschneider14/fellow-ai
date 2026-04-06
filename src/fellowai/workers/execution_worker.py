from fellowai.workers.base_worker import BaseRMQWorker
from fellowai.agents.execution_agent import ExecutionAgent
from fellowai.models.paper import ResearchProject, ExecutionResult
import os


class ExecutionWorker(BaseRMQWorker):
    '''
    A worker that processes research papers. Consumes from the execution queue and publishes to the runtime review queue. Executes the prototype and updates the research project file in the .reports directory with the execution results.
    '''

    def __init__(self):
        super().__init__(listen_queue="execution_queue",
                         publish_queue="runtime_review_queue")
        self.execution_agent = ExecutionAgent()

    def process_message(self, payload: dict):
        '''
        Process a message. Expects a payload with a project_slug key. Payload should point to a project with executable code.

        Args:
            payload (dict): The payload to process.
        '''
        slug = payload.get("project_slug")
        report_path = os.path.join(os.getcwd(), ".reports", f"{slug}.json")
        with open(report_path, 'r') as f:
            project = ResearchProject.model_validate_json(f.read())

        print(f"Execution Engine running prototype for {slug}...")
        project_dir = os.path.join(os.getcwd(), ".projects", slug)
        result = self.execution_agent.execute_prototype(
            project_dir=project_dir)

        project.execution_result = ExecutionResult(**result)
        with open(report_path, 'w') as f:
            f.write(project.model_dump_json(indent=2))

        next_payload = {
            "project_slug": slug,
            "static_retry_count": payload.get("static_retry_count", 0),
            "runtime_retry_count": payload.get("runtime_retry_count", 0)
        }
        self.publish_output_to_queue(next_payload)


if __name__ == "__main__":
    worker = ExecutionWorker()
    worker.start_listening()
