import docker
import docker.errors
import os

class ExecutionAgent:
    def __init__(self):
        try:
            self.client = docker.from_env()
        except docker.errors.DockerException as e:
            print(f"Failed to connect to Docker daemon: {e}")
            self.client = None

    def execute_prototype(self, project_dir: str, entrypoint: str = "main.py") -> dict:
        """
        Executes the code in project_dir within a Docker container.
        Returns a dict with exit_code, logs, and success.
        """
        if not self.client:
            return {
                "exit_code": -1,
                "logs": "Docker is not available on the host machine.",
                "success": False
            }

        abs_project_dir = os.path.abspath(project_dir)
        if not os.path.exists(abs_project_dir):
            return {
                "exit_code": -1,
                "logs": f"Project directory {abs_project_dir} does not exist.",
                "success": False
            }

        # Check if entrypoint exists
        entrypoint_path = os.path.join(abs_project_dir, entrypoint)
        if not os.path.exists(entrypoint_path):
            return {
                "exit_code": -1,
                "logs": f"Entrypoint {entrypoint} not found in project directory.",
                "success": False
            }

        # Check for requirements.txt
        req_path = os.path.join(abs_project_dir, "requirements.txt")
        has_reqs = os.path.exists(req_path)

        command = f"sh -c '"
        if has_reqs:
            command += "pip install --no-cache-dir -r requirements.txt && "
        command += f"python {entrypoint}'"

        volumes = {
            abs_project_dir: {'bind': '/workspace', 'mode': 'rw'}
        }

        try:
            print(f"Running Docker container for {project_dir}...")
            # We use python:3.11-slim as a base
            container = self.client.containers.run(
                image="python:3.11-slim",
                command=command,
                volumes=volumes,
                working_dir="/workspace",
                detach=True,
                auto_remove=False # Wait for it to finish to get logs and status
            )

            result = container.wait(timeout=300) # 5 mins max
            exit_code = result['StatusCode']
            logs = container.logs().decode('utf-8')
            
            container.remove()

            return {
                "exit_code": exit_code,
                "logs": logs,
                "success": exit_code == 0
            }

        except Exception as e:
            return {
                "exit_code": -1,
                "logs": f"Docker execution failed: {str(e)}",
                "success": False
            }
