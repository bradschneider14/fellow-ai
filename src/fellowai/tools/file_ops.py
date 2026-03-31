from crewai.tools import BaseTool
import os
import html
from pydantic import BaseModel, Field

class CreateProjectDirInput(BaseModel):
    project_slug: str = Field(..., description="The slugified name of the project folder to create.")

class CreateProjectDirTool(BaseTool):
    name: str = "Create Project Directory Tool"
    description: str = "Creates a directory under '.projects' for the given project slug if it doesn't already exist. Returns the path."
    args_schema: type[BaseModel] = CreateProjectDirInput

    def _run(self, project_slug: str) -> str:
        base_dir = os.path.join(os.getcwd(), ".projects", project_slug)
        try:
            os.makedirs(base_dir, exist_ok=True)
            return f"Project directory ready at: {base_dir}"
        except Exception as e:
            return f"Error creating project directory: {e}"

class WriteFileInput(BaseModel):
    project_slug: str = Field(..., description="The slugified name of the project folder.")
    file_name: str = Field(..., description="The name of the file to write (e.g., 'model.py', 'train.py', 'README.md').")
    content: str = Field(..., description="The complete string content to write to the file.")

class WriteFileTool(BaseTool):
    name: str = "Write File Tool"
    description: str = "Writes content to a file inside the '.projects/<project_slug>' directory. Used to generate code and documentation files."
    args_schema: type[BaseModel] = WriteFileInput

    def _run(self, project_slug: str, file_name: str, content: str) -> str:
        base_dir = os.path.join(os.getcwd(), ".projects", project_slug)
        if not os.path.exists(base_dir):
            return f"Error: Project directory {base_dir} does not exist. Call CreateProjectDirTool first."
            
        file_path = os.path.join(base_dir, file_name)
        try:
            with open(file_path, "w") as f:
                f.write(html.unescape(content))
            return f"Successfully wrote file to {file_path}"
        except Exception as e:
            return f"Error writing file: {e}"

class DiscoverPreviousWorkInput(BaseModel):
    project_slug: str = Field(..., description="The slugified name of the project folder to inspect.")

class DiscoverPreviousWorkTool(BaseTool):
    name: str = "Discover Previous Work Tool"
    description: str = "Scans a project directory and returns the list of existing files and their contents, allowing agents to resume work without deleting existing code."
    args_schema: type[BaseModel] = DiscoverPreviousWorkInput

    def _run(self, project_slug: str) -> str:
        base_dir = os.path.join(os.getcwd(), ".projects", project_slug)
        if not os.path.exists(base_dir):
            return "No previous work found. Directory does not exist."
            
        files_info = []
        for root, _, files in os.walk(base_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r") as f:
                        content = f.read()
                    # Limit content length in case of large files
                    if len(content) > 5000:
                        content = content[:5000] + "\n...[truncated]..."
                    files_info.append(f"--- File: {file} ---\n{content}\n")
                except Exception as e:
                    files_info.append(f"--- File: {file} (Error reading: {e}) ---\n")
                    
        if not files_info:
            return "Directory exists but is empty."
            
        return "Found the following existing files:\n\n" + "\n".join(files_info)
