import os
from textwrap import dedent
from crewai import Agent, Task, Crew
from fellowai.llm import get_llm
from fellowai.models.paper import ArchitecturalPlan
from typing import Optional

from fellowai.tools.file_ops import CreateProjectDirTool, WriteFileTool, DiscoverPreviousWorkTool

class EngineeringTeam:
    """
    EngineeringTeam implements the end-to-end prototype based on the architectural plan.
    """

    def __init__(self, llm=None):
        self.llm = llm or get_llm(temperature=0.0)
        debug_mode = os.environ.get("DEBUG_TOOLS", "0") == "1"
        
        # Tools
        self.create_dir_tool = CreateProjectDirTool()
        self.write_file_tool = WriteFileTool()
        self.discover_tool = DiscoverPreviousWorkTool()
        
        # Agent: Machine Learning Engineer
        self.ml_engineer = Agent(
            role='Principal Machine Learning Engineer',
            goal='Implement robust, best-practice PyTorch model architectures and training loops based on architectural plans.',
            backstory='You are an expert ML Engineer who writes extremely clear, optimized, and bug-free PyTorch code. You always follow best practices, ensure proper logging of performance metrics (such as loss, accuracy, and throughput), and structure your code beautifully.',
            tools=[self.create_dir_tool, self.write_file_tool, self.discover_tool],
            llm=self.llm,
            verbose=debug_mode,
            allow_delegation=False
        )
        
        # Agent: Documentation Specialist
        self.doc_specialist = Agent(
            role='Technical Documentation Specialist',
            goal='Write crystal clear, step-by-step instructions on how to run, train, and test the machine learning codebase.',
            backstory='You are a meticulous documentation specialist. You read the code written by the engineers and write exactly what commands to run, what dependencies are needed, and how someone can reproduce the prototype training process.',
            tools=[self.discover_tool, self.write_file_tool],
            llm=self.llm,
            verbose=debug_mode,
            allow_delegation=False
        )

    def generate_prototype(self, project_slug: str, title: str, plan: ArchitecturalPlan, feedback: Optional[str] = None) -> str:
        plan_text = f"""
Data Strategy: {plan.data_strategy}

Model Implementation Plan: {plan.model_implementation_plan}
"""
        
        # Task 1: Discover and Scaffold
        scaffold_and_code_task = Task(
            description=dedent(f"""\
                We are building a prototype for the research paper '{title}'.
                The project slug is '{project_slug}'.
                
                First, use the Discover Previous Work Tool to see if any code already exists for '{project_slug}'.
                If it does NOT exist, use the Create Project Directory Tool to create the base folder.
                
                Then, based on the Architectural Plan below, write the complete PyTorch code.
                You should write at least a 'model.py' (containing the model architecture) and a 'train.py' (containing the dataset loading, training loop, and logging).
                Use the Write File Tool to save these files into the project folder. Ensure you are logging performance metrics!
                
                --- ARCHITECTURAL PLAN ---
                {plan_text}
                
                {f"--- RESEARCH ARCHITECT FEEDBACK ---\nThe plan reviewer rejected the last attempt. Please fix the following issues:\n{feedback}" if feedback else ""}
                """),
            expected_output="A confirmation string specifying which Python files were created or updated in the project directory.",
            agent=self.ml_engineer
        )
        
        # Task 2: Documentation
        documentation_task = Task(
            description=dedent(f"""\
                The Machine Learning Engineer has just finished updating the codebase for '{project_slug}'.
                
                Use the Discover Previous Work Tool to read the codebase (focus on 'model.py' and 'train.py').
                Then, write a comprehensive 'README.md' file that explains:
                1. What this prototype does.
                2. Requirements (assume Python 3.10+ and PyTorch).
                3. Exact instructions to run the training process.
                
                Use the Write File Tool to write 'README.md' to the '{project_slug}' folder.
                """),
            expected_output="A confirmation string that the README.md was successfully written.",
            agent=self.doc_specialist
        )
        
        debug_mode = os.environ.get("DEBUG_TOOLS", "0") == "1"
        crew = Crew(
            agents=[self.ml_engineer, self.doc_specialist],
            tasks=[scaffold_and_code_task, documentation_task],
            verbose=debug_mode
        )
        
        result = crew.kickoff()
        return result.raw
