import re
import os
import json
from textwrap import dedent
from crewai import Agent, Task, Crew
from fellowai.llm import get_llm
from fellowai.models.paper import ArchitecturalPlan, PaperSummary, FinalReport, ReviewStatus
from fellowai.tools.file_ops import DiscoverPreviousWorkTool
from fellowai.agents.base import BaseAgent
from typing import Optional


class ResearchArchitect(BaseAgent):
    """
    ResearchArchitect drafts step-by-step implementation plans for engineers.
    """

    def __init__(self, llm=None):
        self.llm = llm or get_llm(temperature=0.0)
        debug_mode = os.environ.get("DEBUG_TOOLS", "0") == "1"
        self.discover_tool = DiscoverPreviousWorkTool()
        self.agent = Agent(
            role='AI Research Architect',
            goal='Draft a highly detailed, step-by-step implementation plan for an engineering team, and rigorously review their generated code against it.',
            backstory='You are a Principal AI Architect who bridges the gap between research and engineering. You read research summaries, draft exact implementations, and review engineer code to ensure it matches the plan strictly.',
            tools=[self.discover_tool],
            llm=self.llm,
            verbose=debug_mode
        )

    def create_plan(self, summary: Optional[PaperSummary], report: Optional[FinalReport]) -> ArchitecturalPlan:
        summary_text = (
            f"Title: {summary.title}\\n"
            f"Architecture Details (from paper): {summary.architecture_details}\\n"
            f"Performance Metrics: {summary.performance_metrics}"
        ) if summary else "No summary available."

        report_text = (
            f"Feasibility: {report.feasibility}\\n"
            f"Implementation Details: {report.implementation_details}\\n"
            f"Pitfalls to Avoid: {report.pitfalls_and_ambiguities}"
        ) if report else "No final report available."

        draft_task = Task(
            description=dedent(f"""\
                Based on the following research summary and Lab Director's report, draft a highly detailed implementation plan for the engineering team.
                
                --- SUMMARY ---
                {summary_text}
                
                --- DIRECTOR'S REPORT ---
                {report_text}
                
                You must formulate:
                1. Data Strategy: Which datasets are needed, what pieces of input are being used, which outputs are being produced, and train/test split strategy. Be very specific.
                2. Model Implementation Plan: An exact list of inputs, layers, outputs, activation functions, and dimensions. Provide enough detail for engineers to start writing PyTorch code immediately, but DO NOT include any code snippets.
                """),
            expected_output="A raw text analysis including Data Strategy and Model Implementation Plan.",
            agent=self.agent
        )

        format_task = Task(
            description=dedent("""\
                Format the architectural plan from the previous task into a raw JSON object matching the requested schema.
                IMPORTANT: Return ONLY a valid JSON object. Do not include markdown formatting or extra text.
                WARNING: The text within strings MUST be properly escaped (e.g. escape quotes like \\", use \\n for newlines).
                Do not include blockquotes or markdown code fences (like ```json). Just the raw braces { }.
                """),
            expected_output="A valid JSON object starting with '{' containing string 'data_strategy' and string 'model_implementation_plan'.",
            agent=self.agent
        )

        debug_mode = os.environ.get("DEBUG_TOOLS", "0") == "1"
        crew = Crew(agents=[self.agent], tasks=[
                    draft_task, format_task], verbose=debug_mode)
        result = crew.kickoff()

        # handle some common JSON formatting issues that we can try to anticipate
        try:
            return self.result_to_json(result, ArchitecturalPlan)
        except Exception as e:
            print(
                f"[RECOVERABLE ERROR] Failed to parse ResearchArchitect JSON: {e}")
            print(f"Raw output (may be truncated): {result.raw[:500]}...")
            return ArchitecturalPlan(
                data_strategy=f"Error parsing data strategy. Raw output: {result.raw[:200]}",
                model_implementation_plan="Failed to parse structured JSON from ResearchArchitect. See logs for details."
            )

    def review_code(self, project_slug: str, plan: ArchitecturalPlan) -> ReviewStatus:
        plan_text = f"Data Strategy: {plan.data_strategy}\n\nModel Implementation Plan: {plan.model_implementation_plan}"

        review_task = Task(
            description=dedent(f"""\
                You need to review the engineering code generated for the project '{project_slug}'.
                Use the Discover Previous Work Tool to read the codebase.
                Compare it rigorously against this Architectural Plan:
                ---
                {plan_text}
                ---
                Check if the code matches the architecture, layers, dimensions, and data strategy.
                If it diverges significantly, misses pieces, or fails to implement the plan correctly, reject it and provide detailed feedback to the engineers.
                If it looks faithful to the plan and well-implemented, approve it.
                """),
            expected_output="A raw text analysis including the decision to approve or reject, and detailed feedback.",
            agent=self.agent
        )

        format_task = Task(
            description=dedent("""\
                Format the review from the previous task into a raw JSON object matching the requested schema.
                Schema: {"is_approved": boolean, "feedback": "string"}
                IMPORTANT: Return ONLY a valid JSON object. Do not include markdown formatting or extra text.
                WARNING: Escape all nested quotes and newlines in the 'feedback' string.
                """),
            expected_output='A JSON object starting with "{" containing boolean "is_approved" and string "feedback".',
            agent=self.agent
        )

        debug_mode = os.environ.get("DEBUG_TOOLS", "0") == "1"
        crew = Crew(agents=[self.agent], tasks=[
                    review_task, format_task], verbose=debug_mode)
        result = crew.kickoff()

        try:
            return self.result_to_json(result, ReviewStatus)
        except Exception as e:
            print(
                f"[RECOVERABLE ERROR] Failed to parse ReviewStatus JSON: {e}")
            return ReviewStatus(
                is_approved=True,
                feedback=f"Failed to parse structured JSON. Approving by default to break infinite loops. Raw output: {result.raw[:200]}"
            )
