import re
import os
from textwrap import dedent
from crewai import Agent, Task, Crew
from fellowai.llm import get_llm
from fellowai.models.paper import ArchitecturalPlan, PaperSummary, FinalReport
from typing import Optional


class ResearchArchitect:
    """
    ResearchArchitect drafts step-by-step implementation plans for engineers.
    """

    def __init__(self, llm=None):
        self.llm = llm or get_llm(temperature=0.0)
        debug_mode = os.environ.get("DEBUG_TOOLS", "0") == "1"
        self.agent = Agent(
            role='AI Research Architect',
            goal='Draft a highly detailed, step-by-step implementation plan for an engineering team, covering data strategy and exact model architecture details.',
            backstory='You are a Principal AI Architect who bridges the gap between research and engineering. You read research summaries and feasibility reports, and synthesise exactly how to build it: datasets to use, full model layers, activation functions, sizes, etc.',
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
                """),
            expected_output="A JSON object containing string 'data_strategy' and string 'model_implementation_plan'.",
            agent=self.agent
        )

        debug_mode = os.environ.get("DEBUG_TOOLS", "0") == "1"
        crew = Crew(agents=[self.agent], tasks=[
                    draft_task, format_task], verbose=debug_mode)
        result = crew.kickoff()

        json_str = result.raw
        match = re.search(r'(\{.*\})', result.raw, re.DOTALL)
        if match:
            json_str = match.group(1)

        try:
            return ArchitecturalPlan.model_validate_json(json_str)
        except Exception as e:
            print(
                f"[RECOVERABLE ERROR] Failed to parse ResearchArchitect JSON: {e}")
            print(f"Raw output (may be truncated): {result.raw[:500]}...")
            return ArchitecturalPlan(
                data_strategy="Error parsing data strategy. Raw output: " +
                result.raw[:200],
                model_implementation_plan="Failed to parse structured JSON from ResearchArchitect. See logs for details."
            )
