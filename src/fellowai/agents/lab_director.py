import re
import os
from textwrap import dedent
from crewai import Agent, Task, Crew
from fellowai.llm import get_llm
from fellowai.tools.pdf import get_pdf_tool
from fellowai.models.paper import PaperMetadata, FinalReport
from fellowai.agents.base import BaseAgent

class LabDirector(BaseAgent):
    """
    LabDirector initiates projects, extracts metadata, and makes final recommendations.
    """
    def __init__(self, llm=None):
        self.llm = llm or get_llm(temperature=0.0)
        debug_mode = os.environ.get("DEBUG_TOOLS", "0") == "1"
        self.agent = Agent(
            role='AI Lab Director & Engineering Manager',
            goal='Evaluate ML research papers for prototyping feasibility, highlighting clear architectural details, performance metrics, and diagnosing potential implementation pitfalls.',
            backstory='You are a senior AI Lab Director managing a team of ML engineers. You are succinct, analytical, and highly critical of vague academic claims. You ensure your team only prototypes papers equipped with sufficient implementation details and solid baselines.',
            llm=self.llm,
            verbose=debug_mode
        )

    def extract_metadata(self, pdf_path: str) -> PaperMetadata:
        pdf_tool = get_pdf_tool(pdf_path)
        
        search_task = Task(
            description=dedent("""\
                You MUST use your PDFSearchTool to find and extract the basic metadata from the document.
                You are looking for the Title of the paper and the list of Authors.
                DO NOT guess. You must formulate a search_query to search the document.
                """),
            expected_output="A raw text summary of the title and authors found in the document.",
            agent=self.agent,
            tools=[pdf_tool]
        )
        
        format_task = Task(
            description=dedent("""\
                Format the findings from the previous task into a raw JSON object matching the expected schema.
                IMPORTANT: Return ONLY a valid JSON object. Do not include any markdown formatting, backticks, or text outside the JSON block.
                """),
            expected_output="A JSON object containing 'title' (string) and 'authors' (list of strings).",
            agent=self.agent
        )

        debug_mode = os.environ.get("DEBUG_TOOLS", "0") == "1"
        crew = Crew(agents=[self.agent], tasks=[search_task, format_task], verbose=debug_mode)
        result = crew.kickoff()
        
        return self.result_to_json(result, PaperMetadata)

    def make_recommendation(self, summary_text: str) -> FinalReport:
        analyze_task = Task(
            description=dedent(f"""\
                Review the following paper summary and extracted details to evaluate if this paper is ready for prototyping by an engineering team.
                
                Summary and Details:
                {summary_text}
                
                You must extract and formulate:
                1. Feasibility: A 2-3 sentence recommendation explaining if the team should prototype this, based on the completeness of the implementation details.
                2. Performance Metrics: A clean list of the core accuracy/performance metrics extracted.
                3. Implementation Details: A clean list of the core layer types, dimensions, and architectural notes extracted.
                4. Pitfalls and Ambiguities: A list of 1-3 potential risks, vague claims, or missing details that the engineering team will need to figure out on their own.
                """),
            expected_output="A raw text analysis including Feasibility, Metrics, Details, and Pitfalls.",
            agent=self.agent
        )
        
        format_task = Task(
            description=dedent("""\
                Format the analysis from the previous task into a raw JSON object matching the requested schema.
                IMPORTANT: Return ONLY a valid JSON object. Do not include markdown formatting or extra text.
                """),
            expected_output="A JSON object containing string 'feasibility', list of strings 'performance_metrics', list of strings 'implementation_details', and list of strings 'pitfalls_and_ambiguities'.",
            agent=self.agent
        )
        
        debug_mode = os.environ.get("DEBUG_TOOLS", "0") == "1"
        crew = Crew(agents=[self.agent], tasks=[analyze_task, format_task], verbose=debug_mode)
        result = crew.kickoff()
        
        # Robustly extract JSON from the output
        try:
            return self.result_to_json(result, FinalReport)
        except Exception as e:
            print(f"[RECOVERABLE ERROR] Failed to parse LabDirector JSON: {e}")
            print(f"Raw output (may be truncated): {result.raw[:500]}...")
            # If JSON is broken, try a very simple fallback to avoid crashing the whole graph
            return FinalReport(
                feasibility="Error parsing feasibility report. Raw output: " + result.raw[:200],
                performance_metrics=[],
                implementation_details=[],
                pitfalls_and_ambiguities=["Failed to parse structured JSON from LabDirector. See logs for details."]
            )
