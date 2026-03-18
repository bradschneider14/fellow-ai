import re
import os
from textwrap import dedent
from crewai import Agent, Task, Crew
from fellowai.llm import get_llm
from fellowai.tools.pdf import get_pdf_tool
from fellowai.models.paper import PaperSummary

class ResearchAnalyst:
    """
    ResearchAnalyst reads the paper and summarizes key findings.
    """
    def __init__(self, llm=None):
        self.llm = llm or get_llm(temperature=0.1)
        debug_mode = os.environ.get("DEBUG_TOOLS", "0") == "1"
        self.agent = Agent(
            role='Senior Research Analyst',
            goal='Understand complex ML papers, summarize their abstracts, and identify the most critical key findings.',
            backstory='You are a brilliant ML researcher with a knack for distilling complex academic jargon into clear, actionable insights.',
            llm=self.llm,
            verbose=debug_mode
        )

    def summarize_paper(self, title: str, pdf_path: str) -> PaperSummary:
        pdf_tool = get_pdf_tool(pdf_path)
        
        search_task = Task(
            description=dedent(f"""\
                You MUST use your PDFSearchTool to read the paper titled '{title}'.
                Search for the abstract or introduction to generate a concise summary.
                Then, search for the conclusion or main results to list 3-5 core key findings.
                DO NOT guess. You must formulate a search_query to search the document.
                """),
            expected_output="A raw text summary of the paper, including abstract and 3-5 core findings.",
            agent=self.agent,
            tools=[pdf_tool]
        )
        
        format_task = Task(
            description=dedent("""\
                Format the findings from the previous task into a raw JSON object matching the expected schema.
                IMPORTANT: Return ONLY a valid JSON object. Do not include any markdown formatting, backticks, or text outside the JSON block.
                """),
            expected_output="A JSON object containing 'title' (string), 'abstract_summary' (string), and 'key_findings' (list of strings).",
            agent=self.agent
        )
        debug_mode = os.environ.get("DEBUG_TOOLS", "0") == "1"
        crew = Crew(agents=[self.agent], tasks=[search_task, format_task], verbose=debug_mode)
        result = crew.kickoff()
        
        match = re.search(r'\{.*\}', result.raw, re.DOTALL)
        json_str = match.group(0) if match else result.raw
        summary = PaperSummary.model_validate_json(json_str)
        
        if not summary.title or summary.title == "Unknown":
            summary.title = title
        return summary
