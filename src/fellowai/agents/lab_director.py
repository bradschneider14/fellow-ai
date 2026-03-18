import re
import os
from textwrap import dedent
from crewai import Agent, Task, Crew
from fellowai.llm import get_llm
from fellowai.tools.pdf import get_pdf_tool
from fellowai.models.paper import PaperMetadata

class LabDirector:
    """
    LabDirector initiates projects, extracts metadata, and makes final recommendations.
    """
    def __init__(self, llm=None):
        self.llm = llm or get_llm(temperature=0.0)
        debug_mode = os.environ.get("DEBUG_TOOLS", "0") == "1"
        self.agent = Agent(
            role='Lab Director',
            goal='Extract accurate metadata from research papers and make data-driven recommendations.',
            backstory='You are a senior AI Lab Director who oversees a team of researchers. You are succinct, analytical, and focused on identifying high-value papers to prototype.',
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
        
        match = re.search(r'\{.*\}', result.raw, re.DOTALL)
        json_str = match.group(0) if match else result.raw
        return PaperMetadata.model_validate_json(json_str)

    def make_recommendation(self, summary_text: str) -> str:
        task = Task(
            description=dedent(f"""\
                Review the following paper summary and decide if it should be handed to the engineering team for prototyping.
                Provide a short 2-3 sentence recommendation explaining your reasoning based on the summary's key findings.
                
                Summary:
                {summary_text}
                """),
            expected_output="A 2-3 sentence recommendation.",
            agent=self.agent
        )
        debug_mode = os.environ.get("DEBUG_TOOLS", "0") == "1"
        crew = Crew(agents=[self.agent], tasks=[task], verbose=debug_mode)
        result = crew.kickoff()
        return result.raw
