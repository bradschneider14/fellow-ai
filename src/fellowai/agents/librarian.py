import re
import os
from typing import List
from textwrap import dedent
from crewai import Agent, Task, Crew
from fellowai.llm import get_llm
from fellowai.tools.pdf import get_pdf_tool
from fellowai.models.paper import Citation
from pydantic import BaseModel

class CitationList(BaseModel):
    citations: List[Citation]

from fellowai.agents.base import BaseAgent

class Librarian(BaseAgent):
    """
    Librarian extracts the most pertinent citations for the key findings.
    """
    def __init__(self, llm=None):
        self.llm = llm or get_llm(temperature=0.0)
        debug_mode = os.environ.get("DEBUG_TOOLS", "0") == "1"
        self.agent = Agent(
            role='ML Implementation Research Librarian',
            goal='Identify and extract key citations that likely contain missing ML implementation details, prior architectures, or baseline codebases.',
            backstory='You are a meticulous technical librarian. You know that papers often omit layer details by citing previous work. Your job is to find those specific citations so the engineering team knows where to look for the missing puzzle pieces.',
            llm=self.llm,
            verbose=debug_mode
        )

    def extract_citations(self, pdf_path: str, extraction_context: List[str]) -> List[Citation]:
        context_str = "\\n- ".join(extraction_context)
        pdf_tool = get_pdf_tool(pdf_path)
        
        search_task = Task(
            description=dedent(f"""\
                You MUST use your PDFSearchTool to search the document for 2-4 key citations that the authors rely on for their model architecture, baselines, or datasets:
                Here are the implementation details and metrics extracted so far to give you context:
                - {context_str}
                DO NOT guess. You must formulate a search_query to search the document.
                
                For each citation, provide the text of the citation exactly as it appears in the bibliography/references section, its source/authors, and why it is relevant to the engineering team (e.g., 'contains the baseline ResNet implementation they modified').
                """),
            expected_output="A raw text compilation of 2-4 key citations and their contexts.",
            agent=self.agent,
            tools=[pdf_tool]
        )
        
        format_task = Task(
            description=dedent("""\
                Format the extracted citations from the previous task into a raw JSON object matching the expected schema.
                IMPORTANT: Return ONLY a valid JSON object. The object MUST have a 'citations' key containing the list. Do not include any markdown formatting, backticks, or text outside the JSON block.
                """),
            expected_output="A JSON object containing a 'citations' list. Each citation object in the list MUST have 'text', 'source', and 'relevance' string keys.",
            agent=self.agent
        )
        debug_mode = os.environ.get("DEBUG_TOOLS", "0") == "1"
        crew = Crew(agents=[self.agent], tasks=[search_task, format_task], verbose=debug_mode)
        result = crew.kickoff()
        
        # Robustly extract JSON from the output
        try:
            final_list = self.result_to_json(result, CitationList)
            return final_list.citations if final_list else []
        except Exception as e:
            print(f"[RECOVERABLE ERROR] Failed to parse Librarian JSON: {e}")
            return []
