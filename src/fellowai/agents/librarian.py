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

class Librarian:
    """
    Librarian extracts the most pertinent citations for the key findings.
    """
    def __init__(self, llm=None):
        self.llm = llm or get_llm(temperature=0.0)
        debug_mode = os.environ.get("DEBUG_TOOLS", "0") == "1"
        self.agent = Agent(
            role='Academic Librarian',
            goal='Identify and extract key citations and references from research papers.',
            backstory='You are a meticulous librarian who excels at tracking down sources, references, and understanding the context of citations within academic texts.',
            llm=self.llm,
            verbose=debug_mode
        )

    def extract_citations(self, pdf_path: str, findings: List[str]) -> List[Citation]:
        findings_str = "\\n- ".join(findings)
        pdf_tool = get_pdf_tool(pdf_path)
        
        search_task = Task(
            description=dedent(f"""\
                You MUST use your PDFSearchTool to search the document for 2-4 key citations that seem most relevant to these findings:
                - {findings_str}
                DO NOT guess. You must formulate a search_query to search the document.
                
                For each citation, provide the text of the citation exactly as it appears in the bibliography/references section, its source/authors, and why it is relevant to the finding.
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
        
        match = re.search(r'\{.*\}', result.raw, re.DOTALL)
        json_str = match.group(0) if match else result.raw
        final_list = CitationList.model_validate_json(json_str)
        return final_list.citations if final_list else []
