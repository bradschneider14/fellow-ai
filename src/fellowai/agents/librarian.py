from typing import List
from textwrap import dedent
from crewai import Agent, Task, Crew
from fellowai.llm import get_llm
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
        self.agent = Agent(
            role='Academic Librarian',
            goal='Identify and extract key citations and references from research papers.',
            backstory='You are a meticulous librarian who excels at tracking down sources, references, and understanding the context of citations within academic texts.',
            llm=self.llm,
            verbose=True
        )

    def extract_citations(self, raw_text: str, findings: List[str]) -> List[Citation]:
        findings_str = "\\n- ".join(findings)
        task = Task(
            description=dedent(f"""\
                Review the following paper text and extract 2-4 key citations that seem most relevant to these findings:
                - {findings_str}
                
                For each citation, provide the text of the citation as it appears, its source/authors, and why it is relevant.
                
                Paper Text:
                {raw_text[-5000:]} # Focus on the end where references usually are, and some body text
                """),
            expected_output="A list of Citation objects.",
            agent=self.agent,
            output_pydantic=CitationList
        )
        crew = Crew(agents=[self.agent], tasks=[task], verbose=False)
        result = crew.kickoff()
        return result.pydantic.citations if result.pydantic else []
