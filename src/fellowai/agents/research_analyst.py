from textwrap import dedent
from crewai import Agent, Task, Crew
from fellowai.llm import get_llm
from fellowai.models.paper import PaperSummary

class ResearchAnalyst:
    """
    ResearchAnalyst reads the paper and summarizes key findings.
    """
    def __init__(self, llm=None):
        self.llm = llm or get_llm(temperature=0.1)
        self.agent = Agent(
            role='Senior Research Analyst',
            goal='Understand complex ML papers, summarize their abstracts, and identify the most critical key findings.',
            backstory='You are a brilliant ML researcher with a knack for distilling complex academic jargon into clear, actionable insights.',
            llm=self.llm,
            verbose=True
        )

    def summarize_paper(self, title: str, raw_text: str) -> PaperSummary:
        task = Task(
            description=dedent(f"""\
                Read the following text from the paper titled '{title}'.
                Generate a concise summary of the abstract (or introduction if abstract is missing), 
                and list 3-5 core key findings.
                
                Paper Text:
                {raw_text[:10000]}
                """),
            expected_output="A PaperSummary object containing the title, abstract_summary, and a list of key_findings.",
            agent=self.agent,
            output_pydantic=PaperSummary
        )
        crew = Crew(agents=[self.agent], tasks=[task], verbose=False)
        result = crew.kickoff()
        summary = result.pydantic
        if not summary.title or summary.title == "Unknown":
            summary.title = title
        return summary
