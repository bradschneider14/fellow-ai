from textwrap import dedent
from crewai import Agent, Task, Crew
from fellowai.llm import get_llm
from fellowai.models.paper import PaperMetadata

class LabDirector:
    """
    LabDirector initiates projects, extracts metadata, and makes final recommendations.
    """
    def __init__(self, llm=None):
        self.llm = llm or get_llm(temperature=0.0)
        self.agent = Agent(
            role='Lab Director',
            goal='Extract accurate metadata from research papers and make data-driven recommendations.',
            backstory='You are a senior AI Lab Director who oversees a team of researchers. You are succinct, analytical, and focused on identifying high-value papers to prototype.',
            llm=self.llm,
            verbose=True
        )

    def extract_metadata(self, raw_text: str) -> PaperMetadata:
        task = Task(
            description=dedent(f"""\
                Analyze the following research paper excerpt and extract the basic metadata.
                Make your best guess for the title and authors. If not found, use "Unknown".
                
                Paper Excerpt:
                {raw_text[:4000]}
                """),
            expected_output="A populated PaperMetadata object with title and authors.",
            agent=self.agent,
            output_pydantic=PaperMetadata
        )
        crew = Crew(agents=[self.agent], tasks=[task], verbose=False)
        result = crew.kickoff()
        return result.pydantic

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
        crew = Crew(agents=[self.agent], tasks=[task], verbose=False)
        result = crew.kickoff()
        return result.raw
