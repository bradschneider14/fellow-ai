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
            role='Machine Learning Architect Analyst',
            goal='Understand complex ML papers to extract actionable implementation details (layer types, sizes) and performance metrics (accuracy, etc) for an engineering team.',
            backstory='You are a brilliant ML researcher and engineering lead. You specialize in reading academic papers and pulling out the exact architectural specifications and baseline metrics needed to code and train the models.',
            llm=self.llm,
            verbose=debug_mode
        )

    def summarize_paper(self, title: str, pdf_path: str) -> PaperSummary:
        pdf_tool = get_pdf_tool(pdf_path)
        
        search_task = Task(
            description=dedent(f"""\
                You MUST use your PDFSearchTool to read the paper titled '{title}'.
                Search for the abstract or introduction to generate a brief summary.
                Then, search for the methodology or model architecture sections to extract specific implementation details needed to code the model (e.g., layer types, exact dimensions, loss functions).
                Finally, search for the results section to extract the core performance metrics (e.g., accuracy, F1 score, specific benchmark results).
                DO NOT guess. You must formulate a search_query to search the document.
                """),
            expected_output="A raw text summary including abstract, list of architecture details, and list of performance metrics.",
            agent=self.agent,
            tools=[pdf_tool]
        )
        
        format_task = Task(
            description=dedent("""\
                Format the findings from the previous task into a raw JSON object matching the expected schema.
                IMPORTANT: Return ONLY a valid JSON object. Do not include any markdown formatting, backticks, or text outside the JSON block.
                """),
            expected_output="A JSON object containing 'title' (string), 'abstract_summary' (string), 'architecture_details' (list of strings), and 'performance_metrics' (list of strings).",
            agent=self.agent
        )
        debug_mode = os.environ.get("DEBUG_TOOLS", "0") == "1"
        crew = Crew(agents=[self.agent], tasks=[search_task, format_task], verbose=debug_mode)
        result = crew.kickoff()
        
        # Robustly extract JSON from the output
        json_str = result.raw
        match = re.search(r'(\{.*\})', result.raw, re.DOTALL)
        if match:
            json_str = match.group(1)
            
        try:
            summary = PaperSummary.model_validate_json(json_str)
        except Exception as e:
            print(f"[RECOVERABLE ERROR] Failed to parse ResearchAnalyst JSON: {e}")
            # Low-fidelity fallback
            summary = PaperSummary(
                title=title,
                abstract_summary="Failed to parse summary. Raw snippets: " + result.raw[:200],
                architecture_details=[],
                performance_metrics=[],
                citations=[]
            )
        
        if not summary.title or summary.title == "Unknown":
            summary.title = title
        return summary
