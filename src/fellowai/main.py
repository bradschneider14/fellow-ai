import os
from dotenv import load_dotenv

# Load environment variables from .env file FIRST; Must happen before importing app!
# This is because app imports llm, which imports os.environ
load_dotenv()

from fellowai.workflow.graph import app

def main():

    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("XAI_API_KEY"):
        print("Please set the OPENAI_API_KEY or XAI_API_KEY environment variable to run the agents.")
        print("Example: export XAI_API_KEY='xai-...'")
        return

    print("--- FellowAI: LangGraph + CrewAI Workflow ---")
    
    sample_text = """
    Attention Is All You Need
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
    
    Abstract
    The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.
    
    1 Introduction
    Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation.
    ...
    [References]
    [1] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. CoRR, abs/1409.0473, 2014.
    [2] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780, 1997.
    """
    
    initial_state = {
        "raw_text": sample_text,
        "project": None,
        "error": None
    }
    
    print("Invoking graph...")
    final_state = app.invoke(initial_state)
    
    project = final_state.get("project")
    if project:
        print("\\n=== FINAL RESULT ===")
        print(f"Title: {project.metadata.title}")
        print(f"Authors: {project.metadata.authors}")
        if project.summary:
            print(f"Summary: {project.summary.abstract_summary}")
            print(f"Key Findings: {project.summary.key_findings}")
            print(f"Citations Extracted: {len(project.summary.citations)}")
            for c in project.summary.citations:
                print(f"  - [{c.source}]: {c.text} ({c.relevance})")
        print(f"\\nRecommendation: {project.recommendation}")

if __name__ == "__main__":
    main()
