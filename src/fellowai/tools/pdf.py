import warnings
warnings.filterwarnings("ignore")

import os
from pydantic import BaseModel, Field
from typing import Type
from crewai.tools import BaseTool

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class SearchPDFInput(BaseModel):
    search_query: str = Field(description="The semantic concept, theme, or string to search for in the PDF.")

class SemanticSearchPDFTool(BaseTool):
    name: str = "PDFSearchTool"
    description: str = "Searches a loaded PDF document for semantically similar concepts and passages."
    args_schema: Type[BaseModel] = SearchPDFInput
    
    pdf_path: str = Field(description="Path to the PDF file")
    
    def __init__(self, pdf_path: str, **kwargs):
        super().__init__(pdf_path=pdf_path, **kwargs)
        # Avoid passing the Chroma instance through Pydantic fields
        # Wait, if we instantiate it here, it works, but we should attach it dynamically 
        # so Pydantic doesn't cry about unserializable fields.
        if not hasattr(self, "_vectorstore"):
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
            
            # Using object's __dict__ to bypass pydantic validation for the Chroma instance
            self.__dict__["_vectorstore"] = Chroma.from_documents(documents=splits, embedding=embeddings)

    def _run(self, search_query: str) -> str:
        """Search the PDF vectorstore for the query."""
        query = search_query
        debug = os.environ.get("DEBUG_TOOLS", "0") == "1"
        if debug:
            print(f"\\n[TOOL LOG] Agent called PDFSearchTool with query: '{query}'")
        docs = self.__dict__.get("_vectorstore").similarity_search(query, k=5)
        if not docs:
            if debug:
                print("[TOOL LOG] No matches found.")
            return f"No semantic matches found for '{query}'."
            
        results = []
        for i, doc in enumerate(docs):
            page_num = doc.metadata.get('page', 'Unknown')
            # Extract content but be mindful of length
            content = doc.page_content.replace("\\n", " ")
            results.append(f"--- Hit {i+1} (Page {page_num}) ---\\n{content}")
            
        result_text = "\\n\\n".join(results)
        if debug:
            print(f"[TOOL LOG] Returning {len(docs)} documents back to the agent.")
        return result_text

def get_pdf_tool(pdf_path: str) -> SemanticSearchPDFTool:
    return SemanticSearchPDFTool(pdf_path=pdf_path)
