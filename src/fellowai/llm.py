import os
from crewai import LLM

def get_llm(temperature: float = 0.0) -> LLM:
    """
    Returns a configured CrewAI LLM instance based on environment variables.
    Handles empty strings gracefully and works around CrewAI's strict OPENAI validation.
    """
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    xai_key = os.environ.get("XAI_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()

    # Prioritize xAI if explicitly requested or if XAI key exists and provider is blank
    if provider == "xai" or (not provider and xai_key):
        if not xai_key:
            raise ValueError("XAI_API_KEY is not set or is empty.")
            
        return LLM(
            model="xai/grok-4-1-fast-reasoning",
            api_key=xai_key,
            temperature=temperature,
            max_tokens=4096
        )
    
    # Check for Gemini (Google) if provided explicitly
    elif provider == "gemini" or os.environ.get("GOOGLE_API_KEY"):
        google_key = os.environ.get("GOOGLE_API_KEY")
        if not google_key:
             raise ValueError("GOOGLE_API_KEY is not set.")
             
        return LLM(
            model="gemini/gemini-2.0-flash",
            api_key=google_key,
            temperature=temperature,
            max_tokens=4096
        )
    
    # Fallback to OpenAI
    elif provider == "openai" or openai_key:
        if not openai_key:
            raise ValueError("OPENAI_API_KEY is not set or is empty.")
            
        return LLM(
            model="gpt-4o-mini",
            api_key=openai_key,
            temperature=temperature,
            max_tokens=4096
        )
    
    raise ValueError("No valid API key found. Please define XAI_API_KEY or OPENAI_API_KEY in your .env.")
