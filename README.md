# FellowAI

An agentic application that reads research papers describing ML models.

## Agents

- **LabDirector**: Initiates the project, collects paper metadata, and makes recommendations.
- **ResearchAnalyst**: Summarizes the paper and its key findings.
- **Librarian**: Extracts pertinent citations for key findings.

## Layout

This project uses the modern Python `src` layout.

## Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install the application
pip install -e "."

# Configure API Keys
# Copy the example environment file
cp .env.example .env

# Edit .env and enter your relevant API keys:
# XAI_API_KEY="your-grok-key"
# OPENAI_API_KEY="your-openai-key"

# Run the app (quiet by default)
python -m fellowai.main

# Run the app with visible AI thought boundaries and tool logging
DEBUG_TOOLS=1 python -m fellowai.main

# Run tests
pytest
```

## Build Verification

To verify the build:
```bash
python -m build
```
