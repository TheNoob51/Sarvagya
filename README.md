# Intelligent Research Assistant

A multi-agent AI system that automates comprehensive research tasks using specialized agents for web search, scraping, fact-checking, and report generation.

## Project Structure

```
intelligent-research-assistant/
├── agents/                 # AI agent implementations
│   └── __init__.py
├── utils/                  # Utility modules
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   └── logging_config.py  # Logging setup
├── ui/                     # Gradio user interface
│   └── __init__.py
├── data/                   # Data storage (ChromaDB, cache)
│   └── .gitkeep
├── config.py              # Main configuration loader
├── requirements.txt       # Python dependencies
├── .env.template         # Environment configuration template
├── verify_setup.py       # Project setup verification
└── README.md             # This file
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.template .env
   # Edit .env with your API keys
   ```

3. **Verify setup:**
   ```bash
   python verify_setup.py
   ```

## Configuration

The system uses environment variables for configuration. Copy `.env.template` to `.env` and configure:

### Required
- `GEMINI_API_KEY`: Google Gemini API key for AI processing

### Optional
- `SERPAPI_KEY`: SerpAPI key for enhanced web search (fallback to DuckDuckGo)
- `GOOGLE_SHEETS_CREDENTIALS_PATH`: Path to Google Service Account JSON for research history

### Customizable
- Database paths, timeouts, rate limits, UI settings, and more

## Architecture

The system follows a multi-agent architecture with:

- **Router Agent**: Query analysis and research strategy planning
- **Web Search Agent**: Internet search using SerpAPI/DuckDuckGo
- **Scraper Agent**: Web content extraction
- **Vector Search Agent**: ChromaDB knowledge base search
- **Fact Checker Agent**: Information validation and contradiction detection
- **Summarizer Agent**: Professional report generation
- **Main Orchestrator**: Workflow coordination and progress tracking

## Development

Run the verification script to ensure proper setup:
```bash
python verify_setup.py
```

This will check directory structure, required files, and configuration validity.