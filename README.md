# Sarvagya - Intelligent Research Assistant

A multi-agent AI system that automates comprehensive research tasks using specialized agents for web search, scraping, fact-checking, and report generation. The system coordinates six specialized AI agents through a central orchestrator to gather, process, verify, and summarize information from multiple sources.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)

## Features

- **Multi-Agent Architecture**: Six specialized AI agents working in coordination
- **Intelligent Query Analysis**: Automatic research strategy planning based on query type
- **Multiple Data Sources**: Web search, web scraping, and vector database integration
- **Fact Checking**: Automatic validation and contradiction detection
- **Professional Reports**: Structured research reports with citations
- **Research History**: Save and access previous research via Google Sheets
- **Real-time Progress**: Live updates during research execution
- **Configurable**: Extensive configuration options for customization
- **Fault Tolerant**: Graceful degradation when services are unavailable

## Architecture

### High-Level Overview

The system implements a coordinated multi-agent architecture where specialized agents collaborate to perform comprehensive research:

```
┌─────────────────────────────────────────────────────────────┐
│                      Gradio Web UI                          │
│  (Query Input, Progress Display, Report Viewing, History)   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Main Orchestrator                         │
│  (Workflow Coordination, Progress Tracking, Error Handling) │
└─┬───────┬───────┬───────┬───────┬───────┬──────────────────┘
  │       │       │       │       │       │
  ▼       ▼       ▼       ▼       ▼       ▼
┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐
│ R │   │WS │   │SC │   │VS │   │FC │   │SU │
│ o │   │e  │   │r  │   │e  │   │a  │   │m  │
│ u │   │b  │   │a  │   │c  │   │c  │   │m  │
│ t │   │   │   │p  │   │t  │   │t  │   │a  │
│ e │   │S  │   │e  │   │o  │   │   │   │r  │
│ r │   │e  │   │r  │   │r  │   │C  │   │i  │
│   │   │a  │   │   │   │   │   │h  │   │z  │
│ A │   │r  │   │A  │   │S  │   │e  │   │e  │
│ g │   │c  │   │g  │   │e  │   │c  │   │r  │
│ e │   │h  │   │e  │   │a  │   │k  │   │   │
│ n │   │   │   │n  │   │r  │   │e  │   │A  │
│ t │   │A  │   │t  │   │c  │   │r  │   │g  │
│   │   │g  │   │   │   │h  │   │   │   │e  │
│   │   │e  │   │   │   │   │   │A  │   │n  │
│   │   │n  │   │   │   │A  │   │g  │   │t  │
│   │   │t  │   │   │   │g  │   │e  │   │   │
│   │   │   │   │   │   │e  │   │n  │   │   │
│   │   │   │   │   │   │n  │   │t  │   │   │
│   │   │   │   │   │   │t  │   │   │   │   │
└─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘
  │       │       │       │       │       │
  ▼       ▼       ▼       ▼       │       │
┌────┐  ┌────┐  ┌────┐  ┌────┐   │       │
│Gem │  │Serp│  │Web │  │Chr │   │       │
│ini │  │API │  │Page│  │oma │   │       │
│API │  │/DDG│  │s   │  │DB  │   │       │
└────┘  └────┘  └────┘  └────┘   │       │
                                  ▼       ▼
                              ┌──────────────┐
                              │  Gemini API  │
                              └──────────────┘
```

### Agent Responsibilities

1. **Router Agent**: Analyzes research queries and creates optimal research strategies
2. **Web Search Agent**: Performs internet searches using SerpAPI or DuckDuckGo
3. **Scraper Agent**: Extracts content from web pages using BeautifulSoup/Selenium
4. **Vector Search Agent**: Searches ChromaDB for relevant stored knowledge
5. **Fact Checker Agent**: Validates information and removes contradictions
6. **Summarizer Agent**: Generates professional research reports with citations

### Workflow Sequence

```
1. User submits query → Router Agent analyzes
2. Router creates research plan (which sources to use)
3. Parallel execution:
   - Web Search Agent searches internet
   - Scraper Agent extracts from targeted websites
   - Vector Search Agent queries knowledge base
4. Fact Checker Agent validates all collected data
5. Summarizer Agent generates final report
6. Results saved to Google Sheets (optional)
7. Report displayed to user
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Google Gemini API key (required)
- SerpAPI key (optional, for enhanced search)
- Google Cloud Service Account (optional, for research history)

### Step-by-Step Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd intelligent-research-assistant
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   # Copy the template
   cp .env.template .env
   
   # Edit .env with your API keys
   # At minimum, add your GEMINI_API_KEY
   ```

5. **Verify installation:**
   ```bash
   python verify_setup.py
   ```

   This will check:
   - Directory structure
   - Required files
   - Configuration validity
   - API connectivity

### Getting API Keys

#### Google Gemini API (Required)
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key to your `.env` file as `GEMINI_API_KEY`

#### SerpAPI (Optional)
1. Visit [SerpAPI](https://serpapi.com/)
2. Sign up for a free account (100 searches/month)
3. Copy your API key to `.env` as `SERPAPI_KEY`
4. If not provided, system will use DuckDuckGo (free, no key required)

#### Google Sheets (Optional)
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google Sheets API
4. Create a Service Account
5. Download the JSON credentials file
6. Set path in `.env` as `GOOGLE_SHEETS_CREDENTIALS_PATH`

## Configuration

### Environment Variables

The system uses a `.env` file for configuration. Copy `.env.template` to `.env` and customize:

#### Required Configuration

```bash
# Gemini API Key (Required for all AI operations)
GEMINI_API_KEY=your_gemini_api_key_here
```

#### Optional API Keys

```bash
# SerpAPI for enhanced web search (fallback to DuckDuckGo if not provided)
SERPAPI_KEY=your_serpapi_key_here

# Google Sheets for research history
GOOGLE_SHEETS_CREDENTIALS_PATH=path/to/credentials.json
GOOGLE_SHEETS_NAME=Research Assistant Data
GOOGLE_SHEETS_WORKSHEET=Research Data
```

#### Database Configuration

```bash
# ChromaDB vector database settings
CHROMA_DB_PATH=./data/chroma_db
CHROMA_COLLECTION_NAME=research_knowledge
```

#### Web Scraping Settings

```bash
# Concurrent scraping limits
MAX_CONCURRENT_SCRAPES=5
SCRAPE_TIMEOUT=30
RESPECT_ROBOTS_TXT=true
USER_AGENT=Intelligent-Research-Assistant/1.0
```

#### Application Settings

```bash
# Debug and logging
DEBUG=false
LOG_LEVEL=INFO

# Gradio UI configuration
GRADIO_HOST=127.0.0.1
GRADIO_PORT=7860
```

#### Research Defaults

```bash
# Default research parameters (can be overridden in UI)
DEFAULT_MAX_SOURCES=10
DEFAULT_TIMEOUT_SECONDS=120
DEFAULT_REPORT_STYLE=academic  # academic, casual, technical
DEFAULT_REPORT_LENGTH=medium   # short, medium, long
```

#### Rate Limiting

```bash
# API rate limits
GEMINI_RATE_LIMIT=60           # requests per minute
WEB_SEARCH_RATE_LIMIT=30       # searches per minute

# Cache durations
SEARCH_CACHE_DURATION=24       # hours
VECTOR_CACHE_DURATION=1        # hours
```

### Configuration Examples

#### Minimal Configuration (Gemini only)
```bash
GEMINI_API_KEY=your_key_here
```

#### Full Configuration (All features)
```bash
GEMINI_API_KEY=your_gemini_key
SERPAPI_KEY=your_serpapi_key
GOOGLE_SHEETS_CREDENTIALS_PATH=./credentials/service-account.json
GOOGLE_SHEETS_NAME=My Research Data
DEBUG=true
LOG_LEVEL=DEBUG
```

#### Production Configuration
```bash
GEMINI_API_KEY=your_key_here
SERPAPI_KEY=your_key_here
GOOGLE_SHEETS_CREDENTIALS_PATH=/secure/path/credentials.json
DEBUG=false
LOG_LEVEL=WARNING
GRADIO_HOST=0.0.0.0
GRADIO_PORT=8080
```

## Usage

### Starting the Application

1. **Activate your virtual environment:**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

2. **Launch the UI:**
   ```bash
   python launch_ui.py
   ```

3. **Access the interface:**
   - Open your browser to `http://127.0.0.1:7860`
   - Or the URL displayed in the terminal

### Using the Research Interface

#### Basic Research

1. **Enter your research query** in the text box
   - Example: "What are the latest developments in quantum computing?"

2. **Configure options** (optional):
   - **Max Sources**: Number of sources to gather (default: 10)
   - **Enable Web Scraping**: Extract full content from websites
   - **Enable Vector Search**: Search existing knowledge base
   - **Report Style**: academic, casual, or technical
   - **Report Length**: short, medium, or long

3. **Click "Start Research"**

4. **Monitor progress** in real-time:
   - Progress bar shows completion percentage
   - Status updates show current agent activity

5. **Review the report**:
   - Executive Summary
   - Key Findings
   - Detailed Analysis
   - Sources & Citations
   - Recommendations

#### Advanced Features

**Research History Tab:**
- View your last 20 research queries
- Click on any query to view the full report
- See metadata (sources used, processing time)

**Analytics Tab:**
- Total researches completed
- Average sources per research
- Average processing time
- Success rate statistics

**Settings Tab:**
- Update API keys without restarting
- Change default parameters
- Configure model settings

### Command-Line Usage

You can also use the system programmatically:

```python
from agents.main_orchestrator import MainOrchestrator
from agents.data_models import ResearchConfig

# Initialize orchestrator
orchestrator = MainOrchestrator()

# Configure research
config = ResearchConfig(
    max_sources=10,
    enable_web_scraping=True,
    enable_vector_search=True,
    report_style="academic",
    report_length="medium",
    timeout_seconds=120
)

# Perform research
result = orchestrator.research(
    query="What are the benefits of renewable energy?",
    config=config
)

# Access results
print(result.report.executive_summary)
print(result.report.key_findings)
print(result.report.detailed_analysis)
```

## API Documentation

### Core Classes

#### MainOrchestrator

Central coordination system for the research workflow.

```python
class MainOrchestrator:
    def research(self, query: str, config: ResearchConfig) -> ResearchResult:
        """
        Execute complete research workflow.
        
        Args:
            query: Research question or topic
            config: Research configuration parameters
            
        Returns:
            ResearchResult containing report and metadata
            
        Raises:
            TimeoutError: If research exceeds timeout
            APIError: If critical API calls fail
        """
        
    def get_progress(self) -> ProgressStatus:
        """Get current research progress (0-100)"""
        
    def cancel_research(self) -> bool:
        """Cancel ongoing research"""
```

#### ResearchConfig

Configuration for research execution.

```python
@dataclass
class ResearchConfig:
    max_sources: int = 10              # Maximum sources to gather
    enable_web_scraping: bool = True   # Enable web content extraction
    enable_vector_search: bool = True  # Enable knowledge base search
    report_style: str = "academic"     # Report writing style
    report_length: str = "medium"      # Report length
    timeout_seconds: int = 120         # Maximum execution time
```

#### ResearchResult

Result of research execution.

```python
@dataclass
class ResearchResult:
    report: ResearchReport             # Generated report
    metadata: dict                     # Execution metadata
    sources_used: List[str]            # URLs of sources
    processing_time: float             # Total time in seconds
    success: bool                      # Whether research completed
```

### Agent APIs

#### RouterAgent

```python
class RouterAgent:
    def analyze_query(self, query: str) -> ResearchPlan:
        """Analyze query and create research strategy"""
```

#### WebSearchAgent

```python
class WebSearchAgent:
    def search(self, queries: List[str]) -> List[SearchResult]:
        """Search web using SerpAPI or DuckDuckGo"""
```

#### ScraperAgent

```python
class ScraperAgent:
    def scrape_urls(self, urls: List[str]) -> List[ScrapedContent]:
        """Extract content from web pages"""
```

#### VectorSearchAgent

```python
class VectorSearchAgent:
    def search_similar(self, query: str, top_k: int = 5) -> List[Document]:
        """Search vector database for relevant documents"""
        
    def add_documents(self, texts: List[str], metadata: List[dict]) -> bool:
        """Add documents to knowledge base"""
```

#### FactCheckerAgent

```python
class FactCheckerAgent:
    def check_facts(self, data: List[InformationSource]) -> FactCheckResult:
        """Validate information and remove contradictions"""
```

#### SummarizerAgent

```python
class SummarizerAgent:
    def generate_report(self, facts: List[str], config: ReportConfig) -> ResearchReport:
        """Generate professional research report"""
```

## Project Structure

```
intelligent-research-assistant/
├── agents/                      # AI agent implementations
│   ├── __init__.py
│   ├── data_models.py          # Data structures and models
│   ├── router_agent.py         # Query analysis agent
│   ├── web_search_agent.py     # Web search agent
│   ├── web_scraper_agent.py    # Web scraping agent
│   ├── vector_search_agent.py  # Vector database agent
│   ├── fact_checker_agent.py   # Fact checking agent
│   ├── summarizer_agent.py     # Report generation agent
│   └── main_orchestrator.py    # Workflow coordinator
│
├── utils/                       # Utility modules
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── logging_config.py       # Logging setup
│   ├── gemini_client.py        # Gemini API wrapper
│   ├── chroma_manager.py       # ChromaDB management
│   ├── google_sheets_handler.py # Google Sheets integration
│   ├── error_handler.py        # Error handling utilities
│   └── prompt_templates.py     # AI prompt templates
│
├── ui/                          # Gradio user interface
│   ├── __init__.py
│   ├── app.py                  # Main application entry
│   └── gradio_interface.py     # UI components
│
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── conftest.py             # Test configuration
│   ├── test_orchestrator_integration.py
│   ├── test_processing_agents.py
│   ├── test_web_agents_integration.py
│   ├── test_vector_database.py
│   ├── test_google_sheets_integration.py
│   ├── test_ui_components.py
│   └── test_end_to_end_system.py
│
├── data/                        # Data storage
│   ├── .gitkeep
│   └── chroma_db/              # Vector database (created at runtime)
│
├── credentials/                 # API credentials (not in git)
│   └── .gitkeep
│
├── .kiro/                       # Kiro specs and documentation
│   └── specs/
│       └── intelligent-research-assistant/
│           ├── requirements.md
│           ├── design.md
│           └── tasks.md
│
├── config.py                    # Main configuration loader
├── launch_ui.py              # Application launcher
├── verify_setup.py             # Setup verification script
├── requirements.txt            # Python dependencies
├── .env.template              # Environment configuration template
├── .env                       # Your configuration (not in git)
├── .gitignore         # Git ignore rules
└── README.md                  # This file
```

## Troubleshooting

### Common Issues

#### 1. "GEMINI_API_KEY not found" Error

**Problem**: The Gemini API key is not configured.

**Solution**:
```bash
# 1. Check if .env file exists
ls .env

# 2. If not, copy template
cp .env.template .env

# 3. Edit .env and add your key
# GEMINI_API_KEY=your_actual_key_here

# 4. Verify configuration
python verify_setup.py
```

#### 2. "Module not found" Errors

**Problem**: Dependencies not installed or virtual environment not activated.

**Solution**:
```bash
# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### 3. ChromaDB Initialization Fails

**Problem**: Database directory doesn't exist or has permission issues.

**Solution**:
```bash
# Create data directory
mkdir -p data/chroma_db

# Check permissions (Linux/macOS)
chmod 755 data/chroma_db

# Or change database path in .env
CHROMA_DB_PATH=./custom/path/chroma_db
```

#### 4. Web Scraping Timeouts

**Problem**: Websites taking too long to load or blocking requests.

**Solution**:
```bash
# Increase timeout in .env
SCRAPE_TIMEOUT=60

# Reduce concurrent scrapes
MAX_CONCURRENT_SCRAPES=3

# Check if robots.txt is blocking
RESPECT_ROBOTS_TXT=false  # Use cautiously
```

#### 5. Google Sheets Connection Fails

**Problem**: Service account credentials invalid or API not enabled.

**Solution**:
1. Verify credentials file exists at specified path
2. Check Google Sheets API is enabled in Cloud Console
3. Ensure service account has editor access to the sheet
4. Test credentials:
   ```python
   from utils.google_sheets_handler import GoogleSheetsHandler
   handler = GoogleSheetsHandler()
   handler.test_connection()
   ```

#### 6. Rate Limit Errors

**Problem**: Exceeding API rate limits.

**Solution**:
```bash
# Reduce rate limits in .env
GEMINI_RATE_LIMIT=30
WEB_SEARCH_RATE_LIMIT=15

# Increase cache duration
SEARCH_CACHE_DURATION=48
```

#### 7. Memory Issues with Large Research

**Problem**: System running out of memory with many sources.

**Solution**:
```bash
# Reduce max sources in UI or .env
DEFAULT_MAX_SOURCES=5

# Disable web scraping for lighter operation
# (Use in UI settings)
```

#### 8. Gradio UI Won't Start

**Problem**: Port already in use or binding issues.

**Solution**:
```bash
# Change port in .env
GRADIO_PORT=7861

# Or bind to all interfaces
GRADIO_HOST=0.0.0.0

# Check if port is in use (Windows)
netstat -ano | findstr :7860

# Check if port is in use (Linux/macOS)
lsof -i :7860
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
# In .env
DEBUG=true
LOG_LEVEL=DEBUG
```

Then check logs for detailed error information.

### Getting Help

If you encounter issues not covered here:

1. Check the logs in the console output
2. Run `python verify_setup.py` to diagnose configuration issues
3. Review the `.env.template` for correct configuration format
4. Check that all API keys are valid and have sufficient quota
5. Ensure Python version is 3.8 or higher: `python --version`

## Development

### Setting Up Development Environment

1. **Clone and install as above**

2. **Install development dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Enable debug mode:**
   ```bash
   # In .env
   DEBUG=true
   LOG_LEVEL=DEBUG
   ```

### Code Style

The project uses:
- **Black** for code formatting
- **Flake8** for linting

```bash
# Format code
black .

# Lint code
flake8 .
```

### Project Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all public methods
- Include type hints where applicable
- Write tests for new features
- Update documentation for API changes

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_orchestrator_integration.py

# Run with coverage
pytest --cov=agents --cov=utils

# Run with verbose output
pytest -v

# Run specific test
pytest tests/test_orchestrator_integration.py::test_research_workflow
```

### Test Structure

- **Unit Tests**: Test individual agent functionality
- **Integration Tests**: Test agent coordination and API integration
- **End-to-End Tests**: Test complete research workflows

### Writing Tests

```python
import pytest
from agents.router_agent import RouterAgent

def test_router_agent_analysis():
    """Test router agent query analysis"""
    agent = RouterAgent()
    plan = agent.analyze_query("What is machine learning?")
    
    assert plan.use_web_search is True
    assert len(plan.search_queries) >= 3
    assert plan.estimated_time > 0
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Development Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
pytest

# Format and lint
black .
flake8 .

# Commit changes
git add .
git commit -m "Add your feature"

# Push and create PR
git push origin feature/your-feature-name
```

---

## License

[Add your license information here]

## Acknowledgments

- Google Gemini API for AI capabilities
- ChromaDB for vector database
- Gradio for the user interface
- All open-source dependencies

## Support

For questions, issues, or feature requests, please open an issue on the repository.