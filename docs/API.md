# API Documentation

## Overview

This document provides detailed API documentation for the Intelligent Research Assistant system. It covers all public interfaces, data models, and usage patterns for programmatic access.

## Table of Contents

- [Core Components](#core-components)
- [Data Models](#data-models)
- [Agent APIs](#agent-apis)
- [Utility APIs](#utility-apis)
- [Error Handling](#error-handling)
- [Usage Examples](#usage-examples)

## Core Components

### MainOrchestrator

The central coordination system that manages the research workflow.

#### Class Definition

```python
class MainOrchestrator:
    """
    Main orchestrator for coordinating research workflow across all agents.
    
    Attributes:
        router_agent: RouterAgent instance for query analysis
        web_search_agent: WebSearchAgent for internet searches
        scraper_agent: ScraperAgent for web content extraction
        vector_search_agent: VectorSearchAgent for knowledge base queries
        fact_checker_agent: FactCheckerAgent for information validation
        summarizer_agent: SummarizerAgent for report generation
        sheets_handler: GoogleSheetsHandler for research history (optional)
    """
```

#### Methods

##### research()

Execute a complete research workflow.

```python
def research(
    self,
    query: str,
    config: ResearchConfig,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> ResearchResult:
    """
    Perform comprehensive research on the given query.
    
    Args:
        query: Research question or topic to investigate
        config: Configuration parameters for the research
        progress_callback: Optional callback for progress updates
                          Signature: callback(progress: int, status: str)
    
    Returns:
        ResearchResult containing the generated report and metadata
    
    Raises:
        TimeoutError: If research exceeds configured timeout
        APIError: If critical API calls fail after retries
        ValueError: If query is empty or invalid
    
    Example:
        >>> orchestrator = MainOrchestrator()
        >>> config = ResearchConfig(max_sources=10)
        >>> result = orchestrator.research("What is AI?", config)
        >>> print(result.report.executive_summary)
    """
```

##### get_progress()

Get current research progress.

```python
def get_progress(self) -> ProgressStatus:
    """
    Get the current progress of ongoing research.
    
    Returns:
        ProgressStatus with percentage (0-100) and current status message
    
    Example:
        >>> status = orchestrator.get_progress()
        >>> print(f"Progress: {status.percentage}% - {status.message}")
    """
```

##### cancel_research()

Cancel an ongoing research operation.

```python
def cancel_research(self) -> bool:
    """
    Cancel the currently running research operation.
    
    Returns:
        True if cancellation was successful, False otherwise
    
    Example:
        >>> orchestrator.cancel_research()
        True
    """
```

## Data Models

### ResearchConfig

Configuration parameters for research execution.

```python
@dataclass
class ResearchConfig:
    """Configuration for research execution."""
    
    max_sources: int = 10
    """Maximum number of sources to gather (1-50)"""
    
    enable_web_scraping: bool = True
    """Whether to enable web content extraction"""
    
    enable_vector_search: bool = True
    """Whether to search the vector database"""
    
    report_style: str = "academic"
    """Report writing style: 'academic', 'casual', or 'technical'"""
    
    report_length: str = "medium"
    """Report length: 'short' (500-700 words), 'medium' (800-1000), 'long' (1200-1500)"""
    
    timeout_seconds: int = 120
    """Maximum time for research execution in seconds"""
    
    def validate(self) -> List[str]:
        """
        Validate configuration parameters.
        
        Returns:
            List of validation error messages (empty if valid)
        """
```

### ResearchResult

Result of a research operation.

```python
@dataclass
class ResearchResult:
    """Result of research execution."""
    
    report: ResearchReport
    """Generated research report"""
    
    metadata: dict
    """Execution metadata including timing and source counts"""
    
    sources_used: List[str]
    """List of source URLs used in the research"""
    
    processing_time: float
    """Total processing time in seconds"""
    
    success: bool
    """Whether the research completed successfully"""
    
    error_message: Optional[str] = None
    """Error message if research failed"""
```

### ResearchReport

Structured research report.

```python
@dataclass
class ResearchReport:
    """Structured research report."""
    
    executive_summary: str
    """3-4 sentence overview of findings"""
    
    key_findings: List[str]
    """5-7 bullet points of main discoveries"""
    
    detailed_analysis: str
    """3-5 paragraphs of in-depth analysis"""
    
    sources: List[str]
    """Numbered list of source citations"""
    
    recommendations: List[str]
    """Actionable recommendations based on findings"""
    
    metadata: dict
    """Report metadata (word count, generation time, etc.)"""
    
    def to_markdown(self) -> str:
        """Convert report to formatted markdown string."""
    
    def to_dict(self) -> dict:
        """Convert report to dictionary."""
```

### ResearchPlan

Research strategy created by RouterAgent.

```python
@dataclass
class ResearchPlan:
    """Research strategy plan."""
    
    use_web_search: bool
    """Whether to use web search"""
    
    use_web_scraping: bool
    """Whether to scrape websites"""
    
    use_vector_search: bool
    """Whether to search vector database"""
    
    search_queries: List[str]
    """Optimized search queries (3-5)"""
    
    target_websites: List[str]
    """Suggested websites for scraping (3-5)"""
    
    estimated_time: int
    """Estimated completion time in seconds"""
    
    reasoning: str
    """Explanation of strategy decisions"""
```

### SearchResult

Web search result.

```python
@dataclass
class SearchResult:
    """Web search result."""
    
    title: str
    """Page title"""
    
    url: str
    """Page URL"""
    
    snippet: str
    """Brief content excerpt"""
    
    credibility_score: float
    """Credibility rating (0.0-10.0)"""
    
    source: str
    """Search provider: 'serpapi' or 'duckduckgo'"""
    
    timestamp: datetime
    """When result was retrieved"""
```

### ScrapedContent

Extracted web page content.

```python
@dataclass
class ScrapedContent:
    """Scraped web page content."""
    
    url: str
    """Source URL"""
    
    title: str
    """Page title"""
    
    content: str
    """Extracted main content"""
    
    author: Optional[str]
    """Article author if available"""
    
    publish_date: Optional[datetime]
    """Publication date if available"""
    
    extraction_method: str
    """Method used: 'beautifulsoup' or 'selenium'"""
    
    word_count: int
    """Number of words in content"""
```

### Document

Vector database document.

```python
@dataclass
class Document:
    """Vector database document."""
    
    content: str
    """Document text content"""
    
    metadata: dict
    """Document metadata (source, timestamp, etc.)"""
    
    similarity_score: float
    """Similarity to query (0.0-1.0)"""
    
    credibility_score: float
    """Source credibility (0.0-10.0)"""
```

### FactCheckResult

Fact checking results.

```python
@dataclass
class FactCheckResult:
    """Fact checking results."""
    
    verified_facts: List[str]
    """List of verified factual statements"""
    
    credibility_scores: Dict[str, float]
    """Credibility scores by source URL"""
    
    contradictions: List[str]
    """Identified contradictions between sources"""
    
    cleaned_data: List[str]
    """Deduplicated and cleaned information"""
    
    removed_sources: List[str]
    """Sources removed due to low credibility"""
```

## Agent APIs

### RouterAgent

Analyzes queries and creates research strategies.

```python
class RouterAgent:
    """Query analysis and research strategy planning."""
    
    def __init__(self, gemini_client: GeminiClient):
        """Initialize with Gemini API client."""
    
    def analyze_query(self, query: str) -> ResearchPlan:
        """
        Analyze research query and create optimal strategy.
        
        Args:
            query: Research question or topic
        
        Returns:
            ResearchPlan with strategy decisions
        
        Raises:
            APIError: If Gemini API call fails
            ValueError: If query is empty or invalid
        
        Example:
            >>> agent = RouterAgent(gemini_client)
            >>> plan = agent.analyze_query("What is quantum computing?")
            >>> print(plan.search_queries)
            ['quantum computing basics', 'quantum computing applications', ...]
        """
```

### WebSearchAgent

Performs web searches using multiple providers.

```python
class WebSearchAgent:
    """Web search using SerpAPI or DuckDuckGo."""
    
    def __init__(self, serpapi_key: Optional[str] = None):
        """
        Initialize with optional SerpAPI key.
        Falls back to DuckDuckGo if key not provided.
        """
    
    def search(
        self,
        queries: List[str],
        max_results_per_query: int = 10
    ) -> List[SearchResult]:
        """
        Search the web for given queries.
        
        Args:
            queries: List of search queries
            max_results_per_query: Maximum results per query
        
        Returns:
            List of SearchResult objects
        
        Raises:
            RateLimitError: If rate limit exceeded
            APIError: If both SerpAPI and DuckDuckGo fail
        
        Example:
            >>> agent = WebSearchAgent(serpapi_key="your_key")
            >>> results = agent.search(["AI research", "machine learning"])
            >>> for result in results:
            ...     print(f"{result.title}: {result.url}")
        """
    
    def calculate_credibility_score(self, url: str) -> float:
        """
        Calculate credibility score for a URL.
        
        Args:
            url: URL to evaluate
        
        Returns:
            Credibility score (0.0-10.0)
        """
```

### ScraperAgent

Extracts content from web pages.

```python
class ScraperAgent:
    """Web content extraction using BeautifulSoup and Selenium."""
    
    def __init__(
        self,
        max_concurrent: int = 5,
        timeout: int = 30,
        respect_robots: bool = True
    ):
        """
        Initialize scraper with configuration.
        
        Args:
            max_concurrent: Maximum concurrent scrapes
            timeout: Timeout per scrape in seconds
            respect_robots: Whether to respect robots.txt
        """
    
    def scrape_urls(self, urls: List[str]) -> List[ScrapedContent]:
        """
        Scrape content from multiple URLs.
        
        Args:
            urls: List of URLs to scrape
        
        Returns:
            List of ScrapedContent objects
        
        Raises:
            TimeoutError: If scraping exceeds timeout
        
        Example:
            >>> agent = ScraperAgent()
            >>> content = agent.scrape_urls(["https://example.com"])
            >>> print(content[0].title)
        """
    
    def scrape_static_page(self, url: str) -> ScrapedContent:
        """Scrape static HTML page using BeautifulSoup."""
    
    def scrape_dynamic_page(self, url: str) -> ScrapedContent:
        """Scrape JavaScript-heavy page using Selenium."""
```

### VectorSearchAgent

Searches and manages the vector database.

```python
class VectorSearchAgent:
    """Vector database search using ChromaDB."""
    
    def __init__(
        self,
        chroma_manager: ChromaManager,
        gemini_client: GeminiClient
    ):
        """Initialize with ChromaDB manager and Gemini client."""
    
    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.6
    ) -> List[Document]:
        """
        Search for semantically similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
        
        Returns:
            List of Document objects ranked by relevance
        
        Example:
            >>> agent = VectorSearchAgent(chroma_manager, gemini_client)
            >>> docs = agent.search_similar("machine learning", top_k=5)
            >>> for doc in docs:
            ...     print(f"Score: {doc.similarity_score}")
        """
    
    def add_documents(
        self,
        texts: List[str],
        metadata: List[dict]
    ) -> bool:
        """
        Add documents to the vector database.
        
        Args:
            texts: List of document texts
            metadata: List of metadata dicts (one per text)
        
        Returns:
            True if successful
        
        Example:
            >>> texts = ["Document 1 content", "Document 2 content"]
            >>> metadata = [
            ...     {"source": "url1", "timestamp": "2024-01-01"},
            ...     {"source": "url2", "timestamp": "2024-01-02"}
            ... ]
            >>> agent.add_documents(texts, metadata)
            True
        """
```

### FactCheckerAgent

Validates information and removes contradictions.

```python
class FactCheckerAgent:
    """Information validation and contradiction detection."""
    
    def __init__(self, gemini_client: GeminiClient):
        """Initialize with Gemini API client."""
    
    def check_facts(
        self,
        data: List[InformationSource],
        credibility_threshold: float = 6.0
    ) -> FactCheckResult:
        """
        Validate information and identify contradictions.
        
        Args:
            data: List of information sources to validate
            credibility_threshold: Minimum credibility score to keep
        
        Returns:
            FactCheckResult with verified facts
        
        Example:
            >>> agent = FactCheckerAgent(gemini_client)
            >>> sources = [
            ...     InformationSource(content="Fact 1", url="url1"),
            ...     InformationSource(content="Fact 2", url="url2")
            ... ]
            >>> result = agent.check_facts(sources)
            >>> print(result.verified_facts)
        """
```

### SummarizerAgent

Generates professional research reports.

```python
class SummarizerAgent:
    """Research report generation."""
    
    def __init__(self, gemini_client: GeminiClient):
        """Initialize with Gemini API client."""
    
    def generate_report(
        self,
        facts: List[str],
        sources: List[str],
        config: ReportConfig
    ) -> ResearchReport:
        """
        Generate structured research report.
        
        Args:
            facts: List of verified facts
            sources: List of source URLs
            config: Report configuration
        
        Returns:
            ResearchReport with all sections
        
        Example:
            >>> agent = SummarizerAgent(gemini_client)
            >>> config = ReportConfig(style="academic", length="medium")
            >>> report = agent.generate_report(facts, sources, config)
            >>> print(report.executive_summary)
        """
```

## Utility APIs

### GeminiClient

Wrapper for Google Gemini API with retry logic.

```python
class GeminiClient:
    """Gemini API client with retry and rate limiting."""
    
    def __init__(
        self,
        api_key: str,
        rate_limit: int = 60,
        max_retries: int = 3
    ):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key
            rate_limit: Requests per minute
            max_retries: Maximum retry attempts
        """
    
    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        Generate text using Gemini.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum response length
        
        Returns:
            Generated text
        
        Raises:
            APIError: If API call fails after retries
            RateLimitError: If rate limit exceeded
        """
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.
        
        Args:
            text: Input text
        
        Returns:
            Embedding vector
        """
```

### ChromaManager

ChromaDB database management.

```python
class ChromaManager:
    """ChromaDB vector database manager."""
    
    def __init__(
        self,
        db_path: str,
        collection_name: str = "research_knowledge"
    ):
        """
        Initialize ChromaDB manager.
        
        Args:
            db_path: Path to database directory
            collection_name: Name of collection
        """
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadata: List[dict],
        ids: List[str]
    ) -> None:
        """Add documents to collection."""
    
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5
    ) -> dict:
        """Query collection for similar documents."""
    
    def delete_old_documents(self, days: int) -> int:
        """Delete documents older than specified days."""
```

### GoogleSheetsHandler

Google Sheets integration for research history.

```python
class GoogleSheetsHandler:
    """Google Sheets integration for research history."""
    
    def __init__(
        self,
        credentials_path: str,
        sheet_name: str = "Research Assistant Data"
    ):
        """
        Initialize Google Sheets handler.
        
        Args:
            credentials_path: Path to service account JSON
            sheet_name: Name of spreadsheet
        """
    
    def save_research(
        self,
        query: str,
        summary: str,
        full_report: str,
        sources: List[str],
        processing_time: float
    ) -> bool:
        """
        Save research to Google Sheets.
        
        Args:
            query: Research query
            summary: Executive summary
            full_report: Complete report text
            sources: List of source URLs
            processing_time: Time taken in seconds
        
        Returns:
            True if successful
        """
    
    def get_recent_research(self, limit: int = 20) -> List[dict]:
        """Get recent research entries."""
    
    def get_analytics(self) -> dict:
        """Get research analytics."""
```

## Error Handling

### Exception Hierarchy

```python
class ResearchAssistantError(Exception):
    """Base exception for all system errors."""

class APIError(ResearchAssistantError):
    """API call failed after retries."""

class RateLimitError(ResearchAssistantError):
    """Rate limit exceeded."""

class TimeoutError(ResearchAssistantError):
    """Operation exceeded timeout."""

class ValidationError(ResearchAssistantError):
    """Input validation failed."""

class ConfigurationError(ResearchAssistantError):
    """Configuration is invalid."""
```

### Error Handling Patterns

```python
from utils.error_handler import handle_api_error, retry_with_backoff

# Automatic retry with exponential backoff
@retry_with_backoff(max_retries=3, base_delay=1.0)
def call_api():
    # API call that might fail
    pass

# Graceful error handling
try:
    result = orchestrator.research(query, config)
except TimeoutError:
    print("Research timed out, try reducing max_sources")
except APIError as e:
    print(f"API error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Usage Examples

### Basic Research

```python
from agents.main_orchestrator import MainOrchestrator
from agents.data_models import ResearchConfig

# Initialize
orchestrator = MainOrchestrator()

# Simple research
result = orchestrator.research(
    query="What is artificial intelligence?",
    config=ResearchConfig()
)

print(result.report.executive_summary)
```

### Custom Configuration

```python
# Advanced configuration
config = ResearchConfig(
    max_sources=15,
    enable_web_scraping=True,
    enable_vector_search=True,
    report_style="technical",
    report_length="long",
    timeout_seconds=180
)

result = orchestrator.research(
    query="Latest developments in quantum computing",
    config=config
)
```

### Progress Tracking

```python
def progress_callback(percentage: int, status: str):
    print(f"Progress: {percentage}% - {status}")

result = orchestrator.research(
    query="Climate change impacts",
    config=ResearchConfig(),
    progress_callback=progress_callback
)
```

### Adding to Knowledge Base

```python
from agents.vector_search_agent import VectorSearchAgent
from utils.chroma_manager import ChromaManager
from utils.gemini_client import GeminiClient

# Initialize components
chroma = ChromaManager("./data/chroma_db")
gemini = GeminiClient(api_key="your_key")
vector_agent = VectorSearchAgent(chroma, gemini)

# Add documents
texts = [
    "Machine learning is a subset of AI...",
    "Deep learning uses neural networks..."
]
metadata = [
    {"source": "https://example.com/ml", "timestamp": "2024-01-01"},
    {"source": "https://example.com/dl", "timestamp": "2024-01-02"}
]

vector_agent.add_documents(texts, metadata)
```

### Searching Knowledge Base

```python
# Search for relevant documents
docs = vector_agent.search_similar(
    query="What is machine learning?",
    top_k=5,
    similarity_threshold=0.7
)

for doc in docs:
    print(f"Similarity: {doc.similarity_score:.2f}")
    print(f"Content: {doc.content[:100]}...")
    print(f"Source: {doc.metadata['source']}")
```

### Saving to Google Sheets

```python
from utils.google_sheets_handler import GoogleSheetsHandler

# Initialize handler
sheets = GoogleSheetsHandler(
    credentials_path="./credentials/service-account.json",
    sheet_name="My Research Data"
)

# Save research
sheets.save_research(
    query="AI trends 2024",
    summary=result.report.executive_summary,
    full_report=result.report.to_markdown(),
    sources=result.sources_used,
    processing_time=result.processing_time
)

# Get history
recent = sheets.get_recent_research(limit=10)
for entry in recent:
    print(f"{entry['timestamp']}: {entry['query']}")
```

### Custom Agent Usage

```python
from agents.router_agent import RouterAgent
from agents.web_search_agent import WebSearchAgent

# Use individual agents
router = RouterAgent(gemini_client)
plan = router.analyze_query("What is blockchain?")

print(f"Use web search: {plan.use_web_search}")
print(f"Search queries: {plan.search_queries}")

# Execute web search
search_agent = WebSearchAgent(serpapi_key="your_key")
results = search_agent.search(plan.search_queries)

for result in results:
    print(f"{result.title}: {result.credibility_score}/10")
```

---

For more examples and detailed usage, see the main README.md and the test suite in the `tests/` directory.
