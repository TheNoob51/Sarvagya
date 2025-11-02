# Architecture Documentation

## Overview

The Intelligent Research Assistant implements a multi-agent architecture where specialized AI agents collaborate through a central orchestrator to perform comprehensive research tasks. This document provides detailed architectural information about the system design, component interactions, and data flow.

## Table of Contents

- [System Architecture](#system-architecture)
- [Component Design](#component-design)
- [Data Flow](#data-flow)
- [Agent Coordination](#agent-coordination)
- [Database Architecture](#database-architecture)
- [API Integration](#api-integration)
- [Performance Considerations](#performance-considerations)
- [Security Architecture](#security-architecture)

## System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface Layer                     │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Gradio Web Interface (ui/)                   │  │
│  │  - Query Input                                            │  │
│  │  - Configuration Controls                                 │  │
│  │  - Progress Display                                       │  │
│  │  - Report Viewer                                          │  │
│  │  - History & Analytics                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestration Layer                           │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           Main Orchestrator (agents/)                     │  │
│  │  - Workflow Coordination                                  │  │
│  │  - Progress Tracking                                      │  │
│  │  - Error Handling                                         │  │
│  │  - Result Aggregation                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Layer                               │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  Router  │  │   Web    │  │ Scraper  │  │  Vector  │       │
│  │  Agent   │  │  Search  │  │  Agent   │  │  Search  │       │
│  │          │  │  Agent   │  │          │  │  Agent   │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                                   │
│  ┌──────────┐  ┌──────────┐                                     │
│  │   Fact   │  │Summarizer│                                     │
│  │ Checker  │  │  Agent   │                                     │
│  │  Agent   │  │          │                                     │
│  └──────────┘  └──────────┘                                     │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Integration Layer                           │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  Gemini  │  │ SerpAPI/ │  │   Web    │  │ ChromaDB │       │
│  │   API    │  │ DuckDuck │  │  Pages   │  │  Vector  │       │
│  │  Client  │  │   Go     │  │          │  │   Store  │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                                   │
│  ┌──────────┐                                                    │
│  │  Google  │                                                    │
│  │  Sheets  │                                                    │
│  │   API    │                                                    │
│  └──────────┘                                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

#### User Interface Layer
- Accepts user input and configuration
- Displays real-time progress updates
- Renders research reports
- Manages research history and analytics
- Handles user interactions

#### Orchestration Layer
- Coordinates agent execution sequence
- Manages parallel processing
- Tracks overall progress
- Handles timeouts and errors
- Aggregates results from agents

#### Agent Layer
- Specialized processing units
- Independent operation
- Well-defined interfaces
- Stateless design for scalability

#### Integration Layer
- External API communication
- Database operations
- Error handling and retries
- Rate limiting

## Component Design

### Main Orchestrator

```
┌─────────────────────────────────────────────────────────┐
│              MainOrchestrator                            │
├─────────────────────────────────────────────────────────┤
│ Responsibilities:                                        │
│ - Initialize all agents                                  │
│ - Execute research workflow                              │
│ - Coordinate parallel operations                         │
│ - Track progress (0-100%)                                │
│ - Handle timeouts and errors                             │
│ - Aggregate results                                      │
├─────────────────────────────────────────────────────────┤
│ Key Methods:                                             │
│ + research(query, config) -> ResearchResult              │
│ + get_progress() -> ProgressStatus                       │
│ + cancel_research() -> bool                              │
├─────────────────────────────────────────────────────────┤
│ Dependencies:                                            │
│ - RouterAgent                                            │
│ - WebSearchAgent                                         │
│ - ScraperAgent                                           │
│ - VectorSearchAgent                                      │
│ - FactCheckerAgent                                       │
│ - SummarizerAgent                                        │
│ - GoogleSheetsHandler (optional)                         │
└─────────────────────────────────────────────────────────┘
```

### Agent Architecture

Each agent follows a consistent design pattern:

```
┌─────────────────────────────────────────────────────────┐
│                    Agent Interface                       │
├─────────────────────────────────────────────────────────┤
│ Common Characteristics:                                  │
│ - Stateless operation                                    │
│ - Single responsibility                                  │
│ - Error handling with retries                            │
│ - Timeout management                                     │
│ - Logging and monitoring                                 │
├─────────────────────────────────────────────────────────┤
│ Standard Methods:                                        │
│ + __init__(dependencies)                                 │
│ + execute(input) -> output                               │
│ + validate_input(input) -> bool                          │
│ + handle_error(error) -> recovery_action                 │
└─────────────────────────────────────────────────────────┘
```

### Router Agent Design

```
┌─────────────────────────────────────────────────────────┐
│                   RouterAgent                            │
├─────────────────────────────────────────────────────────┤
│ Purpose: Query analysis and strategy planning            │
├─────────────────────────────────────────────────────────┤
│ Input:                                                   │
│ - Research query (string)                                │
├─────────────────────────────────────────────────────────┤
│ Processing:                                              │
│ 1. Analyze query intent and complexity                   │
│ 2. Determine optimal data sources                        │
│ 3. Generate search queries (3-5)                         │
│ 4. Suggest target websites (3-5)                         │
│ 5. Estimate processing time                              │
├─────────────────────────────────────────────────────────┤
│ Output:                                                  │
│ - ResearchPlan with strategy decisions                   │
├─────────────────────────────────────────────────────────┤
│ Dependencies:                                            │
│ - GeminiClient (for AI analysis)                         │
│ - PromptTemplates (for structured prompts)               │
└─────────────────────────────────────────────────────────┘
```

### Data Collection Agents

```
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│   WebSearchAgent     │  │    ScraperAgent      │  │  VectorSearchAgent   │
├──────────────────────┤  ├──────────────────────┤  ├──────────────────────┤
│ Purpose:             │  │ Purpose:             │  │ Purpose:             │
│ Internet search      │  │ Content extraction   │  │ Knowledge base query │
├──────────────────────┤  ├──────────────────────┤  ├──────────────────────┤
│ Input:               │  │ Input:               │  │ Input:               │
│ - Search queries     │  │ - Target URLs        │  │ - Search query       │
│ - Max results        │  │ - Scraping config    │  │ - Top K results      │
├──────────────────────┤  ├──────────────────────┤  ├──────────────────────┤
│ Processing:          │  │ Processing:          │  │ Processing:          │
│ 1. Try SerpAPI       │  │ 1. Check robots.txt  │  │ 1. Generate embed    │
│ 2. Fallback to DDG   │  │ 2. Try BeautifulSoup │  │ 2. Query ChromaDB    │
│ 3. Filter results    │  │ 3. Fallback Selenium │  │ 3. Re-rank results   │
│ 4. Score credibility │  │ 4. Extract content   │  │ 4. Filter by score   │
│ 5. Deduplicate       │  │ 5. Clean text        │  │ 5. Return docs       │
├──────────────────────┤  ├──────────────────────┤  ├──────────────────────┤
│ Output:              │  │ Output:              │  │ Output:              │
│ - SearchResult[]     │  │ - ScrapedContent[]   │  │ - Document[]         │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘
```

### Processing Agents

```
┌──────────────────────┐  ┌──────────────────────┐
│  FactCheckerAgent    │  │   SummarizerAgent    │
├──────────────────────┤  ├──────────────────────┤
│ Purpose:             │  │ Purpose:             │
│ Information          │  │ Report generation    │
│ validation           │  │                      │
├──────────────────────┤  ├──────────────────────┤
│ Input:               │  │ Input:               │
│ - All collected data │  │ - Verified facts     │
│ - Credibility thresh │  │ - Source list        │
├──────────────────────┤  ├──────────────────────┤
│ Processing:          │  │ Processing:          │
│ 1. Cross-reference   │  │ 1. Structure content │
│ 2. Find contradicts  │  │ 2. Write summary     │
│ 3. Score credibility │  │ 3. Extract findings  │
│ 4. Remove duplicates │  │ 4. Analyze deeply    │
│ 5. Filter low-cred   │  │ 5. Add citations     │
│ 6. Extract facts     │  │ 6. Generate recomm   │
├──────────────────────┤  ├──────────────────────┤
│ Output:              │  │ Output:              │
│ - FactCheckResult    │  │ - ResearchReport     │
└──────────────────────┘  └──────────────────────┘
```

## Data Flow

### Complete Research Workflow

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ Step 1: Query Analysis (Router Agent)                   │
│ - Analyze query intent                                   │
│ - Create research strategy                               │
│ - Generate search queries                                │
│ Time: ~3 seconds                                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Step 2: Parallel Data Collection                        │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Web Search   │  │ Web Scraping │  │ Vector Search│ │
│  │ (10-30 sec)  │  │ (15-45 sec)  │  │ (5-10 sec)   │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                  │                  │         │
│         └──────────────────┼──────────────────┘         │
└────────────────────────────┼────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ Step 3: Data Aggregation                                │
│ - Combine all sources                                    │
│ - Merge metadata                                         │
│ - Prepare for validation                                 │
│ Time: ~1 second                                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Step 4: Fact Checking (Fact Checker Agent)              │
│ - Validate information                                   │
│ - Identify contradictions                                │
│ - Score credibility                                      │
│ - Remove duplicates                                      │
│ - Filter low-quality sources                             │
│ Time: ~10-15 seconds                                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Step 5: Report Generation (Summarizer Agent)            │
│ - Structure content                                      │
│ - Write executive summary                                │
│ - Extract key findings                                   │
│ - Create detailed analysis                               │
│ - Format citations                                       │
│ - Generate recommendations                               │
│ Time: ~15-20 seconds                                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Step 6: Save & Return                                    │
│ - Save to Google Sheets (optional)                       │
│ - Add to vector database                                 │
│ - Return to user                                         │
│ Time: ~2-5 seconds                                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
                Final Report
```

### Data Transformation Pipeline

```
Raw Query
    │
    ▼
ResearchPlan (JSON)
    │
    ├─→ SearchResult[] ──┐
    │                    │
    ├─→ ScrapedContent[] ├─→ InformationSource[]
    │                    │
    └─→ Document[] ──────┘
                         │
                         ▼
                  FactCheckResult
                         │
                         ├─→ verified_facts[]
                         ├─→ credibility_scores{}
                         └─→ cleaned_data[]
                                  │
                                  ▼
                          ResearchReport
                                  │
                                  ├─→ executive_summary
                                  ├─→ key_findings[]
                                  ├─→ detailed_analysis
                                  ├─→ sources[]
                                  └─→ recommendations[]
```

## Agent Coordination

### Orchestrator State Machine

```
┌─────────┐
│  IDLE   │
└────┬────┘
     │ research() called
     ▼
┌─────────────┐
│ ANALYZING   │ Router Agent
└────┬────────┘
     │ plan ready
     ▼
┌─────────────┐
│ COLLECTING  │ Web Search, Scraper, Vector Search (parallel)
└────┬────────┘
     │ data collected
     ▼
┌─────────────┐
│ VALIDATING  │ Fact Checker Agent
└────┬────────┘
     │ facts verified
     ▼
┌─────────────┐
│ SUMMARIZING │ Summarizer Agent
└────┬────────┘
     │ report ready
     ▼
┌─────────────┐
│   SAVING    │ Google Sheets, Vector DB
└────┬────────┘
     │ saved
     ▼
┌─────────┐
│COMPLETE │
└─────────┘
```

### Parallel Execution Strategy

```
Time →
0s    ┌──────────────────────────────────────────────────┐
      │ Router Agent (Sequential)                        │
3s    └──────────────────────────────────────────────────┘
      ┌──────────────────────────────────────────────────┐
      │                                                  │
      │  ┌────────────────────────┐                     │
      │  │ Web Search Agent       │                     │
      │  └────────────────────────┘                     │
      │                                                  │
      │  ┌──────────────────────────────────┐           │
      │  │ Scraper Agent                    │           │
      │  └──────────────────────────────────┘           │
      │                                                  │
      │  ┌──────────────┐                               │
      │  │ Vector Search│                               │
      │  └──────────────┘                               │
      │                                                  │
45s   └──────────────────────────────────────────────────┘
      ┌──────────────────────────────────────────────────┐
      │ Fact Checker Agent (Sequential)                  │
60s   └──────────────────────────────────────────────────┘
      ┌──────────────────────────────────────────────────┐
      │ Summarizer Agent (Sequential)                    │
80s   └──────────────────────────────────────────────────┘
      ┌──────────────────────────────────────────────────┐
      │ Save Results (Sequential)                        │
85s   └──────────────────────────────────────────────────┘
```

### Error Handling Flow

```
Agent Execution
      │
      ├─→ Success → Continue
      │
      ├─→ Timeout → Log + Continue with partial data
      │
      ├─→ API Error → Retry (max 3) → Success/Fail
      │                                      │
      │                                      ├─→ Success → Continue
      │                                      │
      │                                      └─→ Fail → Log + Continue
      │
      └─→ Critical Error → Abort + Return error report
```

## Database Architecture

### ChromaDB Structure

```
┌─────────────────────────────────────────────────────────┐
│                    ChromaDB Instance                     │
├─────────────────────────────────────────────────────────┤
│ Collection: research_knowledge                           │
├─────────────────────────────────────────────────────────┤
│ Documents:                                               │
│ - Text content (max 5000 chars)                          │
│ - Embeddings (Gemini-generated vectors)                  │
│                                                          │
│ Metadata:                                                │
│ - source_url: string                                     │
│ - timestamp: ISO datetime                                │
│ - credibility_score: float (0-10)                        │
│ - content_type: string (article/research/news)           │
│ - domain: string                                         │
│ - word_count: int                                        │
│                                                          │
│ IDs:                                                     │
│ - UUID v4 for each document                              │
├─────────────────────────────────────────────────────────┤
│ Operations:                                              │
│ - add(): Add documents with embeddings                   │
│ - query(): Semantic similarity search                    │
│ - delete(): Remove old documents                         │
│ - count(): Get document count                            │
└─────────────────────────────────────────────────────────┘
```

### Google Sheets Schema

```
┌─────────────────────────────────────────────────────────┐
│              Research Assistant Data Sheet               │
├──────────┬──────────┬──────────┬──────────┬─────────────┤
│Timestamp │  Query   │ Summary  │  Report  │   Sources   │
├──────────┼──────────┼──────────┼──────────┼─────────────┤
│ datetime │  string  │  string  │  string  │     int     │
│          │          │ (200ch)  │ (full)   │             │
├──────────┼──────────┼──────────┼──────────┼─────────────┤
│Processing│  Status  │          │          │             │
│   Time   │          │          │          │             │
├──────────┼──────────┼──────────┼──────────┼─────────────┤
│  float   │  string  │          │          │             │
│ (seconds)│(success/ │          │          │             │
│          │  error)  │          │          │             │
└──────────┴──────────┴──────────┴──────────┴─────────────┘
```

## API Integration

### Gemini API Integration

```
┌─────────────────────────────────────────────────────────┐
│                   GeminiClient                           │
├─────────────────────────────────────────────────────────┤
│ Features:                                                │
│ - Automatic retry with exponential backoff               │
│ - Rate limiting (60 req/min default)                     │
│ - Request queuing                                        │
│ - Error classification                                   │
│ - Response caching                                       │
├─────────────────────────────────────────────────────────┤
│ Endpoints Used:                                          │
│ - generateContent (text generation)                      │
│ - embedContent (embedding generation)                    │
├─────────────────────────────────────────────────────────┤
│ Retry Strategy:                                          │
│ Attempt 1: Immediate                                     │
│ Attempt 2: Wait 1 second                                 │
│ Attempt 3: Wait 2 seconds                                │
│ Attempt 4: Wait 4 seconds                                │
│ Then: Fail with APIError                                 │
└─────────────────────────────────────────────────────────┘
```

### Web Search Integration

```
┌─────────────────────────────────────────────────────────┐
│              WebSearchAgent Strategy                     │
├─────────────────────────────────────────────────────────┤
│ Primary: SerpAPI                                         │
│ - Requires API key                                       │
│ - 100 searches/month (free tier)                         │
│ - High-quality results                                   │
│ - Structured JSON response                               │
│                                                          │
│ Fallback: DuckDuckGo                                     │
│ - No API key required                                    │
│ - Unlimited searches                                     │
│ - Good quality results                                   │
│ - HTML parsing required                                  │
├─────────────────────────────────────────────────────────┤
│ Decision Logic:                                          │
│ if serpapi_key exists:                                   │
│     try SerpAPI                                          │
│     if fails: use DuckDuckGo                             │
│ else:                                                    │
│     use DuckDuckGo                                       │
└─────────────────────────────────────────────────────────┘
```

## Performance Considerations

### Optimization Strategies

1. **Parallel Execution**
   - Web search, scraping, and vector search run concurrently
   - Reduces total time by ~40%

2. **Caching**
   - Search results: 24 hours
   - Vector search: 1 hour
   - Embeddings: Permanent (in ChromaDB)

3. **Rate Limiting**
   - Gemini API: 60 requests/minute
   - Web searches: 30 searches/minute
   - Web scraping: 5 concurrent requests

4. **Resource Management**
   - Connection pooling for HTTP requests
   - Lazy loading of heavy dependencies
   - Memory-efficient streaming for large content

### Performance Metrics

```
Typical Research Timeline:
- Router analysis: 3 seconds
- Data collection: 30-45 seconds (parallel)
- Fact checking: 10-15 seconds
- Report generation: 15-20 seconds
- Saving: 2-5 seconds
Total: 60-90 seconds average
```

## Security Architecture

### API Key Management

```
┌─────────────────────────────────────────────────────────┐
│                  Security Layers                         │
├─────────────────────────────────────────────────────────┤
│ 1. Environment Variables (.env file)                     │
│    - Never committed to version control                  │
│    - Loaded at runtime only                              │
│    - Validated on startup                                │
│                                                          │
│ 2. Service Account Credentials                           │
│    - Stored in credentials/ directory                    │
│    - Excluded from git (.gitignore)                      │
│    - Minimal permissions (Sheets editor only)            │
│                                                          │
│ 3. Runtime Protection                                    │
│    - Keys never logged                                   │
│    - Keys never sent to client                           │
│    - Keys never included in error messages               │
└─────────────────────────────────────────────────────────┘
```

### Data Privacy

- No user data stored permanently (except optional Google Sheets)
- Vector database can be cleared anytime
- No telemetry or external tracking
- All processing happens locally or through user's API keys

### Web Scraping Ethics

- Respects robots.txt by default
- Rate limiting to avoid overwhelming servers
- User-Agent identification
- Timeout handling to prevent hanging

---

For implementation details, see the source code in the respective agent files. For usage examples, see the API documentation and README.
