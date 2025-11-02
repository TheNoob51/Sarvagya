# Configuration Guide

## Overview

This guide provides detailed information about configuring the Intelligent Research Assistant for different use cases and environments.

## Table of Contents

- [Configuration File](#configuration-file)
- [Configuration Examples](#configuration-examples)
- [Environment Variables Reference](#environment-variables-reference)
- [Advanced Configuration](#advanced-configuration)
- [Performance Tuning](#performance-tuning)
- [Security Best Practices](#security-best-practices)

## Configuration File

### Location

The configuration file is located at the project root:
```
intelligent-research-assistant/.env
```

### Creating Configuration

1. **Copy the template:**
   ```bash
   cp .env.template .env
   ```

2. **Edit with your values:**
   ```bash
   # Linux/macOS
   nano .env
   
   # Windows
   notepad .env
   ```

3. **Verify configuration:**
   ```bash
   python verify_setup.py
   ```

### Configuration Structure

```bash
# Comments start with #
# Format: KEY=value
# No spaces around =
# No quotes needed (usually)

GEMINI_API_KEY=your_key_here
DEBUG=false
```

## Configuration Examples

### Minimal Configuration

For basic usage with just Gemini API:

```bash
# .env
GEMINI_API_KEY=AIzaSyD...your_key_here
```

This provides:
- AI-powered research
- Web search via DuckDuckGo (free)
- Local vector database
- No research history

### Standard Configuration

For typical usage with all free features:

```bash
# .env

# Required
GEMINI_API_KEY=AIzaSyD...your_key_here

# Optional but recommended
SERPAPI_KEY=your_serpapi_key_here

# Database
CHROMA_DB_PATH=./data/chroma_db
CHROMA_COLLECTION_NAME=research_knowledge

# Application
DEBUG=false
LOG_LEVEL=INFO
GRADIO_HOST=127.0.0.1
GRADIO_PORT=7860

# Research defaults
DEFAULT_MAX_SOURCES=10
DEFAULT_TIMEOUT_SECONDS=120
DEFAULT_REPORT_STYLE=academic
DEFAULT_REPORT_LENGTH=medium
```

### Full Configuration

For complete functionality including research history:

```bash
# .env

# ============= REQUIRED =============
GEMINI_API_KEY=AIzaSyD...your_key_here

# ============= OPTIONAL APIs =============
SERPAPI_KEY=your_serpapi_key_here
GOOGLE_SHEETS_CREDENTIALS_PATH=./credentials/service-account.json
GOOGLE_SHEETS_NAME=Research Assistant Data
GOOGLE_SHEETS_WORKSHEET=Research Data
GOOGLE_SHEETS_CLEANUP_DAYS=90

# ============= DATABASE =============
CHROMA_DB_PATH=./data/chroma_db
CHROMA_COLLECTION_NAME=research_knowledge

# ============= WEB SCRAPING =============
MAX_CONCURRENT_SCRAPES=5
SCRAPE_TIMEOUT=30
RESPECT_ROBOTS_TXT=true
USER_AGENT=Intelligent-Research-Assistant/1.0

# ============= APPLICATION =============
DEBUG=false
LOG_LEVEL=INFO
GRADIO_HOST=127.0.0.1
GRADIO_PORT=7860

# ============= RESEARCH DEFAULTS =============
DEFAULT_MAX_SOURCES=10
DEFAULT_TIMEOUT_SECONDS=120
DEFAULT_REPORT_STYLE=academic
DEFAULT_REPORT_LENGTH=medium

# ============= RATE LIMITING =============
GEMINI_RATE_LIMIT=60
WEB_SEARCH_RATE_LIMIT=30
SEARCH_CACHE_DURATION=24
VECTOR_CACHE_DURATION=1
```

### Development Configuration

For development and debugging:

```bash
# .env

# Required
GEMINI_API_KEY=AIzaSyD...your_key_here

# Development settings
DEBUG=true
LOG_LEVEL=DEBUG

# Faster iteration
DEFAULT_MAX_SOURCES=5
DEFAULT_TIMEOUT_SECONDS=60
DEFAULT_REPORT_LENGTH=short

# Local only
GRADIO_HOST=127.0.0.1
GRADIO_PORT=7860

# Relaxed limits for testing
GEMINI_RATE_LIMIT=30
WEB_SEARCH_RATE_LIMIT=15
MAX_CONCURRENT_SCRAPES=3
```

### Production Configuration

For production deployment:

```bash
# .env

# Required
GEMINI_API_KEY=AIzaSyD...your_key_here
SERPAPI_KEY=your_serpapi_key_here
GOOGLE_SHEETS_CREDENTIALS_PATH=/secure/path/credentials.json

# Production settings
DEBUG=false
LOG_LEVEL=WARNING

# Performance optimized
DEFAULT_MAX_SOURCES=15
DEFAULT_TIMEOUT_SECONDS=180
GEMINI_RATE_LIMIT=60
WEB_SEARCH_RATE_LIMIT=30
MAX_CONCURRENT_SCRAPES=5

# Network
GRADIO_HOST=0.0.0.0
GRADIO_PORT=8080

# Caching
SEARCH_CACHE_DURATION=48
VECTOR_CACHE_DURATION=2
```

### Fast Research Configuration

For quick research with minimal sources:

```bash
# .env

GEMINI_API_KEY=AIzaSyD...your_key_here

# Speed optimized
DEFAULT_MAX_SOURCES=5
DEFAULT_TIMEOUT_SECONDS=60
DEFAULT_REPORT_LENGTH=short

# Disable slow operations
# (Set in UI: uncheck "Enable Web Scraping")

# Aggressive caching
SEARCH_CACHE_DURATION=72
VECTOR_CACHE_DURATION=24
```

### Comprehensive Research Configuration

For thorough, detailed research:

```bash
# .env

GEMINI_API_KEY=AIzaSyD...your_key_here
SERPAPI_KEY=your_serpapi_key_here

# Quality optimized
DEFAULT_MAX_SOURCES=20
DEFAULT_TIMEOUT_SECONDS=300
DEFAULT_REPORT_STYLE=academic
DEFAULT_REPORT_LENGTH=long

# Enable all features
MAX_CONCURRENT_SCRAPES=10
SCRAPE_TIMEOUT=60

# Minimal caching for fresh data
SEARCH_CACHE_DURATION=1
VECTOR_CACHE_DURATION=0
```

### Offline Development Configuration

For testing without external APIs:

```bash
# .env

# Mock API key (for testing only)
GEMINI_API_KEY=test_key_for_development

# Disable external services
# SERPAPI_KEY=  # Commented out
# GOOGLE_SHEETS_CREDENTIALS_PATH=  # Commented out

# Local only
DEBUG=true
LOG_LEVEL=DEBUG
GRADIO_HOST=127.0.0.1
GRADIO_PORT=7860

# Minimal timeouts
DEFAULT_TIMEOUT_SECONDS=30
SCRAPE_TIMEOUT=10
```

## Environment Variables Reference

### Required Variables

#### GEMINI_API_KEY
- **Type:** String
- **Required:** Yes
- **Description:** Google Gemini API key for AI operations
- **Example:** `AIzaSyD...`
- **Get from:** https://makersuite.google.com/app/apikey

### Optional API Keys

#### SERPAPI_KEY
- **Type:** String
- **Required:** No
- **Default:** None (uses DuckDuckGo)
- **Description:** SerpAPI key for enhanced web search
- **Example:** `your_serpapi_key`
- **Get from:** https://serpapi.com/

#### GOOGLE_SHEETS_CREDENTIALS_PATH
- **Type:** File path
- **Required:** No
- **Default:** None (no history saved)
- **Description:** Path to Google Service Account JSON
- **Example:** `./credentials/service-account.json`
- **Get from:** https://console.cloud.google.com/

#### GOOGLE_SHEETS_NAME
- **Type:** String
- **Required:** No
- **Default:** `Research Assistant Data`
- **Description:** Name of Google Sheets spreadsheet
- **Example:** `My Research Data`

#### GOOGLE_SHEETS_WORKSHEET
- **Type:** String
- **Required:** No
- **Default:** `Research Data`
- **Description:** Name of worksheet within spreadsheet
- **Example:** `Main Sheet`

#### GOOGLE_SHEETS_CLEANUP_DAYS
- **Type:** Integer
- **Required:** No
- **Default:** `90`
- **Description:** Days to keep research history
- **Example:** `30`

### Database Configuration

#### CHROMA_DB_PATH
- **Type:** Directory path
- **Required:** No
- **Default:** `./data/chroma_db`
- **Description:** Path to ChromaDB vector database
- **Example:** `./custom/db/path`

#### CHROMA_COLLECTION_NAME
- **Type:** String
- **Required:** No
- **Default:** `research_knowledge`
- **Description:** Name of ChromaDB collection
- **Example:** `my_research`

### Web Scraping Configuration

#### MAX_CONCURRENT_SCRAPES
- **Type:** Integer
- **Required:** No
- **Default:** `5`
- **Range:** 1-20
- **Description:** Maximum concurrent web scraping operations
- **Example:** `3`

#### SCRAPE_TIMEOUT
- **Type:** Integer (seconds)
- **Required:** No
- **Default:** `30`
- **Range:** 10-120
- **Description:** Timeout for individual scrape operations
- **Example:** `60`

#### RESPECT_ROBOTS_TXT
- **Type:** Boolean
- **Required:** No
- **Default:** `true`
- **Description:** Whether to respect robots.txt files
- **Example:** `false`

#### USER_AGENT
- **Type:** String
- **Required:** No
- **Default:** `Intelligent-Research-Assistant/1.0`
- **Description:** User agent string for web requests
- **Example:** `Mozilla/5.0 ...`

### Application Configuration

#### DEBUG
- **Type:** Boolean
- **Required:** No
- **Default:** `false`
- **Description:** Enable debug mode with verbose logging
- **Example:** `true`

#### LOG_LEVEL
- **Type:** String
- **Required:** No
- **Default:** `INFO`
- **Options:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Description:** Logging verbosity level
- **Example:** `DEBUG`

#### GRADIO_HOST
- **Type:** String (IP address)
- **Required:** No
- **Default:** `127.0.0.1`
- **Description:** Host to bind Gradio server
- **Example:** `0.0.0.0` (allow external access)

#### GRADIO_PORT
- **Type:** Integer
- **Required:** No
- **Default:** `7860`
- **Range:** 1024-65535
- **Description:** Port for Gradio web interface
- **Example:** `8080`

### Research Defaults

#### DEFAULT_MAX_SOURCES
- **Type:** Integer
- **Required:** No
- **Default:** `10`
- **Range:** 1-50
- **Description:** Default maximum sources to gather
- **Example:** `15`

#### DEFAULT_TIMEOUT_SECONDS
- **Type:** Integer
- **Required:** No
- **Default:** `120`
- **Range:** 30-600
- **Description:** Default research timeout in seconds
- **Example:** `180`

#### DEFAULT_REPORT_STYLE
- **Type:** String
- **Required:** No
- **Default:** `academic`
- **Options:** `academic`, `casual`, `technical`
- **Description:** Default report writing style
- **Example:** `technical`

#### DEFAULT_REPORT_LENGTH
- **Type:** String
- **Required:** No
- **Default:** `medium`
- **Options:** `short`, `medium`, `long`
- **Description:** Default report length
- **Example:** `long`

### Rate Limiting

#### GEMINI_RATE_LIMIT
- **Type:** Integer
- **Required:** No
- **Default:** `60`
- **Range:** 1-120
- **Description:** Maximum Gemini API requests per minute
- **Example:** `30`

#### WEB_SEARCH_RATE_LIMIT
- **Type:** Integer
- **Required:** No
- **Default:** `30`
- **Range:** 1-60
- **Description:** Maximum web searches per minute
- **Example:** `15`

#### SEARCH_CACHE_DURATION
- **Type:** Integer (hours)
- **Required:** No
- **Default:** `24`
- **Range:** 0-168
- **Description:** Cache duration for search results
- **Example:** `48`

#### VECTOR_CACHE_DURATION
- **Type:** Integer (hours)
- **Required:** No
- **Default:** `1`
- **Range:** 0-24
- **Description:** Cache duration for vector search results
- **Example:** `2`

## Advanced Configuration

### Custom Database Location

```bash
# Use external drive
CHROMA_DB_PATH=/mnt/external/chroma_db

# Use network storage
CHROMA_DB_PATH=//network/share/chroma_db

# Use temp directory (not recommended)
CHROMA_DB_PATH=/tmp/chroma_db
```

### Multiple Environments

Create separate .env files:

```bash
# .env.development
DEBUG=true
LOG_LEVEL=DEBUG
DEFAULT_MAX_SOURCES=5

# .env.production
DEBUG=false
LOG_LEVEL=WARNING
DEFAULT_MAX_SOURCES=15

# .env.testing
DEBUG=true
LOG_LEVEL=DEBUG
GEMINI_API_KEY=test_key
```

Load specific environment:
```bash
cp .env.production .env
python launch_ui.py
```

### Environment-Specific Configuration

```python
# config.py
import os

env = os.getenv('ENVIRONMENT', 'development')

if env == 'production':
    # Production settings
    pass
elif env == 'development':
    # Development settings
    pass
```

## Performance Tuning

### For Speed

```bash
# Minimize sources
DEFAULT_MAX_SOURCES=5

# Shorter reports
DEFAULT_REPORT_LENGTH=short

# Aggressive caching
SEARCH_CACHE_DURATION=72
VECTOR_CACHE_DURATION=24

# Reduce timeouts
DEFAULT_TIMEOUT_SECONDS=60
SCRAPE_TIMEOUT=15

# Fewer concurrent operations
MAX_CONCURRENT_SCRAPES=3
```

### For Quality

```bash
# More sources
DEFAULT_MAX_SOURCES=20

# Longer reports
DEFAULT_REPORT_LENGTH=long

# Fresh data
SEARCH_CACHE_DURATION=1
VECTOR_CACHE_DURATION=0

# Longer timeouts
DEFAULT_TIMEOUT_SECONDS=300
SCRAPE_TIMEOUT=60

# More concurrent operations
MAX_CONCURRENT_SCRAPES=10
```

### For Reliability

```bash
# Conservative rate limits
GEMINI_RATE_LIMIT=30
WEB_SEARCH_RATE_LIMIT=15

# Longer timeouts
DEFAULT_TIMEOUT_SECONDS=180
SCRAPE_TIMEOUT=45

# Respect robots.txt
RESPECT_ROBOTS_TXT=true

# Moderate concurrency
MAX_CONCURRENT_SCRAPES=5
```

## Security Best Practices

### API Key Security

1. **Never commit .env to version control:**
   ```bash
   # .gitignore should contain:
   .env
   ```

2. **Use environment variables in production:**
   ```bash
   export GEMINI_API_KEY=your_key
   python launch_ui.py
   ```

3. **Rotate keys regularly:**
   - Generate new API keys monthly
   - Update .env file
   - Revoke old keys

4. **Restrict key permissions:**
   - Use API keys with minimal required permissions
   - Set usage quotas

### File Permissions

```bash
# Secure .env file (Linux/macOS)
chmod 600 .env

# Secure credentials directory
chmod 700 credentials/
chmod 600 credentials/*.json
```

### Network Security

```bash
# Local only (default)
GRADIO_HOST=127.0.0.1

# External access (use with caution)
GRADIO_HOST=0.0.0.0

# Use non-standard port
GRADIO_PORT=8443
```

### Logging Security

```bash
# Don't log sensitive data
LOG_LEVEL=WARNING  # In production

# Review logs for exposed keys
grep -i "api" logs/*.log
```

---

For more information, see the main README.md and troubleshooting guide.
