# Troubleshooting Guide

## Overview

This guide helps you diagnose and resolve common issues with the Intelligent Research Assistant. Issues are organized by category with step-by-step solutions.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Configuration Issues](#configuration-issues)
- [API Issues](#api-issues)
- [Database Issues](#database-issues)
- [Web Scraping Issues](#web-scraping-issues)
- [Performance Issues](#performance-issues)
- [UI Issues](#ui-issues)
- [Google Sheets Issues](#google-sheets-issues)
- [Debug Mode](#debug-mode)
- [Getting Help](#getting-help)

## Installation Issues

### Issue: "pip install fails with dependency conflicts"

**Symptoms:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
```

**Solutions:**

1. **Use a fresh virtual environment:**
   ```bash
   # Remove old environment
   rm -rf venv  # Linux/macOS
   rmdir /s venv  # Windows
   
   # Create new environment
   python -m venv venv
   
   # Activate
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate  # Windows
   
   # Install
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Install with --no-deps flag (advanced):**
   ```bash
   pip install --no-deps -r requirements.txt
   ```

3. **Check Python version:**
   ```bash
   python --version  # Should be 3.8 or higher
   ```

### Issue: "ModuleNotFoundError after installation"

**Symptoms:**
```
ModuleNotFoundError: No module named 'agents'
```

**Solutions:**

1. **Verify virtual environment is activated:**
   ```bash
   # You should see (venv) in your prompt
   which python  # Linux/macOS - should point to venv
   where python  # Windows - should point to venv
   ```

2. **Reinstall in correct environment:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Check you're in the project directory:**
   ```bash
   pwd  # Linux/macOS
   cd  # Windows
   # Should be in intelligent-research-assistant/
   ```

### Issue: "Selenium WebDriver not found"

**Symptoms:**
```
selenium.common.exceptions.WebDriverException: Message: 'chromedriver' executable needs to be in PATH
```

**Solutions:**

1. **Install ChromeDriver (automatic):**
   ```bash
   pip install webdriver-manager
   ```
   The system will auto-download the driver.

2. **Manual installation:**
   - Download ChromeDriver from https://chromedriver.chromium.org/
   - Add to system PATH
   - Or place in project directory

3. **Disable web scraping temporarily:**
   In the UI, uncheck "Enable Web Scraping"

## Configuration Issues

### Issue: "GEMINI_API_KEY not found"

**Symptoms:**
```
ConfigurationError: GEMINI_API_KEY not found in environment
```

**Solutions:**

1. **Create .env file:**
   ```bash
   cp .env.template .env
   ```

2. **Add your API key:**
   ```bash
   # Edit .env file
   GEMINI_API_KEY=your_actual_api_key_here
   ```

3. **Verify .env is in project root:**
   ```bash
   ls -la .env  # Linux/macOS
   dir .env  # Windows
   ```

4. **Check for typos:**
   - Variable name must be exactly `GEMINI_API_KEY`
   - No spaces around the `=`
   - No quotes needed around the value

5. **Test configuration:**
   ```bash
   python verify_setup.py
   ```

### Issue: "Configuration validation failed"

**Symptoms:**
```
Configuration errors found:
  - Invalid GEMINI_API_KEY format
```

**Solutions:**

1. **Run verification script:**
   ```bash
   python verify_setup.py
   ```
   This will show specific errors.

2. **Check API key format:**
   - Gemini keys typically start with "AI"
   - Should be 39 characters long
   - No extra spaces or newlines

3. **Verify API key is active:**
   - Visit https://makersuite.google.com/app/apikey
   - Check if key is enabled
   - Generate new key if needed

4. **Check file encoding:**
   - .env file should be UTF-8
   - No BOM (Byte Order Mark)

### Issue: "Environment variables not loading"

**Symptoms:**
- Configuration shows default values
- API keys not recognized

**Solutions:**

1. **Check .env file location:**
   ```bash
   # Must be in project root, same directory as launch_ui.py
   ls .env
   ```

2. **Verify python-dotenv is installed:**
   ```bash
   pip show python-dotenv
   ```

3. **Check for .env in .gitignore:**
   ```bash
   cat .gitignore | grep .env
   ```
   Should show `.env` is ignored.

4. **Restart application:**
   - Environment variables are loaded at startup
   - Changes require restart

## API Issues

### Issue: "Gemini API rate limit exceeded"

**Symptoms:**
```
RateLimitError: Gemini API rate limit exceeded
```

**Solutions:**

1. **Reduce rate limit in .env:**
   ```bash
   GEMINI_RATE_LIMIT=30  # Reduce from 60
   ```

2. **Wait before retrying:**
   - Free tier: 60 requests per minute
   - Wait 1 minute and try again

3. **Reduce max sources:**
   - In UI, set "Max Sources" to 5 instead of 10
   - Fewer sources = fewer API calls

4. **Check API quota:**
   - Visit https://makersuite.google.com/
   - Check your usage and limits

### Issue: "Gemini API authentication failed"

**Symptoms:**
```
APIError: Invalid API key
```

**Solutions:**

1. **Verify API key:**
   ```bash
   # In .env file
   GEMINI_API_KEY=your_key_here
   ```

2. **Generate new API key:**
   - Visit https://makersuite.google.com/app/apikey
   - Create new key
   - Update .env file

3. **Check API is enabled:**
   - Ensure Gemini API is enabled in your Google Cloud project
   - May need to enable billing (free tier available)

4. **Test API directly:**
   ```python
   import google.generativeai as genai
   genai.configure(api_key="your_key")
   model = genai.GenerativeModel('gemini-pro')
   response = model.generate_content("Hello")
   print(response.text)
   ```

### Issue: "SerpAPI quota exceeded"

**Symptoms:**
```
SerpAPI Error: Monthly search limit reached
```

**Solutions:**

1. **System will auto-fallback to DuckDuckGo:**
   - No action needed
   - DuckDuckGo is free and unlimited

2. **Remove SerpAPI key to force DuckDuckGo:**
   ```bash
   # In .env, comment out or remove:
   # SERPAPI_KEY=your_key
   ```

3. **Upgrade SerpAPI plan:**
   - Visit https://serpapi.com/pricing
   - Free tier: 100 searches/month

4. **Wait for quota reset:**
   - Quota resets monthly
   - Check at https://serpapi.com/dashboard

## Database Issues

### Issue: "ChromaDB initialization failed"

**Symptoms:**
```
Error: Could not initialize ChromaDB at ./data/chroma_db
```

**Solutions:**

1. **Create data directory:**
   ```bash
   mkdir -p data/chroma_db  # Linux/macOS
   mkdir data\chroma_db  # Windows
   ```

2. **Check permissions:**
   ```bash
   # Linux/macOS
   chmod 755 data
   chmod 755 data/chroma_db
   ```

3. **Change database path:**
   ```bash
   # In .env
   CHROMA_DB_PATH=./custom/path/chroma_db
   ```

4. **Delete and recreate:**
   ```bash
   rm -rf data/chroma_db  # Linux/macOS
   rmdir /s data\chroma_db  # Windows
   # Restart application - will recreate
   ```

### Issue: "ChromaDB corrupted"

**Symptoms:**
```
Error: Database corruption detected
```

**Solutions:**

1. **Backup and reset:**
   ```bash
   # Backup
   mv data/chroma_db data/chroma_db.backup
   
   # Restart application - creates new DB
   python launch_ui.py
   ```

2. **Clear old documents:**
   ```python
   from utils.chroma_manager import ChromaManager
   manager = ChromaManager("./data/chroma_db")
   manager.delete_old_documents(days=0)  # Delete all
   ```

### Issue: "Vector search returns no results"

**Symptoms:**
- Vector search enabled but returns nothing
- "No relevant documents found"

**Solutions:**

1. **Check if database has documents:**
   ```python
   from utils.chroma_manager import ChromaManager
   manager = ChromaManager("./data/chroma_db")
   count = manager.collection.count()
   print(f"Documents in DB: {count}")
   ```

2. **Lower similarity threshold:**
   ```python
   # In vector_search_agent.py, reduce threshold
   similarity_threshold = 0.4  # Default is 0.6
   ```

3. **Add documents manually:**
   ```python
   from agents.vector_search_agent import VectorSearchAgent
   agent = VectorSearchAgent(chroma_manager, gemini_client)
   agent.add_documents(
       texts=["Sample content"],
       metadata=[{"source": "manual", "timestamp": "2024-01-01"}]
   )
   ```

## Web Scraping Issues

### Issue: "Web scraping timeouts"

**Symptoms:**
```
TimeoutError: Scraping exceeded 30 second timeout
```

**Solutions:**

1. **Increase timeout:**
   ```bash
   # In .env
   SCRAPE_TIMEOUT=60  # Increase from 30
   ```

2. **Reduce concurrent scrapes:**
   ```bash
   # In .env
   MAX_CONCURRENT_SCRAPES=3  # Reduce from 5
   ```

3. **Disable problematic sites:**
   - Some sites are slow or block scrapers
   - System will skip and continue with others

4. **Disable web scraping:**
   - In UI, uncheck "Enable Web Scraping"
   - Rely on web search snippets instead

### Issue: "403 Forbidden or 429 Too Many Requests"

**Symptoms:**
```
HTTP Error 403: Forbidden
HTTP Error 429: Too Many Requests
```

**Solutions:**

1. **Respect rate limits:**
   ```bash
   # In .env
   MAX_CONCURRENT_SCRAPES=2
   ```

2. **Change User-Agent:**
   ```bash
   # In .env
   USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36
   ```

3. **Enable robots.txt respect:**
   ```bash
   # In .env
   RESPECT_ROBOTS_TXT=true
   ```

4. **Wait and retry:**
   - Some sites have temporary rate limits
   - Try again in a few minutes

### Issue: "Selenium browser won't start"

**Symptoms:**
```
WebDriverException: Chrome failed to start
```

**Solutions:**

1. **Install Chrome browser:**
   - Selenium requires Chrome or Chromium
   - Download from https://www.google.com/chrome/

2. **Update ChromeDriver:**
   ```bash
   pip install --upgrade selenium
   pip install webdriver-manager
   ```

3. **Run in headless mode:**
   - System already uses headless mode
   - Check Chrome is installed

4. **Use BeautifulSoup only:**
   - Disable Selenium in code
   - Only scrape static pages

## Performance Issues

### Issue: "Research takes too long"

**Symptoms:**
- Research exceeds 2 minutes
- Timeout errors

**Solutions:**

1. **Reduce max sources:**
   ```bash
   # In UI or .env
   DEFAULT_MAX_SOURCES=5  # Reduce from 10
   ```

2. **Disable web scraping:**
   - In UI, uncheck "Enable Web Scraping"
   - Scraping is slowest operation

3. **Increase timeout:**
   ```bash
   # In .env
   DEFAULT_TIMEOUT_SECONDS=180  # Increase from 120
   ```

4. **Check internet connection:**
   - Slow connection affects all operations
   - Test with: `ping google.com`

5. **Reduce report length:**
   - In UI, select "Short" instead of "Medium" or "Long"

### Issue: "High memory usage"

**Symptoms:**
- System becomes slow
- Out of memory errors

**Solutions:**

1. **Reduce max sources:**
   - Fewer sources = less memory

2. **Clear ChromaDB periodically:**
   ```python
   from utils.chroma_manager import ChromaManager
   manager = ChromaManager("./data/chroma_db")
   manager.delete_old_documents(days=30)
   ```

3. **Restart application:**
   - Clears memory leaks
   - Fresh start

4. **Close other applications:**
   - Free up system memory

### Issue: "Slow UI response"

**Symptoms:**
- UI feels sluggish
- Buttons don't respond immediately

**Solutions:**

1. **Check if research is running:**
   - UI may be busy processing
   - Wait for completion

2. **Restart Gradio:**
   ```bash
   # Stop with Ctrl+C
   # Restart
   python launch_ui.py
   ```

3. **Clear browser cache:**
   - Refresh page (F5)
   - Hard refresh (Ctrl+F5)

4. **Try different browser:**
   - Chrome recommended
   - Firefox also works well

## UI Issues

### Issue: "Gradio UI won't start"

**Symptoms:**
```
OSError: [Errno 48] Address already in use
```

**Solutions:**

1. **Change port:**
   ```bash
   # In .env
   GRADIO_PORT=7861  # Change from 7860
   ```

2. **Kill process using port:**
   ```bash
   # Linux/macOS
   lsof -ti:7860 | xargs kill -9
   
   # Windows
   netstat -ano | findstr :7860
   taskkill /PID <PID> /F
   ```

3. **Bind to different host:**
   ```bash
   # In .env
   GRADIO_HOST=0.0.0.0  # Allow external access
   ```

### Issue: "UI shows blank page"

**Symptoms:**
- Browser loads but shows nothing
- White screen

**Solutions:**

1. **Check console for errors:**
   - Open browser DevTools (F12)
   - Look for JavaScript errors

2. **Clear browser cache:**
   - Hard refresh (Ctrl+Shift+R)
   - Or clear cache in settings

3. **Try different browser:**
   - Chrome, Firefox, or Edge

4. **Check Gradio version:**
   ```bash
   pip show gradio
   # Should be 4.44.0 or compatible
   ```

### Issue: "Progress bar stuck"

**Symptoms:**
- Progress shows 0% or stuck at some percentage
- Research seems to hang

**Solutions:**

1. **Wait for timeout:**
   - System has 120 second timeout
   - Will complete or error

2. **Check logs:**
   - Look at console output
   - May show which agent is stuck

3. **Cancel and retry:**
   - Refresh page
   - Submit query again

4. **Enable debug mode:**
   ```bash
   # In .env
   DEBUG=true
   LOG_LEVEL=DEBUG
   ```

## Google Sheets Issues

### Issue: "Google Sheets authentication failed"

**Symptoms:**
```
Error: Could not authenticate with Google Sheets
```

**Solutions:**

1. **Check credentials file exists:**
   ```bash
   ls credentials/your-service-account.json
   ```

2. **Verify path in .env:**
   ```bash
   GOOGLE_SHEETS_CREDENTIALS_PATH=./credentials/your-file.json
   ```

3. **Check credentials format:**
   - Must be service account JSON
   - Should contain "type": "service_account"

4. **Regenerate credentials:**
   - Go to Google Cloud Console
   - Create new service account
   - Download new JSON file

### Issue: "Permission denied on Google Sheets"

**Symptoms:**
```
Error: Insufficient permissions to access sheet
```

**Solutions:**

1. **Share sheet with service account:**
   - Open your Google Sheet
   - Click "Share"
   - Add service account email (from JSON file)
   - Give "Editor" permission

2. **Check API is enabled:**
   - Go to Google Cloud Console
   - Enable "Google Sheets API"
   - Enable "Google Drive API"

3. **Create new sheet:**
   - System will auto-create if it doesn't exist
   - Ensure service account has Drive access

### Issue: "Quota exceeded for Google Sheets"

**Symptoms:**
```
Error: Quota exceeded for quota metric 'Write requests'
```

**Solutions:**

1. **Wait for quota reset:**
   - Quotas reset every 100 seconds
   - Wait and try again

2. **Reduce save frequency:**
   - Don't save every research
   - Manual save option

3. **Disable Google Sheets:**
   ```bash
   # In .env, comment out:
   # GOOGLE_SHEETS_CREDENTIALS_PATH=...
   ```

## Debug Mode

### Enabling Debug Mode

```bash
# In .env
DEBUG=true
LOG_LEVEL=DEBUG
```

### What Debug Mode Shows

- Detailed API calls
- Agent execution steps
- Timing information
- Error stack traces
- Configuration values

### Reading Debug Logs

```
[2024-01-01 12:00:00] DEBUG - RouterAgent: Analyzing query...
[2024-01-01 12:00:03] DEBUG - RouterAgent: Generated 5 search queries
[2024-01-01 12:00:03] INFO - Starting parallel data collection
[2024-01-01 12:00:05] DEBUG - WebSearchAgent: Using DuckDuckGo
[2024-01-01 12:00:10] DEBUG - WebSearchAgent: Found 25 results
```

### Common Debug Patterns

**API Errors:**
```
ERROR - GeminiClient: API call failed: Invalid API key
```
→ Check GEMINI_API_KEY in .env

**Timeout:**
```
WARNING - MainOrchestrator: Agent timeout after 30 seconds
```
→ Increase timeout or reduce sources

**Rate Limit:**
```
WARNING - GeminiClient: Rate limit reached, waiting...
```
→ Reduce GEMINI_RATE_LIMIT

## Getting Help

### Before Asking for Help

1. **Run verification script:**
   ```bash
   python verify_setup.py
   ```

2. **Enable debug mode:**
   ```bash
   # In .env
   DEBUG=true
   LOG_LEVEL=DEBUG
   ```

3. **Check logs for errors:**
   - Look at console output
   - Note exact error messages

4. **Try basic troubleshooting:**
   - Restart application
   - Check internet connection
   - Verify API keys

### Information to Include

When reporting issues, include:

1. **Error message** (exact text)
2. **Steps to reproduce**
3. **Configuration** (without API keys!)
4. **Python version:** `python --version`
5. **Operating system**
6. **Debug logs** (relevant portions)

### Diagnostic Commands

```bash
# Python version
python --version

# Package versions
pip list

# Configuration test
python verify_setup.py

# Test Gemini API
python -c "import google.generativeai as genai; print('Gemini OK')"

# Test ChromaDB
python -c "import chromadb; print('ChromaDB OK')"

# Test Gradio
python -c "import gradio; print('Gradio OK')"
```

### Quick Fixes Checklist

- [ ] Virtual environment activated
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] .env file exists and configured
- [ ] GEMINI_API_KEY is valid
- [ ] data/chroma_db directory exists
- [ ] Internet connection working
- [ ] No firewall blocking
- [ ] Port 7860 available
- [ ] Python 3.8 or higher

---

If you've tried everything and still have issues, please open an issue on the repository with the diagnostic information above.
