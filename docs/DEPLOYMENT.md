# Deployment Guide - Intelligent Research Assistant

## Hosting Options

### Option 1: Hugging Face Spaces (Recommended - FREE)

**Pros:**
- Free hosting with GPU support
- Built-in Gradio support
- Easy deployment
- Automatic HTTPS
- Good for demos and small-scale use

**Steps:**
1. Create account at https://huggingface.co/
2. Create a new Space (select Gradio SDK)
3. Upload your files or connect to GitHub
4. Add secrets in Space settings:
   - `GEMINI_API_KEY`
   - `SERPAPI_KEY` (optional)
   - `GOOGLE_SHEETS_CREDENTIALS_PATH` (optional)
5. Space will auto-deploy

**Limitations:**
- 16GB RAM limit
- CPU-only (unless you upgrade)
- Public by default (can make private with Pro)

---

### Option 2: Railway.app (Easy - FREE tier available)

**Pros:**
- Free $5/month credit
- Easy deployment from GitHub
- Automatic HTTPS
- Good performance

**Steps:**
1. Sign up at https://railway.app/
2. Create new project from GitHub repo
3. Add environment variables in Railway dashboard
4. Deploy automatically

**Cost:** Free tier ($5/month credit), then ~$5-20/month

---

### Option 3: Render.com (FREE tier)

**Pros:**
- Free tier available
- Easy deployment
- Automatic HTTPS
- Good documentation

**Steps:**
1. Sign up at https://render.com/
2. Create new Web Service
3. Connect GitHub repo
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `python launch_ui.py --host 0.0.0.0 --port $PORT`
6. Add environment variables

**Limitations:**
- Free tier spins down after inactivity
- 512MB RAM on free tier

---

### Option 4: Google Cloud Run (Scalable)

**Pros:**
- Pay per use (very cheap for low traffic)
- Scales automatically
- Integrates well with Google APIs

**Steps:**
1. Create Dockerfile (see below)
2. Build and push to Google Container Registry
3. Deploy to Cloud Run
4. Set environment variables

**Cost:** ~$0-10/month for low traffic

---

### Option 5: AWS EC2 / DigitalOcean / Linode (Full Control)

**Pros:**
- Full control
- Can run 24/7
- Good for production

**Steps:**
1. Create a VM instance
2. Install Python 3.10+
3. Clone your repo
4. Install dependencies
5. Run with systemd or PM2
6. Set up nginx reverse proxy
7. Configure SSL with Let's Encrypt

**Cost:** ~$5-20/month

---

## Quick Start Files

### Dockerfile (for containerized deployment)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Chrome for Selenium
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data/chroma_db

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "launch_ui.py", "--host", "0.0.0.0", "--port", "7860"]
```

### docker-compose.yml (for local testing)

```yaml
version: '3.8'

services:
  research-assistant:
    build: .
    ports:
      - "7860:7860"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - SERPAPI_KEY=${SERPAPI_KEY}
      - GRADIO_HOST=0.0.0.0
      - GRADIO_PORT=7860
    volumes:
      - ./data:/app/data
      - ./credentials:/app/credentials
    restart: unless-stopped
```

### .dockerignore

```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
venv/
env/
.env
.git/
.gitignore
*.md
tests/
.pytest_cache/
.vscode/
.idea/
*.log
data/chroma_db/*
```

---

## Environment Variables for Production

Make sure to set these in your hosting platform:

```bash
# Required
GEMINI_API_KEY=your_gemini_api_key

# Optional but recommended
SERPAPI_KEY=your_serpapi_key

# Optional
GOOGLE_SHEETS_CREDENTIALS_PATH=./credentials/your-credentials.json

# Database
CHROMA_DB_PATH=./data/chroma_db

# Server config
GRADIO_HOST=0.0.0.0
GRADIO_PORT=7860

# Performance
DEBUG=false
LOG_LEVEL=INFO
```

---

## Security Considerations

1. **Never commit `.env` file** - it contains API keys
2. **Use environment variables** in production
3. **Enable authentication** if making public:
   ```python
   # In launch_ui.py, modify the launch call:
   app.launch(
       auth=("username", "password"),  # Add this
       host=host,
       port=port
   )
   ```
4. **Rate limiting** - Consider adding rate limits for public deployments
5. **HTTPS** - Always use HTTPS in production (most platforms provide this)

---

## Recommended: Hugging Face Spaces Deployment

**Easiest option for getting started:**

1. Create `app.py` in root:
```python
import sys
from ui.app import main

if __name__ == "__main__":
    sys.exit(main())
```

2. Create `README.md` for Hugging Face:
```markdown
---
title: Intelligent Research Assistant
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: launch_ui.py
pinned: false
---

# Intelligent Research Assistant

AI-powered research automation system using Google Gemini.
```

3. Push to Hugging Face:
```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/research-assistant
git push hf main
```

4. Add secrets in Space settings

Done! Your app will be live at `https://huggingface.co/spaces/YOUR_USERNAME/research-assistant`

---

## Performance Tips

1. **Use SerpAPI** instead of DuckDuckGo for reliable results
2. **Enable caching** to reduce API calls
3. **Set reasonable timeouts** (120-180 seconds)
4. **Monitor API usage** to avoid quota limits
5. **Use vector database** for faster repeated queries

---

## Monitoring

Add basic monitoring:

```python
# In your .env
ENABLE_ANALYTICS=true
```

Consider using:
- **Sentry** for error tracking
- **Google Analytics** for usage stats
- **Uptime Robot** for availability monitoring

---

## Need Help?

- Check logs: `tail -f logs/app.log`
- Test locally first: `python launch_ui.py --debug`
- Validate config: `python launch_ui.py --validate-config`
- Health check: `python launch_ui.py --health-check`
