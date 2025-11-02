"""
Configuration management for the Intelligent Research Assistant.
Handles environment variable loading and configuration validation.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class ResearchConfig:
    """Configuration for research parameters"""
    max_sources: int = 10
    enable_web_scraping: bool = True
    enable_vector_search: bool = True
    report_style: str = "academic"  # academic, casual, technical
    report_length: str = "medium"   # short, medium, long
    timeout_seconds: int = 120

@dataclass
class APIConfig:
    """Configuration for external API services"""
    gemini_api_key: Optional[str] = None
    serpapi_key: Optional[str] = None
    google_sheets_credentials_path: Optional[str] = None
    
    def __post_init__(self):
        # Load from environment variables
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.serpapi_key = os.getenv('SERPAPI_KEY')
        self.google_sheets_credentials_path = os.getenv('GOOGLE_SHEETS_CREDENTIALS_PATH')

@dataclass
class DatabaseConfig:
    """Configuration for ChromaDB vector database"""
    chroma_db_path: str = "./data/chroma_db"
    collection_name: str = "research_knowledge"
    embedding_model: str = "gemini"
    
    def __post_init__(self):
        # Load from environment variables with defaults
        self.chroma_db_path = os.getenv('CHROMA_DB_PATH', self.chroma_db_path)
        self.collection_name = os.getenv('CHROMA_COLLECTION_NAME', self.collection_name)

@dataclass
class ScrapingConfig:
    """Configuration for web scraping"""
    max_concurrent_scrapes: int = 5
    scrape_timeout: int = 30
    respect_robots_txt: bool = True
    user_agent: str = "Intelligent-Research-Assistant/1.0"
    
    def __post_init__(self):
        # Load from environment variables with defaults
        self.max_concurrent_scrapes = int(os.getenv('MAX_CONCURRENT_SCRAPES', self.max_concurrent_scrapes))
        self.scrape_timeout = int(os.getenv('SCRAPE_TIMEOUT', self.scrape_timeout))
        self.respect_robots_txt = os.getenv('RESPECT_ROBOTS_TXT', 'true').lower() == 'true'
        self.user_agent = os.getenv('USER_AGENT', self.user_agent)

@dataclass
class SheetsConfig:
    """Configuration for Google Sheets integration"""
    spreadsheet_name: str = "Research Assistant Data"
    worksheet_name: str = "Research Data"
    auto_cleanup_days: int = 90
    max_requests_per_minute: int = 100
    
    def __post_init__(self):
        # Load from environment variables with defaults
        self.spreadsheet_name = os.getenv('GOOGLE_SHEETS_NAME', self.spreadsheet_name)
        self.worksheet_name = os.getenv('GOOGLE_SHEETS_WORKSHEET', self.worksheet_name)
        self.auto_cleanup_days = int(os.getenv('GOOGLE_SHEETS_CLEANUP_DAYS', self.auto_cleanup_days))

@dataclass
class AppConfig:
    """Main application configuration"""
    api: APIConfig
    database: DatabaseConfig
    scraping: ScrapingConfig
    research: ResearchConfig
    sheets: SheetsConfig
    
    # Application settings
    debug: bool = False
    log_level: str = "INFO"
    gradio_port: int = 7860
    gradio_host: str = "127.0.0.1"
    
    def __post_init__(self):
        # Load application settings from environment
        self.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        self.log_level = os.getenv('LOG_LEVEL', self.log_level)
        self.gradio_port = int(os.getenv('GRADIO_PORT', self.gradio_port))
        self.gradio_host = os.getenv('GRADIO_HOST', self.gradio_host)

def load_config() -> AppConfig:
    """Load and return the complete application configuration"""
    api_config = APIConfig()
    database_config = DatabaseConfig()
    scraping_config = ScrapingConfig()
    research_config = ResearchConfig()
    sheets_config = SheetsConfig()
    
    config = AppConfig(
        api=api_config,
        database=database_config,
        scraping=scraping_config,
        research=research_config,
        sheets=sheets_config
    )
    
    return config

def validate_config(config: AppConfig) -> list[str]:
    """Validate configuration and return list of errors"""
    errors = []
    
    # Check required API keys
    if not config.api.gemini_api_key:
        errors.append("GEMINI_API_KEY is required")
    
    # Validate port range
    if not (1024 <= config.gradio_port <= 65535):
        errors.append("GRADIO_PORT must be between 1024 and 65535")
    
    # Validate timeout values
    if config.research.timeout_seconds < 30:
        errors.append("Research timeout must be at least 30 seconds")
    
    if config.scraping.scrape_timeout < 5:
        errors.append("Scrape timeout must be at least 5 seconds")
    
    # Validate report style
    valid_styles = ["academic", "casual", "technical"]
    if config.research.report_style not in valid_styles:
        errors.append(f"Report style must be one of: {valid_styles}")
    
    # Validate report length
    valid_lengths = ["short", "medium", "long"]
    if config.research.report_length not in valid_lengths:
        errors.append(f"Report length must be one of: {valid_lengths}")
    
    return errors