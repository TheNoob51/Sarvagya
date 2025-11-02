"""
Main configuration module for the Intelligent Research Assistant.
This module provides easy access to configuration throughout the application.
"""

from utils.config import load_config, validate_config, AppConfig
import sys
import os

# Global configuration instance
_config: AppConfig = None

def get_config() -> AppConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = load_config()
    return _config

def initialize_config() -> AppConfig:
    """Initialize and validate configuration on application startup"""
    config = load_config()
    
    # Validate configuration
    errors = validate_config(config)
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease check your .env file and fix the above errors.")
        print("You can use .env.template as a reference.")
        sys.exit(1)
    
    # Create data directory if it doesn't exist
    os.makedirs(config.database.chroma_db_path, exist_ok=True)
    
    # Set global config
    global _config
    _config = config
    
    if config.debug:
        print("Configuration loaded successfully:")
        print(f"  - Gemini API: {'✓' if config.api.gemini_api_key else '✗'}")
        print(f"  - SerpAPI: {'✓' if config.api.serpapi_key else '✗ (will use DuckDuckGo)'}")
        print(f"  - Google Sheets: {'✓' if config.api.google_sheets_credentials_path else '✗'}")
        print(f"  - Database path: {config.database.chroma_db_path}")
        print(f"  - Gradio UI: http://{config.gradio_host}:{config.gradio_port}")
    
    return config

if __name__ == "__main__":
    # Test configuration loading
    try:
        config = initialize_config()
        print("✓ Configuration loaded and validated successfully!")
    except SystemExit:
        print("✗ Configuration validation failed!")
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")