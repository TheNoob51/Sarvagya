"""
Main application entry point for the Intelligent Research Assistant.
Provides a complete Gradio web interface with all functionality integrated.
"""

import sys
import os
import argparse
import time
from typing import Optional, Dict, Any
from pathlib import Path

# Add the parent directory to the path so we can import from the project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.gradio_interface import create_gradio_app
from config import initialize_config, get_config
from utils.logging_config import setup_logging, silence_noisy_loggers
import structlog

logger = structlog.get_logger(__name__)

def check_dependencies() -> Dict[str, Any]:
    """
    Check if all required dependencies are installed and accessible.
    
    Returns:
        Dictionary with dependency check results
    """
    dependencies = {
        "all_available": True,
        "missing": [],
        "warnings": []
    }
    
    required_modules = [
        ("gradio", "Gradio UI framework"),
        ("google.generativeai", "Google Gemini API"),
        ("chromadb", "ChromaDB vector database"),
        ("bs4", "BeautifulSoup web scraping"),
        ("selenium", "Selenium web scraping"),
        ("structlog", "Structured logging"),
        ("dotenv", "Environment variable loading")
    ]
    
    for module_name, description in required_modules:
        try:
            # Handle module names with dots
            if "." in module_name:
                parts = module_name.split(".")
                module = __import__(parts[0])
                for part in parts[1:]:
                    module = getattr(module, part)
            else:
                __import__(module_name)
        except (ImportError, AttributeError):
            dependencies["all_available"] = False
            dependencies["missing"].append(f"{description} ({module_name})")
    
    # Check optional dependencies
    optional_modules = [
        ("gspread", "Google Sheets integration"),
        ("duckduckgo_search", "DuckDuckGo search")
    ]
    
    for module_name, description in optional_modules:
        try:
            __import__(module_name)
        except ImportError:
            dependencies["warnings"].append(f"Optional: {description} ({module_name})")
    
    return dependencies


def validate_environment() -> Dict[str, Any]:
    """
    Validate the environment setup including directories and permissions.
    
    Returns:
        Dictionary with validation results
    """
    validation = {
        "valid": True,
        "issues": [],
        "warnings": []
    }
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        validation["warnings"].append(".env file not found - using defaults or environment variables")
    
    # Check data directory
    data_dir = Path("data")
    if not data_dir.exists():
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            validation["warnings"].append("Created data directory")
        except Exception as e:
            validation["valid"] = False
            validation["issues"].append(f"Cannot create data directory: {e}")
    
    # Check write permissions
    try:
        test_file = data_dir / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
    except Exception as e:
        validation["valid"] = False
        validation["issues"].append(f"No write permission in data directory: {e}")
    
    return validation


def perform_health_check() -> Dict[str, Any]:
    """
    Perform comprehensive health check on all system components.
    
    Returns:
        Dictionary with health check results
    """
    health = {
        "status": "healthy",
        "components": {},
        "timestamp": time.time()
    }
    
    try:
        # Check configuration
        config = get_config()
        health["components"]["configuration"] = {
            "status": "healthy",
            "gemini_api_configured": bool(config.api.gemini_api_key),
            "serpapi_configured": bool(config.api.serpapi_key),
            "google_sheets_configured": bool(config.api.google_sheets_credentials_path)
        }
        
        # Check database directory
        db_path = Path(config.database.chroma_db_path)
        health["components"]["database"] = {
            "status": "healthy" if db_path.exists() else "warning",
            "path": str(db_path),
            "exists": db_path.exists()
        }
        
        # Check Gemini API connectivity (basic check)
        try:
            import google.generativeai as genai
            if config.api.gemini_api_key:
                genai.configure(api_key=config.api.gemini_api_key)
                health["components"]["gemini_api"] = {
                    "status": "healthy",
                    "configured": True
                }
            else:
                health["components"]["gemini_api"] = {
                    "status": "warning",
                    "configured": False,
                    "message": "API key not configured"
                }
        except Exception as e:
            health["components"]["gemini_api"] = {
                "status": "error",
                "error": str(e)
            }
            health["status"] = "degraded"
        
        # Check Google Sheets availability
        if config.api.google_sheets_credentials_path:
            creds_path = Path(config.api.google_sheets_credentials_path)
            health["components"]["google_sheets"] = {
                "status": "healthy" if creds_path.exists() else "warning",
                "credentials_exist": creds_path.exists()
            }
        else:
            health["components"]["google_sheets"] = {
                "status": "not_configured",
                "message": "Google Sheets integration not configured"
            }
        
    except Exception as e:
        health["status"] = "unhealthy"
        health["error"] = str(e)
        logger.error("Health check failed", error=str(e))
    
    return health


def safe_print(text: str):
    """
    Print text with proper encoding handling for Windows.
    
    Args:
        text: Text to print
    """
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback to ASCII-safe version
        print(text.encode('ascii', 'replace').decode('ascii'))


def print_startup_banner(config):
    """
    Print a startup banner with system information.
    
    Args:
        config: Application configuration
    """
    safe_print("\n" + "="*70)
    safe_print("  Intelligent Research Assistant")
    safe_print("  AI-Powered Research Automation System")
    safe_print("="*70)
    safe_print(f"\nConfiguration:")
    safe_print(f"   - Gemini API: {'[OK] Configured' if config.api.gemini_api_key else '[--] Not configured'}")
    safe_print(f"   - SerpAPI: {'[OK] Configured' if config.api.serpapi_key else '[--] Not configured (will use DuckDuckGo)'}")
    safe_print(f"   - Google Sheets: {'[OK] Configured' if config.api.google_sheets_credentials_path else '[--] Not configured'}")
    safe_print(f"   - Database: {config.database.chroma_db_path}")
    safe_print(f"   - Debug Mode: {'Enabled' if config.debug else 'Disabled'}")
    safe_print(f"\nServer Configuration:")
    safe_print(f"   - Host: {config.gradio_host}")
    safe_print(f"   - Port: {config.gradio_port}")
    safe_print("\n" + "="*70 + "\n")


def main():
    """
    Main entry point for the Intelligent Research Assistant application.
    Includes proper initialization, validation, health checks, and error handling.
    """
    start_time = time.time()
    
    parser = argparse.ArgumentParser(
        description="Intelligent Research Assistant - AI-powered research automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Start with default settings
  %(prog)s --host 0.0.0.0 --port 8080  # Custom host and port
  %(prog)s --share                  # Create public Gradio link
  %(prog)s --debug                  # Enable debug mode
  %(prog)s --validate-config        # Validate configuration only
  %(prog)s --health-check           # Run health check and exit
        """
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind the server to (default: from config)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind the server to (default: from config)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )
    
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit"
    )
    
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Perform health check and exit"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    try:
        # Step 1: Setup logging
        log_level = "DEBUG" if args.debug else args.log_level
        setup_logging(log_level=log_level, debug=args.debug)
        silence_noisy_loggers()
        
        logger.info("Application startup initiated", 
                   log_level=log_level,
                   debug_mode=args.debug)
        
        # Step 2: Check dependencies
        safe_print("[*] Checking dependencies...")
        dep_check = check_dependencies()
        
        if not dep_check["all_available"]:
            safe_print("[!] Missing required dependencies:")
            for missing in dep_check["missing"]:
                safe_print(f"   - {missing}")
            safe_print("\nPlease install missing dependencies:")
            safe_print("   pip install -r requirements.txt")
            return 1
        
        if dep_check["warnings"]:
            safe_print("[!] Optional dependencies not available:")
            for warning in dep_check["warnings"]:
                safe_print(f"   - {warning}")
        
        safe_print("[OK] All required dependencies available\n")
        logger.info("Dependency check passed", 
                   missing=len(dep_check["missing"]),
                   warnings=len(dep_check["warnings"]))
        
        # Step 3: Validate environment
        safe_print("[*] Validating environment...")
        env_validation = validate_environment()
        
        if not env_validation["valid"]:
            safe_print("[!] Environment validation failed:")
            for issue in env_validation["issues"]:
                safe_print(f"   - {issue}")
            return 1
        
        if env_validation["warnings"]:
            for warning in env_validation["warnings"]:
                safe_print(f"[!] {warning}")
        
        safe_print("[OK] Environment validation passed\n")
        logger.info("Environment validation passed",
                   issues=len(env_validation["issues"]),
                   warnings=len(env_validation["warnings"]))
        
        # Step 4: Initialize configuration
        safe_print("[*] Initializing configuration...")
        config = initialize_config()
        safe_print("[OK] Configuration loaded successfully\n")
        logger.info("Configuration initialized successfully")
        
        # Step 5: Perform health check
        if args.health_check or args.validate_config:
            safe_print("[*] Performing health check...")
            health = perform_health_check()
            
            safe_print(f"\nHealth Check Results:")
            safe_print(f"   Overall Status: {health['status'].upper()}")
            safe_print(f"\n   Components:")
            for component, status in health["components"].items():
                component_status = status.get("status", "unknown")
                status_icon = "[OK]" if component_status == "healthy" else "[!]" if component_status == "warning" else "[X]"
                safe_print(f"   {status_icon} {component}: {component_status}")
                if "message" in status:
                    safe_print(f"      -> {status['message']}")
                if "error" in status:
                    safe_print(f"      -> Error: {status['error']}")
            
            logger.info("Health check completed", status=health["status"])
            
            if args.validate_config or args.health_check:
                safe_print("\n[OK] Validation completed successfully!")
                return 0
        
        # Step 6: Print startup banner
        print_startup_banner(config)
        
        # Step 7: Create and launch the Gradio app
        safe_print("[*] Starting Intelligent Research Assistant...")
        logger.info("Creating Gradio application")
        
        app = create_gradio_app()
        
        safe_print(f"[*] Launching web interface...")
        host = args.host or config.gradio_host
        port = args.port or config.gradio_port
        
        safe_print(f"   - Host: {host}")
        safe_print(f"   - Port: {port}")
        
        if args.share:
            safe_print("   - Public link: Enabled")
            logger.info("Public link sharing enabled")
        
        startup_time = time.time() - start_time
        safe_print(f"\n[*] Startup completed in {startup_time:.2f} seconds")
        safe_print(f"[OK] Application ready! Access at http://{host}:{port}\n")
        
        logger.info("Application startup completed successfully",
                   startup_time=startup_time,
                   host=host,
                   port=port,
                   share=args.share)
        
        # Launch the interface
        app.launch(
            server_name=host,
            server_port=port,
            share=args.share,
            debug=args.debug,
            show_error=True
        )
        
        return 0
        
    except KeyboardInterrupt:
        safe_print("\n\n[*] Shutting down gracefully...")
        logger.info("Application shutdown requested by user")
        return 0
        
    except Exception as e:
        logger.error("Application startup failed", 
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True)
        safe_print(f"\n[!] Fatal Error: {str(e)}")
        safe_print("\nTroubleshooting:")
        safe_print("   1. Check your .env file configuration")
        safe_print("   2. Verify all API keys are correct")
        safe_print("   3. Ensure all dependencies are installed")
        safe_print("   4. Run with --validate-config to check setup")
        safe_print("   5. Run with --health-check for detailed diagnostics")
        safe_print("   6. Check logs for more details")
        
        if args.debug:
            import traceback
            safe_print("\nFull traceback:")
            traceback.print_exc()
        
        return 1

if __name__ == "__main__":
    sys.exit(main())