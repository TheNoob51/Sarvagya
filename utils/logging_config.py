"""
Logging configuration for the Intelligent Research Assistant.
Sets up structured logging with appropriate levels, formatting, error handling, and performance monitoring.
"""

import logging
import sys
import time
import functools
from typing import Optional, Callable, Any, Dict
from pathlib import Path
from datetime import datetime
import structlog
import traceback

# Global performance metrics storage
_performance_metrics = {
    "function_calls": {},
    "errors": {},
    "warnings": []
}


def setup_logging(log_level: str = "INFO", debug: bool = False, log_file: Optional[str] = None) -> None:
    """
    Configure logging for the application with comprehensive error handling.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        debug: Whether to enable debug mode with more verbose output
        log_file: Optional file path to write logs to
    """
    
    # Create handlers list
    handlers = [logging.StreamHandler(sys.stdout)]
    
    # Add file handler if log_file is specified
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            handlers.append(file_handler)
        except Exception as e:
            print(f"Warning: Could not create log file {log_file}: {e}")
    
    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Configure structlog with enhanced processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Add appropriate renderer based on debug mode
    if debug:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        processors.append(structlog.processors.JSONRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Log the logging configuration
    logger = structlog.get_logger(__name__)
    logger.info("Logging configured successfully",
               log_level=log_level,
               debug_mode=debug,
               log_file=log_file if log_file else "console_only")


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def silence_noisy_loggers():
    """Reduce log noise from third-party libraries"""
    noisy_loggers = [
        "urllib3",
        "selenium",
        "chromadb",
        "httpx",
        "httpcore",
        "gradio",
        "httpcore.http11",
        "httpcore.connection",
        "asyncio",
        "google.auth",
        "google.auth.transport",
        "googleapiclient.discovery",
        "googleapiclient.discovery_cache"
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def log_exception(logger: structlog.BoundLogger, exception: Exception, context: Optional[Dict[str, Any]] = None):
    """
    Log an exception with full context and traceback.
    
    Args:
        logger: Logger instance to use
        exception: Exception to log
        context: Optional additional context dictionary
    """
    error_context = {
        "error_type": type(exception).__name__,
        "error_message": str(exception),
        "traceback": traceback.format_exc()
    }
    
    if context:
        error_context.update(context)
    
    logger.error("Exception occurred", **error_context)
    
    # Track error in metrics
    error_type = type(exception).__name__
    if error_type not in _performance_metrics["errors"]:
        _performance_metrics["errors"][error_type] = 0
    _performance_metrics["errors"][error_type] += 1


def log_performance(func: Callable) -> Callable:
    """
    Decorator to log function performance metrics.
    
    Args:
        func: Function to decorate
        
    Returns:
        Wrapped function with performance logging
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = structlog.get_logger(func.__module__)
        func_name = f"{func.__module__}.{func.__name__}"
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log successful execution
            logger.debug("Function executed successfully",
                        function=func_name,
                        execution_time=execution_time)
            
            # Track performance metrics
            if func_name not in _performance_metrics["function_calls"]:
                _performance_metrics["function_calls"][func_name] = {
                    "count": 0,
                    "total_time": 0.0,
                    "min_time": float('inf'),
                    "max_time": 0.0
                }
            
            metrics = _performance_metrics["function_calls"][func_name]
            metrics["count"] += 1
            metrics["total_time"] += execution_time
            metrics["min_time"] = min(metrics["min_time"], execution_time)
            metrics["max_time"] = max(metrics["max_time"], execution_time)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log failed execution
            log_exception(logger, e, {
                "function": func_name,
                "execution_time": execution_time,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            })
            
            raise
    
    return wrapper


def log_async_performance(func: Callable) -> Callable:
    """
    Decorator to log async function performance metrics.
    
    Args:
        func: Async function to decorate
        
    Returns:
        Wrapped async function with performance logging
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger = structlog.get_logger(func.__module__)
        func_name = f"{func.__module__}.{func.__name__}"
        
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log successful execution
            logger.debug("Async function executed successfully",
                        function=func_name,
                        execution_time=execution_time)
            
            # Track performance metrics
            if func_name not in _performance_metrics["function_calls"]:
                _performance_metrics["function_calls"][func_name] = {
                    "count": 0,
                    "total_time": 0.0,
                    "min_time": float('inf'),
                    "max_time": 0.0
                }
            
            metrics = _performance_metrics["function_calls"][func_name]
            metrics["count"] += 1
            metrics["total_time"] += execution_time
            metrics["min_time"] = min(metrics["min_time"], execution_time)
            metrics["max_time"] = max(metrics["max_time"], execution_time)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log failed execution
            log_exception(logger, e, {
                "function": func_name,
                "execution_time": execution_time,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            })
            
            raise
    
    return wrapper


def get_performance_metrics() -> Dict[str, Any]:
    """
    Get collected performance metrics.
    
    Returns:
        Dictionary with performance metrics
    """
    metrics = {
        "function_calls": {},
        "errors": _performance_metrics["errors"].copy(),
        "warnings": _performance_metrics["warnings"].copy(),
        "summary": {
            "total_function_calls": 0,
            "total_errors": sum(_performance_metrics["errors"].values()),
            "total_warnings": len(_performance_metrics["warnings"])
        }
    }
    
    # Calculate average execution times
    for func_name, func_metrics in _performance_metrics["function_calls"].items():
        if func_metrics["count"] > 0:
            avg_time = func_metrics["total_time"] / func_metrics["count"]
            metrics["function_calls"][func_name] = {
                "count": func_metrics["count"],
                "avg_time": avg_time,
                "min_time": func_metrics["min_time"],
                "max_time": func_metrics["max_time"],
                "total_time": func_metrics["total_time"]
            }
            metrics["summary"]["total_function_calls"] += func_metrics["count"]
    
    return metrics


def reset_performance_metrics():
    """Reset all collected performance metrics"""
    global _performance_metrics
    _performance_metrics = {
        "function_calls": {},
        "errors": {},
        "warnings": []
    }


def log_warning(logger: structlog.BoundLogger, message: str, context: Optional[Dict[str, Any]] = None):
    """
    Log a warning with context tracking.
    
    Args:
        logger: Logger instance to use
        message: Warning message
        context: Optional additional context dictionary
    """
    warning_entry = {
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "context": context or {}
    }
    
    _performance_metrics["warnings"].append(warning_entry)
    
    if context:
        logger.warning(message, **context)
    else:
        logger.warning(message)


class ErrorHandler:
    """
    Context manager for comprehensive error handling with logging.
    """
    
    def __init__(self, logger: structlog.BoundLogger, operation: str, 
                 reraise: bool = True, default_return: Any = None):
        """
        Initialize error handler.
        
        Args:
            logger: Logger instance to use
            operation: Description of the operation being performed
            reraise: Whether to reraise exceptions after logging
            default_return: Default value to return if exception occurs and reraise=False
        """
        self.logger = logger
        self.operation = operation
        self.reraise = reraise
        self.default_return = default_return
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(f"Starting operation: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        
        if exc_type is None:
            # Success
            self.logger.debug(f"Operation completed successfully: {self.operation}",
                            execution_time=execution_time)
            return False
        
        # Error occurred
        log_exception(self.logger, exc_val, {
            "operation": self.operation,
            "execution_time": execution_time
        })
        
        if self.reraise:
            return False  # Reraise the exception
        else:
            self.logger.warning(f"Suppressing exception for operation: {self.operation}",
                              returning_default=self.default_return)
            return True  # Suppress the exception


def configure_error_recovery(logger: structlog.BoundLogger, max_retries: int = 3, 
                             backoff_factor: float = 2.0) -> Callable:
    """
    Create a decorator for automatic error recovery with exponential backoff.
    
    Args:
        logger: Logger instance to use
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for backoff delay
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = backoff_factor ** attempt
                        logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed, retrying in {delay}s",
                                     function=func.__name__,
                                     error=str(e),
                                     attempt=attempt + 1)
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed",
                                   function=func.__name__,
                                   error=str(e))
            
            # All retries exhausted
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator