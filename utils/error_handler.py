"""
Comprehensive error handling utilities for the Intelligent Research Assistant.
Provides centralized error handling, recovery strategies, and error reporting.
"""

import sys
import time
import functools
from typing import Optional, Callable, Any, Dict, Type, Tuple
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"  # Minor issues, can continue
    MEDIUM = "medium"  # Significant issues, degraded functionality
    HIGH = "high"  # Critical issues, major functionality affected
    CRITICAL = "critical"  # Fatal issues, cannot continue


class ErrorCategory(Enum):
    """Error categories for classification"""
    API_ERROR = "api_error"  # External API failures
    NETWORK_ERROR = "network_error"  # Network connectivity issues
    CONFIGURATION_ERROR = "configuration_error"  # Configuration problems
    DATA_ERROR = "data_error"  # Data validation or processing errors
    TIMEOUT_ERROR = "timeout_error"  # Operation timeouts
    RESOURCE_ERROR = "resource_error"  # Resource exhaustion (memory, disk, etc.)
    AUTHENTICATION_ERROR = "authentication_error"  # Auth failures
    VALIDATION_ERROR = "validation_error"  # Input validation errors
    UNKNOWN_ERROR = "unknown_error"  # Unclassified errors


class ApplicationError(Exception):
    """
    Base exception class for application-specific errors.
    """
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 context: Optional[Dict[str, Any]] = None,
                 original_exception: Optional[Exception] = None):
        """
        Initialize application error.
        
        Args:
            message: Error message
            category: Error category
            severity: Error severity
            context: Additional context information
            original_exception: Original exception if this wraps another exception
        """
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.original_exception = original_exception
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization"""
        return {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context,
            "timestamp": self.timestamp,
            "original_error": str(self.original_exception) if self.original_exception else None
        }


class APIError(ApplicationError):
    """Error related to external API calls"""
    def __init__(self, message: str, api_name: str, **kwargs):
        super().__init__(message, category=ErrorCategory.API_ERROR, **kwargs)
        self.context["api_name"] = api_name


class NetworkError(ApplicationError):
    """Error related to network connectivity"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.NETWORK_ERROR, **kwargs)


class ConfigurationError(ApplicationError):
    """Error related to configuration"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.CONFIGURATION_ERROR, 
                        severity=ErrorSeverity.HIGH, **kwargs)


class DataValidationError(ApplicationError):
    """Error related to data validation"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.VALIDATION_ERROR, **kwargs)


class TimeoutError(ApplicationError):
    """Error related to operation timeouts"""
    def __init__(self, message: str, operation: str, timeout_seconds: float, **kwargs):
        super().__init__(message, category=ErrorCategory.TIMEOUT_ERROR, **kwargs)
        self.context["operation"] = operation
        self.context["timeout_seconds"] = timeout_seconds


def classify_exception(exception: Exception) -> Tuple[ErrorCategory, ErrorSeverity]:
    """
    Classify an exception into category and severity.
    
    Args:
        exception: Exception to classify
        
    Returns:
        Tuple of (ErrorCategory, ErrorSeverity)
    """
    exception_type = type(exception).__name__
    exception_msg = str(exception).lower()
    
    # API-related errors
    if any(keyword in exception_msg for keyword in ["api", "quota", "rate limit", "authentication"]):
        if "authentication" in exception_msg or "unauthorized" in exception_msg:
            return ErrorCategory.AUTHENTICATION_ERROR, ErrorSeverity.HIGH
        return ErrorCategory.API_ERROR, ErrorSeverity.MEDIUM
    
    # Network errors
    if any(keyword in exception_msg for keyword in ["connection", "network", "timeout", "unreachable"]):
        if "timeout" in exception_msg:
            return ErrorCategory.TIMEOUT_ERROR, ErrorSeverity.MEDIUM
        return ErrorCategory.NETWORK_ERROR, ErrorSeverity.MEDIUM
    
    # Configuration errors
    if any(keyword in exception_msg for keyword in ["config", "missing", "not found", "invalid key"]):
        return ErrorCategory.CONFIGURATION_ERROR, ErrorSeverity.HIGH
    
    # Data errors
    if any(keyword in exception_msg for keyword in ["validation", "invalid data", "parse", "decode"]):
        return ErrorCategory.DATA_ERROR, ErrorSeverity.LOW
    
    # Resource errors
    if any(keyword in exception_msg for keyword in ["memory", "disk", "resource"]):
        return ErrorCategory.RESOURCE_ERROR, ErrorSeverity.HIGH
    
    # Default classification
    return ErrorCategory.UNKNOWN_ERROR, ErrorSeverity.MEDIUM


def handle_error(exception: Exception, context: Optional[Dict[str, Any]] = None,
                reraise: bool = True, default_return: Any = None) -> Any:
    """
    Centralized error handling function.
    
    Args:
        exception: Exception to handle
        context: Additional context information
        reraise: Whether to reraise the exception
        default_return: Default value to return if not reraising
        
    Returns:
        default_return if reraise=False, otherwise raises exception
    """
    category, severity = classify_exception(exception)
    
    error_context = {
        "error_type": type(exception).__name__,
        "error_message": str(exception),
        "category": category.value,
        "severity": severity.value
    }
    
    if context:
        error_context.update(context)
    
    # Log based on severity
    if severity == ErrorSeverity.CRITICAL:
        logger.critical("Critical error occurred", **error_context, exc_info=True)
    elif severity == ErrorSeverity.HIGH:
        logger.error("High severity error occurred", **error_context, exc_info=True)
    elif severity == ErrorSeverity.MEDIUM:
        logger.warning("Medium severity error occurred", **error_context)
    else:
        logger.info("Low severity error occurred", **error_context)
    
    if reraise:
        raise
    else:
        logger.info("Suppressing error and returning default value", 
                   default_return=default_return)
        return default_return


def with_error_handling(category: Optional[ErrorCategory] = None,
                       severity: Optional[ErrorSeverity] = None,
                       reraise: bool = True,
                       default_return: Any = None,
                       log_args: bool = False) -> Callable:
    """
    Decorator for automatic error handling.
    
    Args:
        category: Optional error category override
        severity: Optional severity override
        reraise: Whether to reraise exceptions
        default_return: Default value to return on error if not reraising
        log_args: Whether to log function arguments
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            context = {
                "function": f"{func.__module__}.{func.__name__}"
            }
            
            if log_args:
                context["args_count"] = len(args)
                context["kwargs_keys"] = list(kwargs.keys())
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Use provided category/severity or classify
                error_category = category if category else classify_exception(e)[0]
                error_severity = severity if severity else classify_exception(e)[1]
                
                context["category"] = error_category.value
                context["severity"] = error_severity.value
                
                return handle_error(e, context, reraise, default_return)
        
        return wrapper
    return decorator


def with_retry(max_attempts: int = 3, 
              backoff_factor: float = 2.0,
              exceptions: Tuple[Type[Exception], ...] = (Exception,),
              on_retry: Optional[Callable[[Exception, int], None]] = None) -> Callable:
    """
    Decorator for automatic retry with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        backoff_factor: Multiplier for backoff delay
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback function called on each retry
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        delay = backoff_factor ** attempt
                        
                        logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed, retrying in {delay}s",
                                     function=func.__name__,
                                     error=str(e),
                                     attempt=attempt + 1,
                                     max_attempts=max_attempts)
                        
                        if on_retry:
                            on_retry(e, attempt + 1)
                        
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed",
                                   function=func.__name__,
                                   error=str(e))
            
            # All retries exhausted
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


def with_timeout(timeout_seconds: float, 
                operation_name: Optional[str] = None) -> Callable:
    """
    Decorator to enforce timeout on function execution.
    Note: This is a simple implementation. For production, consider using
    threading.Timer or asyncio.wait_for for async functions.
    
    Args:
        timeout_seconds: Timeout in seconds
        operation_name: Optional operation name for logging
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                execution_time = time.time() - start_time
                if execution_time > timeout_seconds:
                    logger.warning("Function exceeded timeout but completed",
                                 function=func.__name__,
                                 timeout=timeout_seconds,
                                 actual_time=execution_time)
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                if execution_time >= timeout_seconds:
                    raise TimeoutError(
                        f"Operation '{operation_name or func.__name__}' timed out after {timeout_seconds}s",
                        operation=operation_name or func.__name__,
                        timeout_seconds=timeout_seconds,
                        original_exception=e
                    )
                raise
        
        return wrapper
    return decorator


class ErrorRecoveryStrategy:
    """
    Strategy pattern for error recovery.
    """
    
    @staticmethod
    def retry_with_backoff(func: Callable, max_attempts: int = 3, 
                          backoff_factor: float = 2.0) -> Any:
        """
        Retry a function with exponential backoff.
        
        Args:
            func: Function to retry
            max_attempts: Maximum number of attempts
            backoff_factor: Multiplier for backoff delay
            
        Returns:
            Function result
        """
        for attempt in range(max_attempts):
            try:
                return func()
            except Exception as e:
                if attempt < max_attempts - 1:
                    delay = backoff_factor ** attempt
                    logger.warning(f"Retry attempt {attempt + 1}/{max_attempts}",
                                 error=str(e), delay=delay)
                    time.sleep(delay)
                else:
                    raise
    
    @staticmethod
    def fallback(primary_func: Callable, fallback_func: Callable,
                exceptions: Tuple[Type[Exception], ...] = (Exception,)) -> Any:
        """
        Try primary function, fall back to fallback function on error.
        
        Args:
            primary_func: Primary function to try
            fallback_func: Fallback function to use on error
            exceptions: Tuple of exception types to catch
            
        Returns:
            Result from primary or fallback function
        """
        try:
            return primary_func()
        except exceptions as e:
            logger.warning("Primary function failed, using fallback",
                         error=str(e))
            return fallback_func()
    
    @staticmethod
    def circuit_breaker(func: Callable, failure_threshold: int = 5,
                       timeout: float = 60.0) -> Any:
        """
        Simple circuit breaker pattern implementation.
        
        Args:
            func: Function to protect with circuit breaker
            failure_threshold: Number of failures before opening circuit
            timeout: Time to wait before attempting to close circuit
            
        Returns:
            Function result
        """
        # This is a simplified implementation
        # In production, use a proper circuit breaker library
        if not hasattr(ErrorRecoveryStrategy, '_circuit_state'):
            ErrorRecoveryStrategy._circuit_state = {
                'failures': 0,
                'last_failure_time': 0,
                'state': 'closed'  # closed, open, half-open
            }
        
        state = ErrorRecoveryStrategy._circuit_state
        
        # Check if circuit is open
        if state['state'] == 'open':
            if time.time() - state['last_failure_time'] > timeout:
                state['state'] = 'half-open'
                logger.info("Circuit breaker entering half-open state")
            else:
                raise ApplicationError(
                    "Circuit breaker is open",
                    category=ErrorCategory.RESOURCE_ERROR,
                    severity=ErrorSeverity.HIGH
                )
        
        try:
            result = func()
            
            # Success - reset or close circuit
            if state['state'] == 'half-open':
                state['state'] = 'closed'
                state['failures'] = 0
                logger.info("Circuit breaker closed")
            
            return result
            
        except Exception as e:
            state['failures'] += 1
            state['last_failure_time'] = time.time()
            
            if state['failures'] >= failure_threshold:
                state['state'] = 'open'
                logger.error("Circuit breaker opened",
                           failures=state['failures'],
                           threshold=failure_threshold)
            
            raise


def create_error_report(exception: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a comprehensive error report.
    
    Args:
        exception: Exception to report
        context: Additional context information
        
    Returns:
        Dictionary with error report
    """
    import traceback
    
    category, severity = classify_exception(exception)
    
    report = {
        "error_type": type(exception).__name__,
        "error_message": str(exception),
        "category": category.value,
        "severity": severity.value,
        "timestamp": time.time(),
        "traceback": traceback.format_exc(),
        "context": context or {},
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform
        }
    }
    
    return report
