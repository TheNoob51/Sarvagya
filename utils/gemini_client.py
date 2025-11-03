"""
Gemini API client wrapper with retry logic and rate limiting.
Provides a robust interface to Google's Gemini API with error handling.
"""

import time
import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class GeminiConfig:
    """Configuration for Gemini API client"""
    api_key: str
    model_name: str = "gemini-2.5-flash"
    max_retries: int = 3
    base_delay: float = 1.0  # Base delay for exponential backoff
    max_delay: float = 60.0  # Maximum delay between retries
    rate_limit_per_minute: int = 60
    default_temperature: float = 0.7
    max_tokens: int = 16384  # Increased for comprehensive responses

class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_calls_per_minute: int):
        self.max_calls = max_calls_per_minute
        self.calls = []
        self.lock = asyncio.Lock() if asyncio.iscoroutinefunction(self.__init__) else None
    
    def can_make_call(self) -> bool:
        """Check if we can make a call without exceeding rate limit"""
        now = datetime.now()
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls 
                     if now - call_time < timedelta(minutes=1)]
        
        return len(self.calls) < self.max_calls
    
    def record_call(self):
        """Record that a call was made"""
        self.calls.append(datetime.now())
    
    def wait_time(self) -> float:
        """Get the time to wait before next call is allowed"""
        if self.can_make_call():
            return 0.0
        
        # Find the oldest call that's still within the minute window
        now = datetime.now()
        oldest_call = min(self.calls)
        wait_time = 60 - (now - oldest_call).total_seconds()
        return max(0, wait_time)

class GeminiClient:
    """
    Wrapper for Google Gemini API with retry logic, rate limiting, and error handling.
    
    Features:
    - Exponential backoff retry mechanism
    - Rate limiting to respect API quotas
    - Structured error handling and logging
    - Support for different model configurations
    - Safety settings configuration
    """
    
    def __init__(self, config: GeminiConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config.rate_limit_per_minute)
        
        # Configure the Gemini API
        genai.configure(api_key=config.api_key)
        
        # Initialize the model with safety settings
        self.model = genai.GenerativeModel(
            model_name=config.model_name,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )
        
        logger.info("GeminiClient initialized", 
                   model=config.model_name, 
                   rate_limit=config.rate_limit_per_minute)
    
    def _wait_for_rate_limit(self):
        """Wait if rate limit would be exceeded"""
        wait_time = self.rate_limiter.wait_time()
        if wait_time > 0:
            logger.info("Rate limit reached, waiting", wait_seconds=wait_time)
            time.sleep(wait_time)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff"""
        delay = self.config.base_delay * (2 ** attempt)
        return min(delay, self.config.max_delay)
    
    def generate_text(
        self, 
        prompt: str, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Generate text using Gemini API with retry logic.
        
        Args:
            prompt: The input prompt for text generation
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            system_instruction: System instruction for the model
            
        Returns:
            Generated text response
            
        Raises:
            Exception: If all retry attempts fail
        """
        temperature = temperature or self.config.default_temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        for attempt in range(self.config.max_retries):
            try:
                # Wait for rate limit
                self._wait_for_rate_limit()
                
                # Record the API call
                self.rate_limiter.record_call()
                
                # Prepare the model with system instruction if provided
                if system_instruction:
                    model = genai.GenerativeModel(
                        model_name=self.config.model_name,
                        system_instruction=system_instruction,
                        safety_settings=self.model._safety_settings
                    )
                else:
                    model = self.model
                
                # Make the API call
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                # Check if response was blocked
                if response.candidates[0].finish_reason.name == "SAFETY":
                    logger.warning("Response blocked by safety filters", prompt_preview=prompt[:100])
                    raise Exception("Response blocked by safety filters")
                
                result = response.text
                logger.info("Text generated successfully", 
                           prompt_length=len(prompt),
                           response_length=len(result),
                           attempt=attempt + 1)
                
                return result
                
            except Exception as e:
                logger.warning("API call failed", 
                             attempt=attempt + 1, 
                             max_retries=self.config.max_retries,
                             error=str(e))
                
                if attempt == self.config.max_retries - 1:
                    logger.error("All retry attempts failed", error=str(e))
                    raise Exception(f"Gemini API call failed after {self.config.max_retries} attempts: {str(e)}")
                
                # Wait before retrying
                delay = self._calculate_delay(attempt)
                logger.info("Retrying after delay", delay_seconds=delay)
                time.sleep(delay)
    
    def generate_json(
        self, 
        prompt: str, 
        temperature: Optional[float] = None,
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate JSON response using Gemini API.
        
        Args:
            prompt: The input prompt for JSON generation
            temperature: Controls randomness (0.0 to 1.0)
            system_instruction: System instruction for the model
            
        Returns:
            Parsed JSON response as dictionary
            
        Raises:
            Exception: If response is not valid JSON or API call fails
        """
        import json
        
        # Add JSON formatting instruction to prompt
        json_prompt = f"""{prompt}

Please respond with valid JSON only. Do not include any explanatory text before or after the JSON."""
        
        response_text = self.generate_text(
            json_prompt, 
            temperature=temperature or 0.3,  # Lower temperature for structured output
            system_instruction=system_instruction
        )
        
        try:
            # Clean the response - remove markdown code blocks if present
            cleaned_response = response_text.strip()
            
            # Remove markdown code blocks
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            elif cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]   # Remove ```
            
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # Remove trailing ```
            
            cleaned_response = cleaned_response.strip()
            
            # Try to parse the cleaned response as JSON
            result = json.loads(cleaned_response)
            logger.info("JSON generated and parsed successfully")
            return result
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON response", 
                        response=response_text[:500], 
                        error=str(e))
            
            # Try to fix incomplete JSON by adding closing braces
            try:
                # Count opening and closing braces
                open_braces = cleaned_response.count('{')
                close_braces = cleaned_response.count('}')
                open_brackets = cleaned_response.count('[')
                close_brackets = cleaned_response.count(']')
                
                # Add missing closing characters
                fixed_response = cleaned_response
                if open_braces > close_braces:
                    fixed_response += '}' * (open_braces - close_braces)
                if open_brackets > close_brackets:
                    fixed_response += ']' * (open_brackets - close_brackets)
                
                result = json.loads(fixed_response)
                logger.warning("JSON was incomplete but successfully fixed")
                return result
            except:
                # If fixing doesn't work, return a minimal valid response
                logger.error("Could not fix incomplete JSON, returning minimal response")
                return {
                    "error": "Incomplete JSON response from API",
                    "partial_response": cleaned_response[:200]
                }
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            Exception: If embedding generation fails
        """
        embeddings = []
        
        for i, text in enumerate(texts):
            for attempt in range(self.config.max_retries):
                try:
                    # Wait for rate limit
                    self._wait_for_rate_limit()
                    
                    # Record the API call
                    self.rate_limiter.record_call()
                    
                    # Generate embedding
                    result = genai.embed_content(
                        model="models/embedding-001",
                        content=text,
                        task_type="retrieval_document"
                    )
                    
                    embeddings.append(result['embedding'])
                    logger.debug("Embedding generated", text_index=i, text_length=len(text))
                    break
                    
                except Exception as e:
                    logger.warning("Embedding generation failed", 
                                 text_index=i,
                                 attempt=attempt + 1, 
                                 error=str(e))
                    
                    if attempt == self.config.max_retries - 1:
                        logger.error("All embedding attempts failed", 
                                   text_index=i, 
                                   error=str(e))
                        raise Exception(f"Embedding generation failed: {str(e)}")
                    
                    # Wait before retrying
                    delay = self._calculate_delay(attempt)
                    time.sleep(delay)
        
        logger.info("All embeddings generated successfully", count=len(embeddings))
        return embeddings
    
    def health_check(self) -> bool:
        """
        Perform a health check on the Gemini API connection.
        
        Returns:
            True if API is accessible, False otherwise
        """
        try:
            test_response = self.generate_text(
                "Say 'OK' if you can read this message.",
                temperature=0.0
            )
            
            is_healthy = "OK" in test_response.upper()
            logger.info("Health check completed", is_healthy=is_healthy)
            return is_healthy
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        try:
            model_info = genai.get_model(f"models/{self.config.model_name}")
            return {
                "name": model_info.name,
                "display_name": model_info.display_name,
                "description": model_info.description,
                "input_token_limit": model_info.input_token_limit,
                "output_token_limit": model_info.output_token_limit,
                "supported_generation_methods": model_info.supported_generation_methods,
            }
        except Exception as e:
            logger.error("Failed to get model info", error=str(e))
            return {"error": str(e)}