"""
Web Search Agent for performing web searches using multiple providers.
Implements SerpAPI integration with fallback to DuckDuckGo, including result filtering,
deduplication, and credibility scoring with rate limiting and caching mechanisms.
"""

import time
import hashlib
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import requests
from urllib.parse import urlparse, urljoin
import structlog
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

from agents.data_models import SearchResult
from utils.config import AppConfig

logger = structlog.get_logger(__name__)

@dataclass
class SearchCache:
    """Cache entry for search results"""
    results: List[SearchResult]
    timestamp: datetime
    ttl_hours: int = 24

class CredibilityScorer:
    """Utility class for calculating website credibility scores"""
    
    # Domain authority scores (1-10 scale)
    DOMAIN_SCORES = {
        # Academic and research institutions
        'edu': 9.0, 'ac.uk': 9.0, 'ac.in': 8.5, 'edu.au': 8.5,
        
        # Government sources
        'gov': 9.5, 'gov.uk': 9.5, 'europa.eu': 9.0, 'un.org': 9.0,
        
        # High-authority news and media
        'reuters.com': 8.5, 'bbc.com': 8.5, 'npr.org': 8.5,
        'apnews.com': 8.5, 'pbs.org': 8.0, 'cnn.com': 7.5,
        'nytimes.com': 8.0, 'washingtonpost.com': 8.0,
        
        # Scientific and technical sources
        'nature.com': 9.5, 'science.org': 9.5, 'ieee.org': 9.0,
        'arxiv.org': 8.5, 'pubmed.ncbi.nlm.nih.gov': 9.0,
        'scholar.google.com': 8.0, 'researchgate.net': 7.5,
        
        # Technology and business
        'stackoverflow.com': 8.0, 'github.com': 7.5, 'medium.com': 6.5,
        'techcrunch.com': 7.0, 'wired.com': 7.5, 'arstechnica.com': 7.5,
        
        # Wikipedia and reference
        'wikipedia.org': 7.5, 'britannica.com': 8.0,
        
        # Low credibility indicators
        'blogspot.com': 4.0, 'wordpress.com': 4.5, 'tumblr.com': 3.5,
        'facebook.com': 3.0, 'twitter.com': 3.5, 'reddit.com': 5.0
    }
    
    @classmethod
    def calculate_credibility_score(cls, url: str, title: str = "", snippet: str = "") -> float:
        """
        Calculate credibility score for a URL based on domain, title, and content indicators.
        
        Args:
            url: The URL to score
            title: Page title (optional)
            snippet: Page snippet/description (optional)
            
        Returns:
            Credibility score from 1.0 to 10.0
        """
        try:
            parsed_url = urlparse(url.lower())
            domain = parsed_url.netloc
            
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Start with base score
            score = 6.0
            
            # Check exact domain matches
            if domain in cls.DOMAIN_SCORES:
                score = cls.DOMAIN_SCORES[domain]
            else:
                # Check domain endings
                for domain_ending, domain_score in cls.DOMAIN_SCORES.items():
                    if domain.endswith(domain_ending):
                        score = domain_score
                        break
            
            # Adjust based on URL structure
            path = parsed_url.path.lower()
            
            # Positive indicators
            if any(indicator in path for indicator in ['/research/', '/study/', '/report/', '/paper/', '/article/']):
                score += 0.5
            
            if any(indicator in path for indicator in ['/news/', '/press/', '/announcement/']):
                score += 0.3
            
            # Negative indicators
            if any(indicator in path for indicator in ['/blog/', '/opinion/', '/editorial/', '/comment/']):
                score -= 0.3
            
            if any(indicator in url.lower() for indicator in ['ad=', 'utm_', 'affiliate', 'sponsored']):
                score -= 1.0
            
            # Check title and snippet for quality indicators
            combined_text = f"{title} {snippet}".lower()
            
            # Positive content indicators
            quality_indicators = [
                'research', 'study', 'analysis', 'report', 'findings',
                'peer-reviewed', 'published', 'journal', 'academic'
            ]
            quality_count = sum(1 for indicator in quality_indicators if indicator in combined_text)
            score += min(quality_count * 0.2, 1.0)
            
            # Negative content indicators
            spam_indicators = [
                'click here', 'amazing', 'incredible', 'shocking',
                'you won\'t believe', 'doctors hate', 'one weird trick'
            ]
            spam_count = sum(1 for indicator in spam_indicators if indicator in combined_text)
            score -= spam_count * 0.5
            
            # Ensure score is within bounds
            score = max(1.0, min(10.0, score))
            
            logger.debug("Credibility score calculated", 
                        url=url[:100], 
                        domain=domain, 
                        score=score)
            
            return score
            
        except Exception as e:
            logger.warning("Error calculating credibility score", url=url, error=str(e))
            return 5.0  # Default neutral score

class WebSearchAgent:
    """
    Web Search Agent that performs searches using multiple providers with intelligent fallback.
    
    Features:
    - SerpAPI integration with DuckDuckGo fallback
    - Result filtering and deduplication
    - Credibility scoring for sources
    - Rate limiting and caching
    - Error handling and retry logic
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize Web Search Agent with configuration.
        
        Args:
            config: Application configuration containing API keys and settings
        """
        self.config = config
        self.serpapi_key = config.api.serpapi_key
        self.cache: Dict[str, SearchCache] = {}
        self.last_serpapi_call = 0
        self.last_ddg_call = 0
        self.serpapi_rate_limit = 100  # calls per hour
        self.ddg_rate_limit = 30  # calls per minute
        
        # Initialize DuckDuckGo search
        self.ddgs = DDGS()
        
        logger.info("WebSearchAgent initialized", 
                   has_serpapi=bool(self.serpapi_key),
                   cache_ttl_hours=24)
    
    def _get_cache_key(self, query: str, provider: str) -> str:
        """Generate cache key for search query and provider"""
        return hashlib.md5(f"{query}:{provider}".encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: SearchCache) -> bool:
        """Check if cache entry is still valid"""
        age = datetime.now() - cache_entry.timestamp
        return age < timedelta(hours=cache_entry.ttl_hours)
    
    def _can_call_serpapi(self) -> bool:
        """Check if we can make a SerpAPI call without exceeding rate limits"""
        now = time.time()
        time_since_last = now - self.last_serpapi_call
        # Allow 1 call per 36 seconds (100 calls per hour)
        return time_since_last >= 36
    
    def _can_call_ddg(self) -> bool:
        """Check if we can make a DuckDuckGo call without exceeding rate limits"""
        now = time.time()
        time_since_last = now - self.last_ddg_call
        # Allow 1 call per 2 seconds (30 calls per minute)
        return time_since_last >= 2
    
    def _wait_for_rate_limit(self, provider: str) -> None:
        """Wait if necessary to respect rate limits"""
        if provider == "serpapi" and not self._can_call_serpapi():
            wait_time = 36 - (time.time() - self.last_serpapi_call)
            if wait_time > 0:
                logger.info("SerpAPI rate limit, waiting", wait_seconds=wait_time)
                time.sleep(wait_time)
        elif provider == "duckduckgo" and not self._can_call_ddg():
            wait_time = 2 - (time.time() - self.last_ddg_call)
            if wait_time > 0:
                logger.info("DuckDuckGo rate limit, waiting", wait_seconds=wait_time)
                time.sleep(wait_time)
    
    def search_serpapi(self, queries: List[str], max_results_per_query: int = 10) -> List[SearchResult]:
        """
        Search using SerpAPI with rate limiting and error handling.
        
        Args:
            queries: List of search queries
            max_results_per_query: Maximum results per query
            
        Returns:
            List of SearchResult objects
            
        Raises:
            Exception: If SerpAPI is not available or all queries fail
        """
        if not self.serpapi_key:
            raise Exception("SerpAPI key not configured")
        
        all_results = []
        
        for query in queries:
            try:
                # Check cache first
                cache_key = self._get_cache_key(query, "serpapi")
                if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
                    cached_results = self.cache[cache_key].results
                    all_results.extend(cached_results)
                    logger.info("Using cached SerpAPI results", query=query, count=len(cached_results))
                    continue
                
                # Wait for rate limit
                self._wait_for_rate_limit("serpapi")
                
                # Make SerpAPI request
                params = {
                    'q': query,
                    'api_key': self.serpapi_key,
                    'engine': 'google',
                    'num': max_results_per_query,
                    'gl': 'us',  # Country
                    'hl': 'en'   # Language
                }
                
                response = requests.get('https://serpapi.com/search', params=params, timeout=30)
                self.last_serpapi_call = time.time()
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Parse organic results
                    organic_results = data.get('organic_results', [])
                    query_results = []
                    
                    for result in organic_results:
                        title = result.get('title', '')
                        url = result.get('link', '')
                        snippet = result.get('snippet', '')
                        
                        if url and title:  # Basic validation
                            credibility_score = CredibilityScorer.calculate_credibility_score(
                                url, title, snippet
                            )
                            
                            search_result = SearchResult(
                                title=title,
                                url=url,
                                snippet=snippet,
                                credibility_score=credibility_score,
                                source="serpapi"
                            )
                            query_results.append(search_result)
                    
                    # Cache results
                    self.cache[cache_key] = SearchCache(
                        results=query_results,
                        timestamp=datetime.now()
                    )
                    
                    all_results.extend(query_results)
                    logger.info("SerpAPI search completed", 
                               query=query, 
                               results_count=len(query_results))
                
                else:
                    logger.warning("SerpAPI request failed", 
                                 query=query, 
                                 status_code=response.status_code,
                                 response=response.text[:200])
                    
            except Exception as e:
                logger.error("SerpAPI search error", query=query, error=str(e))
                # Continue with other queries rather than failing completely
                continue
        
        if not all_results:
            raise Exception("All SerpAPI queries failed")
        
        return all_results
    
    def search_duckduckgo(self, queries: List[str], max_results_per_query: int = 10) -> List[SearchResult]:
        """
        Search using DuckDuckGo with rate limiting and error handling.
        
        Args:
            queries: List of search queries
            max_results_per_query: Maximum results per query
            
        Returns:
            List of SearchResult objects
        """
        all_results = []
        
        for query in queries:
            try:
                # Check cache first
                cache_key = self._get_cache_key(query, "duckduckgo")
                if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
                    cached_results = self.cache[cache_key].results
                    all_results.extend(cached_results)
                    logger.info("Using cached DuckDuckGo results", query=query, count=len(cached_results))
                    continue
                
                # Wait for rate limit
                self._wait_for_rate_limit("duckduckgo")
                
                # Perform DuckDuckGo search
                results = self.ddgs.text(query, max_results=max_results_per_query)
                self.last_ddg_call = time.time()
                
                query_results = []
                for result in results:
                    title = result.get('title', '')
                    url = result.get('href', '')
                    snippet = result.get('body', '')
                    
                    if url and title:  # Basic validation
                        credibility_score = CredibilityScorer.calculate_credibility_score(
                            url, title, snippet
                        )
                        
                        search_result = SearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            credibility_score=credibility_score,
                            source="duckduckgo"
                        )
                        query_results.append(search_result)
                
                # Cache results
                self.cache[cache_key] = SearchCache(
                    results=query_results,
                    timestamp=datetime.now()
                )
                
                all_results.extend(query_results)
                logger.info("DuckDuckGo search completed", 
                           query=query, 
                           results_count=len(query_results))
                
            except Exception as e:
                logger.error("DuckDuckGo search error", query=query, error=str(e))
                # Continue with other queries rather than failing completely
                continue
        
        return all_results
    
    def _filter_results(self, results: List[SearchResult], min_credibility: float = 6.0) -> List[SearchResult]:
        """
        Filter search results based on credibility and quality criteria.
        
        Args:
            results: List of search results to filter
            min_credibility: Minimum credibility score to include
            
        Returns:
            Filtered list of search results
        """
        filtered_results = []
        
        for result in results:
            # Filter by credibility score
            if result.credibility_score < min_credibility:
                logger.debug("Filtered out low credibility result", 
                           url=result.url, 
                           score=result.credibility_score)
                continue
            
            # Filter out obvious ads and spam
            if any(spam_indicator in result.url.lower() for spam_indicator in [
                'ad=', 'utm_campaign', 'affiliate', 'sponsored', 'promo'
            ]):
                logger.debug("Filtered out promotional result", url=result.url)
                continue
            
            # Filter out social media posts (unless high credibility)
            social_domains = ['facebook.com', 'twitter.com', 'instagram.com', 'tiktok.com']
            if any(domain in result.url.lower() for domain in social_domains) and result.credibility_score < 7.0:
                logger.debug("Filtered out social media result", url=result.url)
                continue
            
            filtered_results.append(result)
        
        logger.info("Results filtered", 
                   original_count=len(results), 
                   filtered_count=len(filtered_results))
        
        return filtered_results
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Remove duplicate results based on URL and content similarity.
        
        Args:
            results: List of search results to deduplicate
            
        Returns:
            Deduplicated list of search results
        """
        seen_urls: Set[str] = set()
        seen_titles: Set[str] = set()
        deduplicated_results = []
        
        for result in results:
            # Normalize URL for comparison
            normalized_url = result.url.lower().rstrip('/')
            
            # Remove query parameters for comparison
            if '?' in normalized_url:
                normalized_url = normalized_url.split('?')[0]
            
            # Check for exact URL duplicates
            if normalized_url in seen_urls:
                logger.debug("Removed duplicate URL", url=result.url)
                continue
            
            # Check for very similar titles (exact match)
            normalized_title = result.title.lower().strip()
            if normalized_title in seen_titles:
                logger.debug("Removed duplicate title", title=result.title[:50])
                continue
            
            seen_urls.add(normalized_url)
            seen_titles.add(normalized_title)
            deduplicated_results.append(result)
        
        logger.info("Results deduplicated", 
                   original_count=len(results), 
                   deduplicated_count=len(deduplicated_results))
        
        return deduplicated_results
    
    def search(self, queries: List[str], max_results: int = 10) -> List[SearchResult]:
        """
        Perform web search using available providers with intelligent fallback.
        
        Args:
            queries: List of search queries
            max_results: Maximum total results to return
            
        Returns:
            List of SearchResult objects, filtered and deduplicated
            
        Raises:
            Exception: If all search providers fail
        """
        logger.info("Starting web search", queries=queries, max_results=max_results)
        
        all_results = []
        
        # Try SerpAPI first if available
        if self.serpapi_key:
            try:
                serpapi_results = self.search_serpapi(queries, max_results_per_query=5)
                all_results.extend(serpapi_results)
                logger.info("SerpAPI search successful", results_count=len(serpapi_results))
            except Exception as e:
                logger.warning("SerpAPI search failed, falling back to DuckDuckGo", error=str(e))
        
        # Use DuckDuckGo as fallback or primary
        if not all_results or len(all_results) < max_results // 2:
            try:
                ddg_results = self.search_duckduckgo(queries, max_results_per_query=5)
                all_results.extend(ddg_results)
                logger.info("DuckDuckGo search successful", results_count=len(ddg_results))
            except Exception as e:
                logger.error("DuckDuckGo search failed", error=str(e))
                if not all_results:
                    raise Exception("All search providers failed")
        
        if not all_results:
            raise Exception("No search results obtained from any provider")
        
        # Filter and deduplicate results
        filtered_results = self._filter_results(all_results)
        deduplicated_results = self._deduplicate_results(filtered_results)
        
        # Sort by credibility score (descending) and limit results
        sorted_results = sorted(deduplicated_results, 
                              key=lambda x: x.credibility_score, 
                              reverse=True)
        
        final_results = sorted_results[:max_results]
        
        logger.info("Web search completed", 
                   total_results=len(final_results),
                   avg_credibility=sum(r.credibility_score for r in final_results) / len(final_results) if final_results else 0)
        
        return final_results
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on web search capabilities.
        
        Returns:
            Dictionary with health check results
        """
        health_status = {
            "serpapi_available": bool(self.serpapi_key),
            "duckduckgo_available": True,  # Always available
            "cache_entries": len(self.cache),
            "last_serpapi_call": self.last_serpapi_call,
            "last_ddg_call": self.last_ddg_call
        }
        
        # Test search functionality
        try:
            test_results = self.search(["test query"], max_results=1)
            health_status["search_functional"] = len(test_results) > 0
        except Exception as e:
            health_status["search_functional"] = False
            health_status["search_error"] = str(e)
        
        logger.info("Web search health check completed", status=health_status)
        return health_status
    
    def clear_cache(self) -> None:
        """Clear the search results cache"""
        cache_size = len(self.cache)
        self.cache.clear()
        logger.info("Search cache cleared", previous_size=cache_size)


# Factory function for easy instantiation
def create_web_search_agent(config: AppConfig) -> WebSearchAgent:
    """
    Factory function to create a WebSearchAgent with proper configuration.
    
    Args:
        config: Application configuration
        
    Returns:
        Configured WebSearchAgent instance
    """
    return WebSearchAgent(config)