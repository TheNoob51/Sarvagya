"""
Web Scraper Agent for extracting content from web pages.
Implements BeautifulSoup scraping for static pages and Selenium support for JavaScript-heavy sites,
including content extraction, cleaning, and comprehensive error handling.
"""

import time
import asyncio
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser
import requests
from bs4 import BeautifulSoup, Comment
import structlog

# Selenium imports with error handling
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

from agents.data_models import ScrapedContent
from utils.config import AppConfig

logger = structlog.get_logger(__name__)

@dataclass
class ScrapeResult:
    """Result from a scraping attempt"""
    success: bool
    content: Optional[ScrapedContent] = None
    error: Optional[str] = None
    method_used: Optional[str] = None
    response_time: float = 0.0

class RobotsChecker:
    """Utility class for checking robots.txt compliance"""
    
    def __init__(self):
        self.robots_cache: Dict[str, RobotFileParser] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.cache_ttl_hours = 24
    
    def can_fetch(self, url: str, user_agent: str = "*") -> bool:
        """
        Check if URL can be fetched according to robots.txt
        
        Args:
            url: URL to check
            user_agent: User agent string to check for
            
        Returns:
            True if URL can be fetched, False otherwise
        """
        try:
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            robots_url = urljoin(base_url, "/robots.txt")
            
            # Check cache
            now = datetime.now()
            if (robots_url in self.robots_cache and 
                robots_url in self.cache_expiry and 
                now < self.cache_expiry[robots_url]):
                
                rp = self.robots_cache[robots_url]
                return rp.can_fetch(user_agent, url)
            
            # Fetch and parse robots.txt
            rp = RobotFileParser()
            rp.set_url(robots_url)
            
            try:
                rp.read()
                self.robots_cache[robots_url] = rp
                self.cache_expiry[robots_url] = now + timedelta(hours=self.cache_ttl_hours)
                
                return rp.can_fetch(user_agent, url)
            except Exception:
                # If robots.txt can't be fetched, assume allowed
                logger.debug("Could not fetch robots.txt, assuming allowed", robots_url=robots_url)
                return True
                
        except Exception as e:
            logger.warning("Error checking robots.txt", url=url, error=str(e))
            return True  # Default to allowed if check fails

class ContentExtractor:
    """Utility class for extracting and cleaning web page content"""
    
    # Common content selectors in order of preference
    CONTENT_SELECTORS = [
        'article',
        '[role="main"]',
        'main',
        '.content',
        '.post-content',
        '.entry-content',
        '.article-content',
        '.story-body',
        '.post-body',
        '#content',
        '#main-content',
        '.main-content'
    ]
    
    # Elements to remove (noise)
    NOISE_SELECTORS = [
        'script', 'style', 'nav', 'header', 'footer', 'aside',
        '.advertisement', '.ads', '.sidebar', '.menu', '.navigation',
        '.social-share', '.comments', '.related-posts', '.popup',
        '[class*="ad-"]', '[id*="ad-"]', '[class*="advertisement"]'
    ]
    
    @classmethod
    def extract_content(cls, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Extract main content from BeautifulSoup object
        
        Args:
            soup: BeautifulSoup parsed HTML
            url: Original URL for context
            
        Returns:
            Dictionary with extracted content and metadata
        """
        try:
            # Remove noise elements
            for selector in cls.NOISE_SELECTORS:
                for element in soup.select(selector):
                    element.decompose()
            
            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
            
            # Extract title
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            
            # Try h1 if title is empty or generic
            if not title or len(title) < 10:
                h1_tag = soup.find('h1')
                if h1_tag:
                    title = h1_tag.get_text().strip()
            
            # Extract author
            author = None
            author_selectors = [
                '[rel="author"]', '.author', '.byline', '[class*="author"]',
                '[itemprop="author"]', '.post-author', '.article-author'
            ]
            
            for selector in author_selectors:
                author_element = soup.select_one(selector)
                if author_element:
                    author = author_element.get_text().strip()
                    break
            
            # Extract publish date
            publish_date = None
            date_selectors = [
                '[datetime]', '[itemprop="datePublished"]', '[itemprop="dateCreated"]',
                '.publish-date', '.post-date', '.article-date', 'time'
            ]
            
            for selector in date_selectors:
                date_element = soup.select_one(selector)
                if date_element:
                    date_text = date_element.get('datetime') or date_element.get_text().strip()
                    if date_text:
                        try:
                            # Try to parse common date formats
                            from dateutil import parser
                            publish_date = parser.parse(date_text)
                            break
                        except:
                            continue
            
            # Extract main content
            content_text = ""
            
            # Try content selectors in order of preference
            for selector in cls.CONTENT_SELECTORS:
                content_elements = soup.select(selector)
                if content_elements:
                    # Use the largest content element
                    largest_element = max(content_elements, key=lambda x: len(x.get_text()))
                    content_text = largest_element.get_text(separator=' ', strip=True)
                    if len(content_text) > 200:  # Minimum content threshold
                        break
            
            # Fallback: extract from body if no content found
            if len(content_text) < 200:
                body = soup.find('body')
                if body:
                    content_text = body.get_text(separator=' ', strip=True)
            
            # Clean the content text
            content_text = cls._clean_text(content_text)
            
            return {
                'title': title,
                'content': content_text,
                'author': author,
                'publish_date': publish_date,
                'word_count': len(content_text.split()),
                'extraction_quality': cls._assess_quality(content_text, title)
            }
            
        except Exception as e:
            logger.error("Content extraction failed", url=url, error=str(e))
            return {
                'title': '',
                'content': '',
                'author': None,
                'publish_date': None,
                'word_count': 0,
                'extraction_quality': 'poor'
            }
    
    @classmethod
    def _clean_text(cls, text: str) -> str:
        """Clean extracted text content"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common noise patterns
        noise_patterns = [
            r'Cookie Policy.*?(?=\.|$)',
            r'Privacy Policy.*?(?=\.|$)',
            r'Terms of Service.*?(?=\.|$)',
            r'Subscribe to.*?(?=\.|$)',
            r'Follow us on.*?(?=\.|$)',
            r'Share this.*?(?=\.|$)',
            r'Advertisement.*?(?=\.|$)'
        ]
        
        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    @classmethod
    def _assess_quality(cls, content: str, title: str) -> str:
        """Assess the quality of extracted content"""
        if not content:
            return 'poor'
        
        word_count = len(content.split())
        
        # Quality indicators
        has_title = bool(title and len(title) > 5)
        sufficient_length = word_count >= 100
        good_length = word_count >= 300
        
        # Check for content quality indicators
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        if good_length and has_title and avg_sentence_length > 8:
            return 'excellent'
        elif sufficient_length and has_title:
            return 'good'
        elif sufficient_length:
            return 'fair'
        else:
            return 'poor'

class WebScraperAgent:
    """
    Web Scraper Agent that extracts content from web pages using multiple strategies.
    
    Features:
    - BeautifulSoup for static HTML pages
    - Selenium for JavaScript-heavy sites
    - Robots.txt compliance checking
    - Content extraction and cleaning
    - Rate limiting and error handling
    - Concurrent scraping with limits
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize Web Scraper Agent with configuration.
        
        Args:
            config: Application configuration containing scraping settings
        """
        self.config = config
        self.scraping_config = config.scraping
        self.robots_checker = RobotsChecker()
        self.session = requests.Session()
        
        # Configure session
        self.session.headers.update({
            'User-Agent': self.scraping_config.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Rate limiting
        self.last_scrape_time = 0
        self.min_delay = 1.0  # Minimum delay between requests
        
        # Selenium setup
        self.selenium_available = SELENIUM_AVAILABLE
        self._webdriver = None
        
        logger.info("WebScraperAgent initialized", 
                   selenium_available=self.selenium_available,
                   respect_robots=self.scraping_config.respect_robots_txt,
                   max_concurrent=self.scraping_config.max_concurrent_scrapes)
    
    def _wait_for_rate_limit(self):
        """Wait to respect rate limiting"""
        now = time.time()
        time_since_last = now - self.last_scrape_time
        
        if time_since_last < self.min_delay:
            wait_time = self.min_delay - time_since_last
            logger.debug("Rate limiting, waiting", wait_seconds=wait_time)
            time.sleep(wait_time)
        
        self.last_scrape_time = time.time()
    
    def _get_webdriver(self) -> webdriver.Chrome:
        """Get or create Selenium WebDriver instance"""
        if not self.selenium_available:
            raise Exception("Selenium not available")
        
        if self._webdriver is None:
            options = ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--disable-extensions')
            options.add_argument('--disable-plugins')
            options.add_argument('--disable-images')
            options.add_argument(f'--user-agent={self.scraping_config.user_agent}')
            
            try:
                self._webdriver = webdriver.Chrome(options=options)
                self._webdriver.set_page_load_timeout(self.scraping_config.scrape_timeout)
                logger.info("Selenium WebDriver initialized")
            except Exception as e:
                logger.error("Failed to initialize WebDriver", error=str(e))
                raise Exception(f"WebDriver initialization failed: {str(e)}")
        
        return self._webdriver
    
    def _close_webdriver(self):
        """Close Selenium WebDriver if open"""
        if self._webdriver:
            try:
                self._webdriver.quit()
                self._webdriver = None
                logger.debug("WebDriver closed")
            except Exception as e:
                logger.warning("Error closing WebDriver", error=str(e))
    
    def scrape_static_page(self, url: str) -> ScrapeResult:
        """
        Scrape static HTML page using BeautifulSoup.
        
        Args:
            url: URL to scrape
            
        Returns:
            ScrapeResult with scraped content or error information
        """
        start_time = time.time()
        
        try:
            # Check robots.txt if enabled
            if self.scraping_config.respect_robots_txt:
                if not self.robots_checker.can_fetch(url, self.scraping_config.user_agent):
                    return ScrapeResult(
                        success=False,
                        error="Blocked by robots.txt",
                        method_used="beautifulsoup"
                    )
            
            # Wait for rate limit
            self._wait_for_rate_limit()
            
            # Make request
            response = self.session.get(
                url, 
                timeout=self.scraping_config.scrape_timeout,
                allow_redirects=True
            )
            
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract content
            extracted = ContentExtractor.extract_content(soup, url)
            
            # Create ScrapedContent object
            scraped_content = ScrapedContent(
                url=url,
                title=extracted['title'],
                content=extracted['content'],
                author=extracted['author'],
                publish_date=extracted['publish_date'],
                extraction_method="beautifulsoup"
            )
            
            response_time = time.time() - start_time
            
            logger.info("Static page scraped successfully", 
                       url=url,
                       word_count=extracted['word_count'],
                       quality=extracted['extraction_quality'],
                       response_time=response_time)
            
            return ScrapeResult(
                success=True,
                content=scraped_content,
                method_used="beautifulsoup",
                response_time=response_time
            )
            
        except requests.exceptions.Timeout:
            return ScrapeResult(
                success=False,
                error="Request timeout",
                method_used="beautifulsoup",
                response_time=time.time() - start_time
            )
        except requests.exceptions.HTTPError as e:
            return ScrapeResult(
                success=False,
                error=f"HTTP error: {e.response.status_code}",
                method_used="beautifulsoup",
                response_time=time.time() - start_time
            )
        except Exception as e:
            logger.error("Static page scraping failed", url=url, error=str(e))
            return ScrapeResult(
                success=False,
                error=str(e),
                method_used="beautifulsoup",
                response_time=time.time() - start_time
            )
    
    def scrape_dynamic_page(self, url: str) -> ScrapeResult:
        """
        Scrape JavaScript-heavy page using Selenium.
        
        Args:
            url: URL to scrape
            
        Returns:
            ScrapeResult with scraped content or error information
        """
        if not self.selenium_available:
            return ScrapeResult(
                success=False,
                error="Selenium not available",
                method_used="selenium"
            )
        
        start_time = time.time()
        
        try:
            # Check robots.txt if enabled
            if self.scraping_config.respect_robots_txt:
                if not self.robots_checker.can_fetch(url, self.scraping_config.user_agent):
                    return ScrapeResult(
                        success=False,
                        error="Blocked by robots.txt",
                        method_used="selenium"
                    )
            
            # Wait for rate limit
            self._wait_for_rate_limit()
            
            # Get WebDriver
            driver = self._get_webdriver()
            
            # Load page
            driver.get(url)
            
            # Wait for page to load (basic wait)
            time.sleep(2)
            
            # Try to wait for content to load
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except TimeoutException:
                logger.warning("Page load timeout, proceeding with available content", url=url)
            
            # Get page source and parse
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Extract content
            extracted = ContentExtractor.extract_content(soup, url)
            
            # Create ScrapedContent object
            scraped_content = ScrapedContent(
                url=url,
                title=extracted['title'],
                content=extracted['content'],
                author=extracted['author'],
                publish_date=extracted['publish_date'],
                extraction_method="selenium"
            )
            
            response_time = time.time() - start_time
            
            logger.info("Dynamic page scraped successfully", 
                       url=url,
                       word_count=extracted['word_count'],
                       quality=extracted['extraction_quality'],
                       response_time=response_time)
            
            return ScrapeResult(
                success=True,
                content=scraped_content,
                method_used="selenium",
                response_time=response_time
            )
            
        except TimeoutException:
            return ScrapeResult(
                success=False,
                error="Page load timeout",
                method_used="selenium",
                response_time=time.time() - start_time
            )
        except WebDriverException as e:
            return ScrapeResult(
                success=False,
                error=f"WebDriver error: {str(e)}",
                method_used="selenium",
                response_time=time.time() - start_time
            )
        except Exception as e:
            logger.error("Dynamic page scraping failed", url=url, error=str(e))
            return ScrapeResult(
                success=False,
                error=str(e),
                method_used="selenium",
                response_time=time.time() - start_time
            )
    
    def scrape_page(self, url: str, prefer_selenium: bool = False) -> ScrapeResult:
        """
        Scrape a web page using the most appropriate method.
        
        Args:
            url: URL to scrape
            prefer_selenium: Whether to prefer Selenium over BeautifulSoup
            
        Returns:
            ScrapeResult with scraped content or error information
        """
        logger.info("Starting page scrape", url=url, prefer_selenium=prefer_selenium)
        
        # Determine scraping strategy
        if prefer_selenium and self.selenium_available:
            # Try Selenium first
            result = self.scrape_dynamic_page(url)
            if result.success:
                return result
            
            # Fallback to BeautifulSoup
            logger.info("Selenium failed, falling back to BeautifulSoup", url=url)
            return self.scrape_static_page(url)
        else:
            # Try BeautifulSoup first
            result = self.scrape_static_page(url)
            
            # If content is poor and Selenium is available, try Selenium
            if (result.success and result.content and 
                len(result.content.content.split()) < 100 and 
                self.selenium_available):
                
                logger.info("Poor content quality, trying Selenium", url=url)
                selenium_result = self.scrape_dynamic_page(url)
                
                # Use Selenium result if it's better
                if (selenium_result.success and selenium_result.content and
                    len(selenium_result.content.content.split()) > len(result.content.content.split())):
                    return selenium_result
            
            return result
    
    def scrape_multiple_pages(self, urls: List[str], prefer_selenium: bool = False) -> List[ScrapeResult]:
        """
        Scrape multiple pages with concurrent processing.
        
        Args:
            urls: List of URLs to scrape
            prefer_selenium: Whether to prefer Selenium over BeautifulSoup
            
        Returns:
            List of ScrapeResult objects
        """
        logger.info("Starting batch scraping", url_count=len(urls))
        
        results = []
        successful_scrapes = 0
        
        # Limit concurrent scrapes
        max_concurrent = min(len(urls), self.scraping_config.max_concurrent_scrapes)
        
        for i, url in enumerate(urls):
            try:
                result = self.scrape_page(url, prefer_selenium)
                results.append(result)
                
                if result.success:
                    successful_scrapes += 1
                
                # Progress logging
                if (i + 1) % 5 == 0 or i == len(urls) - 1:
                    logger.info("Batch scraping progress", 
                               completed=i + 1,
                               total=len(urls),
                               successful=successful_scrapes)
                
            except Exception as e:
                logger.error("Unexpected error during scraping", url=url, error=str(e))
                results.append(ScrapeResult(
                    success=False,
                    error=f"Unexpected error: {str(e)}",
                    method_used="unknown"
                ))
        
        logger.info("Batch scraping completed", 
                   total_urls=len(urls),
                   successful_scrapes=successful_scrapes,
                   success_rate=successful_scrapes / len(urls) if urls else 0)
        
        return results
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on web scraping capabilities.
        
        Returns:
            Dictionary with health check results
        """
        health_status = {
            "beautifulsoup_available": True,  # Always available
            "selenium_available": self.selenium_available,
            "robots_checking_enabled": self.scraping_config.respect_robots_txt,
            "max_concurrent_scrapes": self.scraping_config.max_concurrent_scrapes,
            "scrape_timeout": self.scraping_config.scrape_timeout
        }
        
        # Test scraping functionality
        try:
            # Test with a simple page
            test_result = self.scrape_page("https://httpbin.org/html")
            health_status["scraping_functional"] = test_result.success
            if not test_result.success:
                health_status["scraping_error"] = test_result.error
        except Exception as e:
            health_status["scraping_functional"] = False
            health_status["scraping_error"] = str(e)
        
        # Test Selenium if available
        if self.selenium_available:
            try:
                driver = self._get_webdriver()
                health_status["selenium_functional"] = True
            except Exception as e:
                health_status["selenium_functional"] = False
                health_status["selenium_error"] = str(e)
        
        logger.info("Web scraper health check completed", status=health_status)
        return health_status
    
    def cleanup(self):
        """Clean up resources"""
        self._close_webdriver()
        self.session.close()
        logger.info("WebScraperAgent cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass


# Factory function for easy instantiation
def create_web_scraper_agent(config: AppConfig) -> WebScraperAgent:
    """
    Factory function to create a WebScraperAgent with proper configuration.
    
    Args:
        config: Application configuration
        
    Returns:
        Configured WebScraperAgent instance
    """
    return WebScraperAgent(config)