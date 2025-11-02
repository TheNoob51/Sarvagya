"""
Integration tests for Web Search Agent and Web Scraper Agent.
Tests web search with mock API responses and validates scraping with sample HTML pages.
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import List, Dict, Any

from agents.web_search_agent import WebSearchAgent, CredibilityScorer
from agents.web_scraper_agent import WebScraperAgent, ContentExtractor, ScrapeResult
from agents.data_models import SearchResult, ScrapedContent
from utils.config import AppConfig, APIConfig, ScrapingConfig, DatabaseConfig, ResearchConfig


class TestWebSearchAgentIntegration:
    """Integration tests for WebSearchAgent with mock API responses"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        api_config = APIConfig()
        api_config.serpapi_key = "test_serpapi_key"
        api_config.gemini_api_key = "test_gemini_key"
        
        scraping_config = ScrapingConfig()
        database_config = DatabaseConfig()
        research_config = ResearchConfig()
        
        config = AppConfig(
            api=api_config,
            database=database_config,
            scraping=scraping_config,
            research=research_config
        )
        return config
    
    @pytest.fixture
    def web_search_agent(self, mock_config):
        """Create WebSearchAgent instance for testing"""
        return WebSearchAgent(mock_config)
    
    @pytest.fixture
    def mock_serpapi_response(self):
        """Mock SerpAPI response data"""
        return {
            "organic_results": [
                {
                    "title": "Test Article 1",
                    "link": "https://example.com/article1",
                    "snippet": "This is a test article about research methodology."
                },
                {
                    "title": "Academic Paper on AI",
                    "link": "https://university.edu/paper",
                    "snippet": "Peer-reviewed research on artificial intelligence applications."
                },
                {
                    "title": "News Article",
                    "link": "https://news.com/story",
                    "snippet": "Breaking news about technology developments."
                }
            ]
        }
    
    @pytest.fixture
    def mock_duckduckgo_results(self):
        """Mock DuckDuckGo search results"""
        return [
            {
                "title": "DuckDuckGo Result 1",
                "href": "https://example.org/result1",
                "body": "First result from DuckDuckGo search engine."
            },
            {
                "title": "DuckDuckGo Result 2", 
                "href": "https://wikipedia.org/article",
                "body": "Wikipedia article with reliable information."
            }
        ]
    
    def test_serpapi_search_success(self, web_search_agent, mock_serpapi_response):
        """Test successful SerpAPI search with mock response"""
        with patch('requests.get') as mock_get:
            # Mock successful API response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_serpapi_response
            mock_get.return_value = mock_response
            
            # Execute search
            results = web_search_agent.search_serpapi(["test query"])
            
            # Verify results
            assert len(results) == 3
            assert all(isinstance(result, SearchResult) for result in results)
            assert results[0].title == "Test Article 1"
            assert results[0].url == "https://example.com/article1"
            assert results[0].source == "serpapi"
            assert results[0].credibility_score > 0
            
            # Verify API was called correctly
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert "serpapi.com" in call_args[0][0]
            assert call_args[1]["params"]["q"] == "test query"
    
    def test_serpapi_search_failure_fallback(self, web_search_agent):
        """Test SerpAPI failure handling"""
        with patch('requests.get') as mock_get:
            # Mock API failure
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_get.return_value = mock_response
            
            # Should raise exception when all queries fail
            with pytest.raises(Exception, match="All SerpAPI queries failed"):
                web_search_agent.search_serpapi(["test query"])
    
    def test_duckduckgo_search_success(self, web_search_agent, mock_duckduckgo_results):
        """Test successful DuckDuckGo search with mock results"""
        with patch.object(web_search_agent.ddgs, 'text') as mock_ddg_search:
            mock_ddg_search.return_value = mock_duckduckgo_results
            
            # Execute search
            results = web_search_agent.search_duckduckgo(["test query"])
            
            # Verify results
            assert len(results) == 2
            assert all(isinstance(result, SearchResult) for result in results)
            assert results[0].title == "DuckDuckGo Result 1"
            assert results[0].url == "https://example.org/result1"
            assert results[0].source == "duckduckgo"
            assert results[1].url == "https://wikipedia.org/article"
            
            # Wikipedia should have higher credibility
            assert results[1].credibility_score > results[0].credibility_score
    
    def test_search_with_fallback(self, web_search_agent, mock_duckduckgo_results):
        """Test search method with SerpAPI failure and DuckDuckGo fallback"""
        with patch('requests.get') as mock_serpapi, \
             patch.object(web_search_agent.ddgs, 'text') as mock_ddg:
            
            # Mock SerpAPI failure
            mock_serpapi.side_effect = Exception("SerpAPI unavailable")
            
            # Mock DuckDuckGo success
            mock_ddg.return_value = mock_duckduckgo_results
            
            # Execute search
            results = web_search_agent.search(["test query"], max_results=5)
            
            # Should get DuckDuckGo results
            assert len(results) == 2
            assert all(result.source == "duckduckgo" for result in results)
    
    def test_credibility_scoring(self):
        """Test credibility scoring for different URL types"""
        # Test high-credibility domains
        edu_score = CredibilityScorer.calculate_credibility_score(
            "https://university.edu/research/paper",
            "Academic Research Paper",
            "Peer-reviewed study on machine learning"
        )
        assert edu_score >= 8.5
        
        # Test government domains
        gov_score = CredibilityScorer.calculate_credibility_score(
            "https://agency.gov/report",
            "Government Report",
            "Official government analysis"
        )
        assert gov_score >= 9.0
        
        # Test low-credibility domains
        blog_score = CredibilityScorer.calculate_credibility_score(
            "https://myblog.blogspot.com/post",
            "Personal Opinion",
            "My thoughts on this topic"
        )
        assert blog_score <= 5.0
        
        # Test spam indicators
        spam_score = CredibilityScorer.calculate_credibility_score(
            "https://example.com/article?utm_campaign=spam",
            "You won't believe this amazing trick!",
            "Click here for incredible results"
        )
        assert spam_score <= 4.0
    
    def test_result_filtering(self, web_search_agent):
        """Test result filtering based on credibility"""
        # Create test results with different credibility scores
        test_results = [
            SearchResult("High Quality", "https://university.edu/paper", "Research", 8.5, "test"),
            SearchResult("Medium Quality", "https://news.com/article", "News", 7.0, "test"),
            SearchResult("Low Quality", "https://spam.com/ad", "Advertisement", 3.0, "test"),
            SearchResult("Filtered Out", "https://example.com?ad=true", "Spam", 2.0, "test")
        ]
        
        filtered = web_search_agent._filter_results(test_results, min_credibility=6.0)
        
        # Should only keep high and medium quality results
        assert len(filtered) == 2
        assert all(result.credibility_score >= 6.0 for result in filtered)
        assert filtered[0].title == "High Quality"
        assert filtered[1].title == "Medium Quality"
    
    def test_result_deduplication(self, web_search_agent):
        """Test result deduplication"""
        # Create test results with duplicates
        test_results = [
            SearchResult("Article 1", "https://example.com/article", "Content", 7.0, "test"),
            SearchResult("Article 1", "https://example.com/article?utm=123", "Content", 7.0, "test"),  # Duplicate URL
            SearchResult("Article 2", "https://different.com/page", "Content", 7.0, "test"),
            SearchResult("Article 1", "https://another.com/page", "Content", 7.0, "test")  # Duplicate title
        ]
        
        deduplicated = web_search_agent._deduplicate_results(test_results)
        
        # Should remove duplicates
        assert len(deduplicated) == 2
        assert deduplicated[0].title == "Article 1"
        assert deduplicated[1].title == "Article 2"
    
    def test_caching_mechanism(self, web_search_agent, mock_serpapi_response):
        """Test search result caching"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_serpapi_response
            mock_get.return_value = mock_response
            
            # First search - should call API
            results1 = web_search_agent.search_serpapi(["cached query"])
            assert mock_get.call_count == 1
            
            # Second search with same query - should use cache
            results2 = web_search_agent.search_serpapi(["cached query"])
            assert mock_get.call_count == 1  # No additional API call
            
            # Results should be identical
            assert len(results1) == len(results2)
            assert results1[0].title == results2[0].title
    
    def test_rate_limiting(self, web_search_agent):
        """Test rate limiting functionality"""
        # Test SerpAPI rate limiting
        web_search_agent.last_serpapi_call = time.time()
        assert not web_search_agent._can_call_serpapi()
        
        # Test DuckDuckGo rate limiting
        web_search_agent.last_ddg_call = time.time()
        assert not web_search_agent._can_call_ddg()
        
        # Test after sufficient time has passed
        web_search_agent.last_serpapi_call = time.time() - 40
        assert web_search_agent._can_call_serpapi()


class TestWebScraperAgentIntegration:
    """Integration tests for WebScraperAgent with sample HTML pages"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        api_config = APIConfig()
        scraping_config = ScrapingConfig()
        scraping_config.respect_robots_txt = False  # Disable for testing
        scraping_config.scrape_timeout = 10
        scraping_config.max_concurrent_scrapes = 3
        
        database_config = DatabaseConfig()
        research_config = ResearchConfig()
        
        config = AppConfig(
            api=api_config,
            database=database_config,
            scraping=scraping_config,
            research=research_config
        )
        return config
    
    @pytest.fixture
    def web_scraper_agent(self, mock_config):
        """Create WebScraperAgent instance for testing"""
        return WebScraperAgent(mock_config)
    
    @pytest.fixture
    def sample_html_article(self):
        """Sample HTML content for testing"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Article: Machine Learning Advances</title>
            <meta name="author" content="Dr. Jane Smith">
            <meta name="publish-date" content="2024-01-15">
        </head>
        <body>
            <header>
                <nav>Navigation menu</nav>
            </header>
            <main>
                <article>
                    <h1>Machine Learning Advances in 2024</h1>
                    <div class="author">By Dr. Jane Smith</div>
                    <time datetime="2024-01-15">January 15, 2024</time>
                    
                    <div class="content">
                        <p>Machine learning has seen significant advances in 2024, particularly in the areas of 
                        natural language processing and computer vision. Researchers have developed new 
                        architectures that improve both efficiency and accuracy.</p>
                        
                        <p>The key breakthrough this year has been the development of more efficient 
                        transformer models that require less computational resources while maintaining 
                        high performance. This has made advanced AI capabilities more accessible to 
                        smaller organizations and researchers.</p>
                        
                        <p>Furthermore, the integration of multimodal learning approaches has enabled 
                        systems to better understand and process information across different data types, 
                        leading to more robust and versatile AI applications.</p>
                    </div>
                </article>
            </main>
            <aside class="sidebar">
                <div class="advertisement">Buy our product!</div>
                <div class="related-posts">Related articles...</div>
            </aside>
            <footer>
                <p>Copyright 2024</p>
            </footer>
        </body>
        </html>
        """
    
    @pytest.fixture
    def sample_html_news(self):
        """Sample news HTML for testing"""
        return """
        <html>
        <head><title>Breaking: Tech Company Announces New AI Model</title></head>
        <body>
            <div class="story-body">
                <h1>Tech Company Announces Revolutionary AI Model</h1>
                <div class="byline">By Tech Reporter</div>
                
                <p>A major technology company announced today the release of their latest 
                artificial intelligence model, which they claim represents a significant 
                leap forward in machine learning capabilities.</p>
                
                <p>The new model demonstrates improved performance across multiple benchmarks 
                and includes enhanced safety features designed to prevent misuse.</p>
            </div>
            <script>analytics.track('page_view');</script>
        </body>
        </html>
        """
    
    @pytest.fixture
    def sample_html_poor_quality(self):
        """Sample poor quality HTML for testing"""
        return """
        <html>
        <body>
            <div>
                <p>Short content.</p>
                <div class="ads">Advertisement here</div>
            </div>
        </body>
        </html>
        """
    
    def test_content_extraction_article(self, sample_html_article):
        """Test content extraction from well-structured article"""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(sample_html_article, 'html.parser')
        extracted = ContentExtractor.extract_content(soup, "https://example.com/article")
        
        # Verify extracted content
        assert extracted['title'] == "Test Article: Machine Learning Advances"
        assert "Machine learning has seen significant advances" in extracted['content']
        assert extracted['author'] == "By Dr. Jane Smith"
        assert extracted['word_count'] > 50
        assert extracted['extraction_quality'] in ['good', 'excellent']
        
        # Verify noise removal
        assert "Navigation menu" not in extracted['content']
        assert "Buy our product!" not in extracted['content']
        assert "Copyright 2024" not in extracted['content']
    
    def test_content_extraction_news(self, sample_html_news):
        """Test content extraction from news article"""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(sample_html_news, 'html.parser')
        extracted = ContentExtractor.extract_content(soup, "https://news.com/story")
        
        # Verify extracted content
        assert "Tech Company Announces" in extracted['title']
        assert "major technology company announced" in extracted['content']
        assert extracted['author'] == "By Tech Reporter"
        assert extracted['word_count'] > 20
        
        # Verify script removal
        assert "analytics.track" not in extracted['content']
    
    def test_content_extraction_poor_quality(self, sample_html_poor_quality):
        """Test content extraction from poor quality HTML"""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(sample_html_poor_quality, 'html.parser')
        extracted = ContentExtractor.extract_content(soup, "https://spam.com/page")
        
        # Should extract minimal content
        assert extracted['word_count'] < 10
        assert extracted['extraction_quality'] == 'poor'
        assert "Advertisement here" not in extracted['content']
    
    def test_static_page_scraping_success(self, web_scraper_agent, sample_html_article):
        """Test successful static page scraping"""
        with patch.object(web_scraper_agent.session, 'get') as mock_get:
            # Mock successful HTTP response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = sample_html_article.encode('utf-8')
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            # Execute scraping
            result = web_scraper_agent.scrape_static_page("https://example.com/article")
            
            # Verify result
            assert result.success is True
            assert result.method_used == "beautifulsoup"
            assert isinstance(result.content, ScrapedContent)
            assert result.content.title == "Test Article: Machine Learning Advances"
            assert result.content.author == "By Dr. Jane Smith"
            assert result.content.extraction_method == "beautifulsoup"
            assert len(result.content.content) > 100
    
    def test_static_page_scraping_http_error(self, web_scraper_agent):
        """Test static page scraping with HTTP error"""
        with patch.object(web_scraper_agent.session, 'get') as mock_get:
            # Mock HTTP error
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.raise_for_status.side_effect = Exception("404 Not Found")
            mock_get.return_value = mock_response
            
            # Execute scraping
            result = web_scraper_agent.scrape_static_page("https://example.com/notfound")
            
            # Verify error handling
            assert result.success is False
            assert "404" in result.error or "Not Found" in result.error
            assert result.method_used == "beautifulsoup"
    
    def test_static_page_scraping_timeout(self, web_scraper_agent):
        """Test static page scraping with timeout"""
        with patch.object(web_scraper_agent.session, 'get') as mock_get:
            # Mock timeout
            from requests.exceptions import Timeout
            mock_get.side_effect = Timeout("Request timeout")
            
            # Execute scraping
            result = web_scraper_agent.scrape_static_page("https://example.com/slow")
            
            # Verify timeout handling
            assert result.success is False
            assert "timeout" in result.error.lower()
            assert result.method_used == "beautifulsoup"
    
    def test_robots_txt_checking(self, web_scraper_agent):
        """Test robots.txt compliance checking"""
        # Enable robots.txt checking
        web_scraper_agent.scraping_config.respect_robots_txt = True
        
        with patch.object(web_scraper_agent.robots_checker, 'can_fetch') as mock_can_fetch:
            # Mock robots.txt blocking
            mock_can_fetch.return_value = False
            
            # Execute scraping
            result = web_scraper_agent.scrape_static_page("https://example.com/blocked")
            
            # Verify robots.txt blocking
            assert result.success is False
            assert "robots.txt" in result.error
    
    def test_multiple_page_scraping(self, web_scraper_agent, sample_html_article, sample_html_news):
        """Test scraping multiple pages"""
        urls = [
            "https://example.com/article1",
            "https://example.com/article2",
            "https://example.com/notfound"
        ]
        
        with patch.object(web_scraper_agent.session, 'get') as mock_get:
            # Mock responses for different URLs
            def mock_response_side_effect(url, **kwargs):
                mock_response = Mock()
                if "article1" in url:
                    mock_response.status_code = 200
                    mock_response.content = sample_html_article.encode('utf-8')
                    mock_response.raise_for_status = Mock()
                elif "article2" in url:
                    mock_response.status_code = 200
                    mock_response.content = sample_html_news.encode('utf-8')
                    mock_response.raise_for_status = Mock()
                else:  # notfound
                    mock_response.status_code = 404
                    mock_response.raise_for_status.side_effect = Exception("404 Not Found")
                return mock_response
            
            mock_get.side_effect = mock_response_side_effect
            
            # Execute batch scraping
            results = web_scraper_agent.scrape_multiple_pages(urls)
            
            # Verify results
            assert len(results) == 3
            assert results[0].success is True
            assert results[1].success is True
            assert results[2].success is False
            
            # Verify content
            assert "Machine Learning Advances" in results[0].content.title
            assert "Tech Company Announces" in results[1].content.title
    
    def test_page_scraping_strategy_selection(self, web_scraper_agent, sample_html_article):
        """Test automatic strategy selection for page scraping"""
        with patch.object(web_scraper_agent.session, 'get') as mock_get:
            # Mock successful static scraping
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = sample_html_article.encode('utf-8')
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            # Test default strategy (BeautifulSoup first)
            result = web_scraper_agent.scrape_page("https://example.com/article")
            
            assert result.success is True
            assert result.method_used == "beautifulsoup"
    
    def test_rate_limiting_scraper(self, web_scraper_agent):
        """Test rate limiting in web scraper"""
        # Set last scrape time to now
        web_scraper_agent.last_scrape_time = time.time()
        
        with patch('time.sleep') as mock_sleep:
            # This should trigger rate limiting
            web_scraper_agent._wait_for_rate_limit()
            
            # Should have called sleep
            mock_sleep.assert_called_once()
            assert mock_sleep.call_args[0][0] > 0  # Sleep time should be positive
    
    def test_health_check_scraper(self, web_scraper_agent):
        """Test web scraper health check"""
        with patch.object(web_scraper_agent, 'scrape_page') as mock_scrape:
            # Mock successful scrape
            mock_result = ScrapeResult(success=True, method_used="beautifulsoup")
            mock_scrape.return_value = mock_result
            
            # Execute health check
            health = web_scraper_agent.health_check()
            
            # Verify health status
            assert health['beautifulsoup_available'] is True
            assert health['scraping_functional'] is True
            assert 'max_concurrent_scrapes' in health
            assert 'scrape_timeout' in health


class TestWebAgentsIntegration:
    """Integration tests combining both web agents"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        api_config = APIConfig()
        api_config.serpapi_key = "test_key"
        
        scraping_config = ScrapingConfig()
        scraping_config.respect_robots_txt = False
        
        database_config = DatabaseConfig()
        research_config = ResearchConfig()
        
        config = AppConfig(
            api=api_config,
            database=database_config,
            scraping=scraping_config,
            research=research_config
        )
        return config
    
    def test_search_and_scrape_workflow(self, mock_config):
        """Test complete workflow: search -> scrape -> extract content"""
        search_agent = WebSearchAgent(mock_config)
        scraper_agent = WebScraperAgent(mock_config)
        
        # Mock search results
        mock_search_results = [
            SearchResult(
                title="Test Article",
                url="https://example.com/article",
                snippet="Test snippet",
                credibility_score=8.0,
                source="test"
            )
        ]
        
        # Mock HTML content
        sample_html = """
        <html>
        <head><title>Test Article</title></head>
        <body>
            <article>
                <h1>Test Article</h1>
                <p>This is a comprehensive article about machine learning 
                techniques and their applications in modern technology. 
                The content provides detailed analysis and research findings.</p>
            </article>
        </body>
        </html>
        """
        
        with patch.object(search_agent, 'search') as mock_search, \
             patch.object(scraper_agent.session, 'get') as mock_get:
            
            # Mock search results
            mock_search.return_value = mock_search_results
            
            # Mock scraping response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = sample_html.encode('utf-8')
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            # Execute workflow
            search_results = search_agent.search(["machine learning"], max_results=5)
            assert len(search_results) == 1
            
            # Scrape the first result
            scrape_result = scraper_agent.scrape_page(search_results[0].url)
            
            # Verify complete workflow
            assert scrape_result.success is True
            assert scrape_result.content.title == "Test Article"
            assert "machine learning techniques" in scrape_result.content.content
            assert len(scrape_result.content.content.split()) > 10
    
    def test_error_handling_integration(self, mock_config):
        """Test error handling across both agents"""
        search_agent = WebSearchAgent(mock_config)
        scraper_agent = WebScraperAgent(mock_config)
        
        # Test search failure
        with patch.object(search_agent, 'search_serpapi') as mock_serpapi, \
             patch.object(search_agent, 'search_duckduckgo') as mock_ddg:
            
            # Mock both search methods failing
            mock_serpapi.side_effect = Exception("SerpAPI failed")
            mock_ddg.side_effect = Exception("DuckDuckGo failed")
            
            # Should handle gracefully
            with pytest.raises(Exception):
                search_agent.search(["test query"])
        
        # Test scraping failure with fallback
        with patch.object(scraper_agent.session, 'get') as mock_get:
            mock_get.side_effect = Exception("Network error")
            
            result = scraper_agent.scrape_page("https://example.com/test")
            assert result.success is False
            assert "error" in result.error.lower()