"""
End-to-End System Tests for the Intelligent Research Assistant.
Tests complete user workflows from query to report and validates system performance under load.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import List, Dict, Any
import concurrent.futures

from agents.main_orchestrator import MainOrchestrator, ResearchStage
from agents.router_agent import RouterAgent
from agents.web_search_agent import WebSearchAgent
from agents.web_scraper_agent import WebScraperAgent
from agents.vector_search_agent import VectorSearchAgent
from agents.fact_checker_agent import FactCheckerAgent
from agents.summarizer_agent import SummarizerAgent
from agents.data_models import (
    ResearchConfig, ResearchResult, ResearchReport, ProgressStatus,
    SearchResult, ScrapedContent, Document, FactCheckResult,
    ReportStyle, ReportLength, ResearchPlan, QueryAnalysis, ResearchStrategy,
    InformationType, ComplexityLevel
)
from utils.config import AppConfig, APIConfig, DatabaseConfig, ScrapingConfig, ResearchConfig as UtilsResearchConfig, SheetsConfig
from ui.gradio_interface import GradioInterface


class TestEndToEndUserWorkflows:
    """Test complete user workflows from query submission to report generation"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        api_config = APIConfig()
        api_config.gemini_api_key = "test_gemini_key"
        api_config.serpapi_key = "test_serpapi_key"
        api_config.google_sheets_credentials_path = None  # Disable sheets for E2E tests
        
        database_config = DatabaseConfig()
        scraping_config = ScrapingConfig()
        research_config = UtilsResearchConfig()
        sheets_config = SheetsConfig()
        
        return AppConfig(
            api=api_config,
            database=database_config,
            scraping=scraping_config,
            research=research_config,
            sheets=sheets_config
        )

    @pytest.fixture
    def create_orchestrator_with_mocks(self, mock_config):
        """Factory to create orchestrator with mock agents"""
        def _create():
            # Create mock agents
            router_agent = Mock(spec=RouterAgent)
            web_search_agent = Mock(spec=WebSearchAgent)
            web_scraper_agent = Mock(spec=WebScraperAgent)
            vector_search_agent = Mock(spec=VectorSearchAgent)
            fact_checker_agent = Mock(spec=FactCheckerAgent)
            summarizer_agent = Mock(spec=SummarizerAgent)
            
            # Create orchestrator
            orchestrator = MainOrchestrator(
                router_agent=router_agent,
                web_search_agent=web_search_agent,
                web_scraper_agent=web_scraper_agent,
                vector_search_agent=vector_search_agent,
                fact_checker_agent=fact_checker_agent,
                summarizer_agent=summarizer_agent,
                config=mock_config
            )
            
            return orchestrator, {
                'router': router_agent,
                'web_search': web_search_agent,
                'web_scraper': web_scraper_agent,
                'vector_search': vector_search_agent,
                'fact_checker': fact_checker_agent,
                'summarizer': summarizer_agent
            }
        return _create
    
    def test_complete_research_workflow_success(self, create_orchestrator_with_mocks):
        """Test complete successful research workflow from query to report"""
        orchestrator, mock_agents = create_orchestrator_with_mocks()
        
        # Configure mock responses for complete workflow
        mock_agents['router'].analyze_query.return_value = ResearchPlan(
            query_analysis=QueryAnalysis(
                topic_category="Technology",
                information_type=InformationType.FACTUAL,
                complexity_level=ComplexityLevel.MEDIUM,
                estimated_time_minutes=2
            ),
            research_strategy=ResearchStrategy(
                use_web_search=True,
                use_web_scraping=True,
                use_vector_search=True,
                priority_order=["web_search", "vector_search", "web_scraping"]
            ),
            search_queries=["artificial intelligence applications", "AI use cases 2024"],
            target_websites=["https://example.com/ai-article"],
            expected_challenges=[]
        )
        
        mock_agents['web_search'].search.return_value = [
            SearchResult(
                title="AI Applications in Healthcare",
                url="https://healthcare.com/ai",
                snippet="AI is transforming healthcare with diagnostic tools",
                credibility_score=9.0,
                source="serpapi"
            ),
            SearchResult(
                title="Machine Learning in Finance",
                url="https://finance.com/ml",
                snippet="ML algorithms are revolutionizing financial analysis",
                credibility_score=8.5,
                source="serpapi"
            )
        ]

        mock_agents['web_scraper'].scrape_multiple_pages.return_value = [
            Mock(
                success=True,
                content=ScrapedContent(
                    url="https://example.com/ai-article",
                    title="Comprehensive AI Guide",
                    content="Artificial intelligence is being applied across multiple industries including healthcare, finance, and education. Recent advances in deep learning have enabled more sophisticated applications.",
                    author="Dr. Jane Smith",
                    publish_date=datetime.now() - timedelta(days=10),
                    extraction_method="beautifulsoup"
                )
            )
        ]
        
        mock_agents['vector_search'].search.return_value = [
            Document(
                content="Historical context: AI research has evolved significantly over the past decade with breakthroughs in neural networks.",
                metadata={'source_url': 'https://archive.com/ai-history', 'timestamp': datetime.now()},
                similarity_score=0.82,
                credibility_score=8.0
            )
        ]
        
        mock_agents['fact_checker'].check_facts.return_value = FactCheckResult(
            verified_facts=[
                {"fact": "AI is transforming healthcare", "confidence": 0.95, "sources": 2},
                {"fact": "ML algorithms are used in financial analysis", "confidence": 0.90, "sources": 2}
            ],
            credibility_scores={"healthcare.com": 9.0, "finance.com": 8.5},
            contradictions=[],
            cleaned_data=["AI healthcare applications", "ML financial analysis"]
        )
        
        mock_agents['summarizer'].generate_report.return_value = ResearchReport(
            executive_summary="Artificial intelligence is rapidly transforming multiple industries with significant applications in healthcare and finance.",
            key_findings=[
                "AI diagnostic tools are improving healthcare outcomes",
                "Machine learning is revolutionizing financial analysis",
                "Deep learning advances enable sophisticated applications"
            ],
            detailed_analysis="The research reveals that AI technologies are being successfully deployed across healthcare and finance sectors. Healthcare applications focus on diagnostic accuracy, while financial applications emphasize predictive analytics and risk assessment.",
            sources=[
                "[1] AI Applications in Healthcare - https://healthcare.com/ai",
                "[2] Machine Learning in Finance - https://finance.com/ml",
                "[3] Comprehensive AI Guide - https://example.com/ai-article"
            ],
            recommendations=[
                "Organizations should explore AI integration opportunities",
                "Investment in AI infrastructure is recommended for competitive advantage"
            ],
            metadata={"word_count": 850, "confidence_level": "high", "source_diversity": "good"}
        )
        
        # Execute complete workflow
        query = "What are the current applications of artificial intelligence?"
        config = ResearchConfig(max_sources=10, timeout_seconds=120)
        
        result = orchestrator.research(query, config)
        
        # Verify complete workflow success
        assert result.success is True
        assert result.query == query
        assert isinstance(result.report, ResearchReport)
        assert result.source_count > 0
        assert result.execution_time > 0
        
        # Verify report content
        assert "artificial intelligence" in result.report.executive_summary.lower()
        assert len(result.report.key_findings) >= 3
        assert len(result.report.sources) >= 3
        assert len(result.report.recommendations) >= 1
        
        # Verify all agents were called
        mock_agents['router'].analyze_query.assert_called_once()
        mock_agents['web_search'].search.assert_called_once()
        mock_agents['web_scraper'].scrape_multiple_pages.assert_called_once()
        mock_agents['vector_search'].search.assert_called_once()
        mock_agents['fact_checker'].check_facts.assert_called_once()
        mock_agents['summarizer'].generate_report.assert_called_once()
        
        # Verify metadata
        assert 'execution_time' in result.metadata
        assert 'agents_used' in result.metadata
        assert 'data_sources' in result.metadata
        assert len(result.metadata['agents_used']) >= 5

    def test_workflow_with_partial_data_collection_failure(self, create_orchestrator_with_mocks):
        """Test workflow continues successfully when some data sources fail"""
        orchestrator, mock_agents = create_orchestrator_with_mocks()
        
        # Configure router success
        mock_agents['router'].analyze_query.return_value = ResearchPlan(
            query_analysis=QueryAnalysis(
                topic_category="Science",
                information_type=InformationType.FACTUAL,
                complexity_level=ComplexityLevel.LOW,
                estimated_time_minutes=1
            ),
            research_strategy=ResearchStrategy(
                use_web_search=True,
                use_web_scraping=True,
                use_vector_search=True,
                priority_order=["web_search", "vector_search", "web_scraping"]
            ),
            search_queries=["climate change impacts"],
            target_websites=["https://blocked-site.com"],
            expected_challenges=[]
        )
        
        # Web search succeeds
        mock_agents['web_search'].search.return_value = [
            SearchResult(
                title="Climate Change Report 2024",
                url="https://climate.org/report",
                snippet="Global temperatures continue to rise",
                credibility_score=9.5,
                source="serpapi"
            )
        ]
        
        # Web scraper fails
        mock_agents['web_scraper'].scrape_multiple_pages.side_effect = Exception("Scraping blocked")
        
        # Vector search fails
        mock_agents['vector_search'].search.side_effect = Exception("Database connection error")
        
        # Fact checker and summarizer succeed with available data
        mock_agents['fact_checker'].check_facts.return_value = FactCheckResult(
            verified_facts=[{"fact": "Global temperatures rising", "confidence": 0.95, "sources": 1}],
            credibility_scores={"climate.org": 9.5},
            contradictions=[],
            cleaned_data=["Climate change data"]
        )
        
        mock_agents['summarizer'].generate_report.return_value = ResearchReport(
            executive_summary="Climate change continues to impact global temperatures.",
            key_findings=["Global temperatures are rising"],
            detailed_analysis="Available data indicates ongoing climate change effects.",
            sources=["[1] Climate Change Report 2024 - https://climate.org/report"],
            recommendations=["Monitor climate trends"],
            metadata={"word_count": 200, "confidence_level": "medium"}
        )
        
        # Execute workflow
        result = orchestrator.research("What are the impacts of climate change?")
        
        # Should succeed with partial data
        assert result.success is True
        assert result.source_count >= 1
        assert 'web_scraper' in result.metadata['failed_agents']
        assert 'vector_search' in result.metadata['failed_agents']
        assert 'web_search' in result.metadata['agents_used']

    def test_workflow_with_progress_tracking(self, create_orchestrator_with_mocks):
        """Test that progress updates are sent throughout the workflow"""
        orchestrator, mock_agents = create_orchestrator_with_mocks()
        
        progress_updates = []
        
        def progress_callback(status: ProgressStatus):
            progress_updates.append({
                'step': status.current_step,
                'percentage': status.completion_percentage,
                'message': status.status_message,
                'completed_agents': list(status.completed_agents),
                'failed_agents': list(status.failed_agents)
            })
        
        orchestrator.progress_callback = progress_callback
        
        # Configure successful workflow
        mock_agents['router'].analyze_query.return_value = ResearchPlan(
            query_analysis=QueryAnalysis(
                topic_category="Technology",
                information_type=InformationType.FACTUAL,
                complexity_level=ComplexityLevel.LOW,
                estimated_time_minutes=1
            ),
            research_strategy=ResearchStrategy(
                use_web_search=True,
                use_web_scraping=False,
                use_vector_search=False,
                priority_order=["web_search"]
            ),
            search_queries=["test query"],
            target_websites=[],
            expected_challenges=[]
        )
        
        mock_agents['web_search'].search.return_value = [
            SearchResult("Test", "https://test.com", "snippet", 8.0, "test")
        ]
        
        mock_agents['fact_checker'].check_facts.return_value = FactCheckResult(
            verified_facts=[{"fact": "Test fact", "confidence": 0.9, "sources": 1}],
            credibility_scores={"test.com": 8.0},
            contradictions=[],
            cleaned_data=["Test data"]
        )
        
        mock_agents['summarizer'].generate_report.return_value = ResearchReport(
            executive_summary="Test summary",
            key_findings=["Finding 1"],
            detailed_analysis="Test analysis",
            sources=["[1] Test"],
            recommendations=["Recommendation"],
            metadata={"word_count": 100}
        )
        
        # Execute workflow
        result = orchestrator.research("test query")
        
        # Verify progress updates were sent
        assert len(progress_updates) > 0
        
        # Verify progress stages
        stages = [update['step'] for update in progress_updates]
        assert 'planning' in stages
        assert 'data_collection' in stages
        assert 'fact_checking' in stages
        assert 'report_generation' in stages
        
        # Verify progress increases
        percentages = [update['percentage'] for update in progress_updates]
        assert percentages[-1] > percentages[0]
        
        # Verify completed agents are tracked
        final_update = progress_updates[-1]
        assert len(final_update['completed_agents']) > 0

    def test_workflow_with_different_report_styles(self, create_orchestrator_with_mocks):
        """Test workflow with different report style configurations"""
        orchestrator, mock_agents = create_orchestrator_with_mocks()
        
        # Configure basic successful workflow
        mock_agents['router'].analyze_query.return_value = ResearchPlan(
            query_analysis=QueryAnalysis(
                topic_category="General",
                information_type=InformationType.FACTUAL,
                complexity_level=ComplexityLevel.LOW,
                estimated_time_minutes=1
            ),
            research_strategy=ResearchStrategy(
                use_web_search=True,
                use_web_scraping=False,
                use_vector_search=False,
                priority_order=["web_search"]
            ),
            search_queries=["test"],
            target_websites=[],
            expected_challenges=[]
        )
        
        mock_agents['web_search'].search.return_value = [
            SearchResult("Test", "https://test.com", "snippet", 8.0, "test")
        ]
        
        mock_agents['fact_checker'].check_facts.return_value = FactCheckResult(
            verified_facts=[{"fact": "Test", "confidence": 0.9, "sources": 1}],
            credibility_scores={},
            contradictions=[],
            cleaned_data=[]
        )
        
        # Test different report styles
        for style in [ReportStyle.ACADEMIC, ReportStyle.CASUAL, ReportStyle.TECHNICAL]:
            mock_agents['summarizer'].generate_report.return_value = ResearchReport(
                executive_summary=f"{style.value} style summary",
                key_findings=["Finding"],
                detailed_analysis="Analysis",
                sources=["Source"],
                recommendations=["Recommendation"],
                metadata={"word_count": 100, "style": style.value}
            )
            
            config = ResearchConfig(report_style=style)
            result = orchestrator.research("test query", config)
            
            assert result.success is True
            assert style.value in result.report.executive_summary

    def test_workflow_timeout_handling(self, create_orchestrator_with_mocks):
        """Test workflow handles timeouts gracefully"""
        orchestrator, mock_agents = create_orchestrator_with_mocks()
        
        # Configure router to succeed
        mock_agents['router'].analyze_query.return_value = ResearchPlan(
            query_analysis=QueryAnalysis(
                topic_category="Test",
                information_type=InformationType.FACTUAL,
                complexity_level=ComplexityLevel.LOW,
                estimated_time_minutes=1
            ),
            research_strategy=ResearchStrategy(
                use_web_search=True,
                use_web_scraping=False,
                use_vector_search=False,
                priority_order=["web_search"]
            ),
            search_queries=["test"],
            target_websites=[],
            expected_challenges=[]
        )
        
        # Configure web search to be slow
        def slow_search(*args, **kwargs):
            time.sleep(2)
            return []
        
        mock_agents['web_search'].search.side_effect = slow_search
        
        # Execute with short timeout
        config = ResearchConfig(timeout_seconds=1)
        result = orchestrator.research("test query", config)
        
        # Should handle timeout gracefully
        assert isinstance(result, ResearchResult)
        # May succeed or fail depending on timing, but should not crash


class TestSystemPerformance:
    """Test system performance under various load conditions"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        api_config = APIConfig()
        api_config.gemini_api_key = "test_key"
        api_config.serpapi_key = "test_key"
        api_config.google_sheets_credentials_path = None
        
        database_config = DatabaseConfig()
        scraping_config = ScrapingConfig()
        research_config = UtilsResearchConfig()
        sheets_config = SheetsConfig()
        
        return AppConfig(
            api=api_config,
            database=database_config,
            scraping=scraping_config,
            research=research_config,
            sheets=sheets_config
        )
    
    def test_sequential_research_requests(self, mock_config):
        """Test system handles sequential research requests efficiently"""
        # Create orchestrator with mock agents
        router_agent = Mock(spec=RouterAgent)
        web_search_agent = Mock(spec=WebSearchAgent)
        web_scraper_agent = Mock(spec=WebScraperAgent)
        vector_search_agent = Mock(spec=VectorSearchAgent)
        fact_checker_agent = Mock(spec=FactCheckerAgent)
        summarizer_agent = Mock(spec=SummarizerAgent)
        
        orchestrator = MainOrchestrator(
            router_agent=router_agent,
            web_search_agent=web_search_agent,
            web_scraper_agent=web_scraper_agent,
            vector_search_agent=vector_search_agent,
            fact_checker_agent=fact_checker_agent,
            summarizer_agent=summarizer_agent,
            config=mock_config
        )
        
        # Configure mock responses
        router_agent.analyze_query.return_value = ResearchPlan(
            query_analysis=QueryAnalysis(
                topic_category="Test",
                information_type=InformationType.FACTUAL,
                complexity_level=ComplexityLevel.LOW,
                estimated_time_minutes=1
            ),
            research_strategy=ResearchStrategy(
                use_web_search=True,
                use_web_scraping=False,
                use_vector_search=False,
                priority_order=["web_search"]
            ),
            search_queries=["test"],
            target_websites=[],
            expected_challenges=[]
        )
        
        web_search_agent.search.return_value = [
            SearchResult("Test", "https://test.com", "snippet", 8.0, "test")
        ]
        
        fact_checker_agent.check_facts.return_value = FactCheckResult(
            verified_facts=[{"fact": "Test", "confidence": 0.9, "sources": 1}],
            credibility_scores={},
            contradictions=[],
            cleaned_data=[]
        )
        
        summarizer_agent.generate_report.return_value = ResearchReport(
            executive_summary="Summary",
            key_findings=["Finding"],
            detailed_analysis="Analysis",
            sources=["Source"],
            recommendations=["Recommendation"],
            metadata={"word_count": 100}
        )
        
        # Execute multiple sequential requests
        num_requests = 5
        execution_times = []
        
        for i in range(num_requests):
            start_time = time.time()
            result = orchestrator.research(f"test query {i}")
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            
            assert result.success is True
        
        # Verify all requests completed
        assert len(execution_times) == num_requests
        
        # Verify reasonable performance (each request should complete quickly with mocks)
        avg_time = sum(execution_times) / len(execution_times)
        assert avg_time < 5.0  # Should be fast with mocks

    def test_concurrent_research_requests(self, mock_config):
        """Test system handles concurrent research requests"""
        # Create orchestrator with mock agents
        router_agent = Mock(spec=RouterAgent)
        web_search_agent = Mock(spec=WebSearchAgent)
        web_scraper_agent = Mock(spec=WebScraperAgent)
        vector_search_agent = Mock(spec=VectorSearchAgent)
        fact_checker_agent = Mock(spec=FactCheckerAgent)
        summarizer_agent = Mock(spec=SummarizerAgent)
        
        orchestrator = MainOrchestrator(
            router_agent=router_agent,
            web_search_agent=web_search_agent,
            web_scraper_agent=web_scraper_agent,
            vector_search_agent=vector_search_agent,
            fact_checker_agent=fact_checker_agent,
            summarizer_agent=summarizer_agent,
            config=mock_config
        )
        
        # Configure mock responses
        router_agent.analyze_query.return_value = ResearchPlan(
            query_analysis=QueryAnalysis(
                topic_category="Test",
                information_type=InformationType.FACTUAL,
                complexity_level=ComplexityLevel.LOW,
                estimated_time_minutes=1
            ),
            research_strategy=ResearchStrategy(
                use_web_search=True,
                use_web_scraping=False,
                use_vector_search=False,
                priority_order=["web_search"]
            ),
            search_queries=["test"],
            target_websites=[],
            expected_challenges=[]
        )
        
        web_search_agent.search.return_value = [
            SearchResult("Test", "https://test.com", "snippet", 8.0, "test")
        ]
        
        fact_checker_agent.check_facts.return_value = FactCheckResult(
            verified_facts=[{"fact": "Test", "confidence": 0.9, "sources": 1}],
            credibility_scores={},
            contradictions=[],
            cleaned_data=[]
        )
        
        summarizer_agent.generate_report.return_value = ResearchReport(
            executive_summary="Summary",
            key_findings=["Finding"],
            detailed_analysis="Analysis",
            sources=["Source"],
            recommendations=["Recommendation"],
            metadata={"word_count": 100}
        )
        
        # Execute concurrent requests
        num_concurrent = 3
        results = []
        
        def execute_research(query_id):
            result = orchestrator.research(f"concurrent query {query_id}")
            return result
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(execute_research, i) for i in range(num_concurrent)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Verify all requests completed successfully
        assert len(results) == num_concurrent
        for result in results:
            assert result.success is True

    def test_memory_efficiency_with_large_datasets(self, mock_config):
        """Test system handles large datasets efficiently"""
        # Create orchestrator with mock agents
        router_agent = Mock(spec=RouterAgent)
        web_search_agent = Mock(spec=WebSearchAgent)
        web_scraper_agent = Mock(spec=WebScraperAgent)
        vector_search_agent = Mock(spec=VectorSearchAgent)
        fact_checker_agent = Mock(spec=FactCheckerAgent)
        summarizer_agent = Mock(spec=SummarizerAgent)
        
        orchestrator = MainOrchestrator(
            router_agent=router_agent,
            web_search_agent=web_search_agent,
            web_scraper_agent=web_scraper_agent,
            vector_search_agent=vector_search_agent,
            fact_checker_agent=fact_checker_agent,
            summarizer_agent=summarizer_agent,
            config=mock_config
        )
        
        # Configure router
        router_agent.analyze_query.return_value = ResearchPlan(
            query_analysis=QueryAnalysis(
                topic_category="Test",
                information_type=InformationType.FACTUAL,
                complexity_level=ComplexityLevel.HIGH,
                estimated_time_minutes=3
            ),
            research_strategy=ResearchStrategy(
                use_web_search=True,
                use_web_scraping=True,
                use_vector_search=True,
                priority_order=["web_search", "vector_search", "web_scraping"]
            ),
            search_queries=["test"] * 5,
            target_websites=["https://test.com"] * 5,
            expected_challenges=[]
        )
        
        # Create large dataset responses
        large_search_results = [
            SearchResult(f"Title {i}", f"https://test{i}.com", "content" * 100, 8.0, "test")
            for i in range(50)
        ]
        
        large_scraped_content = [
            Mock(
                success=True,
                content=ScrapedContent(
                    url=f"https://test{i}.com",
                    title=f"Article {i}",
                    content="Large content " * 500,
                    author="Author",
                    publish_date=datetime.now(),
                    extraction_method="beautifulsoup"
                )
            )
            for i in range(20)
        ]
        
        large_vector_docs = [
            Document(
                content="Vector content " * 100,
                metadata={'source': f'doc{i}'},
                similarity_score=0.8,
                credibility_score=8.0
            )
            for i in range(30)
        ]
        
        web_search_agent.search.return_value = large_search_results
        web_scraper_agent.scrape_multiple_pages.return_value = large_scraped_content
        vector_search_agent.search.return_value = large_vector_docs
        
        # Configure fact checker to handle large dataset
        fact_checker_agent.check_facts.return_value = FactCheckResult(
            verified_facts=[{"fact": f"Fact {i}", "confidence": 0.9, "sources": 2} for i in range(20)],
            credibility_scores={f"test{i}.com": 8.0 for i in range(10)},
            contradictions=[],
            cleaned_data=["Data"] * 50
        )
        
        summarizer_agent.generate_report.return_value = ResearchReport(
            executive_summary="Large dataset summary",
            key_findings=[f"Finding {i}" for i in range(10)],
            detailed_analysis="Comprehensive analysis " * 50,
            sources=[f"[{i}] Source {i}" for i in range(20)],
            recommendations=[f"Recommendation {i}" for i in range(5)],
            metadata={"word_count": 1000, "source_count": 100}
        )
        
        # Execute research with large dataset
        config = ResearchConfig(max_sources=50)
        result = orchestrator.research("large dataset query", config)
        
        # Verify successful handling
        assert result.success is True
        assert result.source_count >= 50
        assert len(result.report.sources) >= 10

    def test_performance_metrics_collection(self, mock_config):
        """Test that performance metrics are properly collected"""
        # Create orchestrator with mock agents
        router_agent = Mock(spec=RouterAgent)
        web_search_agent = Mock(spec=WebSearchAgent)
        web_scraper_agent = Mock(spec=WebScraperAgent)
        vector_search_agent = Mock(spec=VectorSearchAgent)
        fact_checker_agent = Mock(spec=FactCheckerAgent)
        summarizer_agent = Mock(spec=SummarizerAgent)
        
        orchestrator = MainOrchestrator(
            router_agent=router_agent,
            web_search_agent=web_search_agent,
            web_scraper_agent=web_scraper_agent,
            vector_search_agent=vector_search_agent,
            fact_checker_agent=fact_checker_agent,
            summarizer_agent=summarizer_agent,
            config=mock_config
        )
        
        # Configure successful workflow
        router_agent.analyze_query.return_value = ResearchPlan(
            query_analysis=QueryAnalysis(
                topic_category="Test",
                information_type=InformationType.FACTUAL,
                complexity_level=ComplexityLevel.LOW,
                estimated_time_minutes=1
            ),
            research_strategy=ResearchStrategy(
                use_web_search=True,
                use_web_scraping=False,
                use_vector_search=False,
                priority_order=["web_search"]
            ),
            search_queries=["test"],
            target_websites=[],
            expected_challenges=[]
        )
        
        web_search_agent.search.return_value = [
            SearchResult("Test", "https://test.com", "snippet", 8.0, "test")
        ]
        
        fact_checker_agent.check_facts.return_value = FactCheckResult(
            verified_facts=[{"fact": "Test", "confidence": 0.9, "sources": 1}],
            credibility_scores={},
            contradictions=[],
            cleaned_data=[]
        )
        
        summarizer_agent.generate_report.return_value = ResearchReport(
            executive_summary="Summary",
            key_findings=["Finding"],
            detailed_analysis="Analysis",
            sources=["Source"],
            recommendations=["Recommendation"],
            metadata={"word_count": 100}
        )
        
        # Execute research
        result = orchestrator.research("performance test query")
        
        # Verify performance metrics are collected
        assert result.execution_time > 0
        assert 'execution_time' in result.metadata
        assert 'agents_used' in result.metadata
        assert 'data_sources' in result.metadata
        
        # Verify timing information
        assert isinstance(result.execution_time, float)
        assert result.execution_time < 60  # Should complete quickly with mocks


class TestUIIntegration:
    """Test UI integration with the complete system"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        config = Mock()
        config.api = Mock()
        config.api.gemini_api_key = "test_key"
        config.api.serpapi_key = "test_key"
        config.api.google_sheets_credentials_path = None
        config.database = Mock()
        config.database.chroma_db_path = "/test/chroma"
        config.gradio_host = "127.0.0.1"
        config.gradio_port = 7860
        return config
    
    def test_ui_to_orchestrator_integration(self, mock_config):
        """Test complete integration from UI to orchestrator"""
        with patch('ui.gradio_interface.get_config', return_value=mock_config):
            with patch('os.path.exists', return_value=True):
                interface = GradioInterface()
                
                # Mock orchestrator
                mock_orchestrator = Mock()
                mock_result = Mock()
                mock_result.success = True
                mock_result.query = "UI integration test"
                mock_result.execution_time = 30.0
                mock_result.source_count = 5
                mock_result.report = ResearchReport(
                    executive_summary="UI integration summary",
                    key_findings=["UI finding 1", "UI finding 2"],
                    detailed_analysis="UI integration analysis",
                    sources=["[1] UI source"],
                    recommendations=["UI recommendation"],
                    metadata={"word_count": 300}
                )
                mock_result.metadata = {
                    'execution_time': 30.0,
                    'agents_used': ['router', 'web_search', 'summarizer'],
                    'data_sources': {'web_search_results': 5}
                }
                
                mock_orchestrator.research.return_value = mock_result
                interface.orchestrator = mock_orchestrator
                
                # Execute research through UI
                result = interface.conduct_research(
                    "UI integration test query",
                    10, True, True, "academic", "medium", 120
                )
                
                report, metadata, status, progress = result
                
                # Verify UI integration
                assert "# Research Report:" in report
                assert "UI integration" in report
                assert "UI integration summary" in report
                assert "✅ Research completed successfully" in status
                assert "30.0s" in status

    def test_ui_error_handling_integration(self, mock_config):
        """Test UI error handling with orchestrator failures"""
        with patch('ui.gradio_interface.get_config', return_value=mock_config):
            with patch('os.path.exists', return_value=True):
                interface = GradioInterface()
                
                # Mock orchestrator with failure
                mock_orchestrator = Mock()
                mock_result = Mock()
                mock_result.success = False
                mock_result.query = "failed query"
                mock_result.error_message = "Data collection failed"
                mock_result.report = None
                mock_result.metadata = {
                    'failed_agents': ['web_search'],
                    'errors': ['Connection timeout']
                }
                
                mock_orchestrator.research.return_value = mock_result
                interface.orchestrator = mock_orchestrator
                
                # Execute research through UI
                result = interface.conduct_research(
                    "failing query", 10, True, True, "academic", "medium", 120
                )
                
                report, metadata, status, progress = result
                
                # Verify error handling
                assert "❌ **Research Failed**" in report
                assert "Data collection failed" in report
                assert "Research failed" in status


class TestDataFlowValidation:
    """Test data flow validation between system components"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        api_config = APIConfig()
        api_config.gemini_api_key = "test_key"
        api_config.serpapi_key = "test_key"
        api_config.google_sheets_credentials_path = None
        
        database_config = DatabaseConfig()
        scraping_config = ScrapingConfig()
        research_config = UtilsResearchConfig()
        sheets_config = SheetsConfig()
        
        return AppConfig(
            api=api_config,
            database=database_config,
            scraping=scraping_config,
            research=research_config,
            sheets=sheets_config
        )
    
    def test_data_transformation_between_agents(self, mock_config):
        """Test data is properly transformed between agent stages"""
        # Create orchestrator with mock agents
        router_agent = Mock(spec=RouterAgent)
        web_search_agent = Mock(spec=WebSearchAgent)
        web_scraper_agent = Mock(spec=WebScraperAgent)
        vector_search_agent = Mock(spec=VectorSearchAgent)
        fact_checker_agent = Mock(spec=FactCheckerAgent)
        summarizer_agent = Mock(spec=SummarizerAgent)
        
        orchestrator = MainOrchestrator(
            router_agent=router_agent,
            web_search_agent=web_search_agent,
            web_scraper_agent=web_scraper_agent,
            vector_search_agent=vector_search_agent,
            fact_checker_agent=fact_checker_agent,
            summarizer_agent=summarizer_agent,
            config=mock_config
        )
        
        # Configure workflow
        router_agent.analyze_query.return_value = ResearchPlan(
            query_analysis=QueryAnalysis(
                topic_category="Test",
                information_type=InformationType.FACTUAL,
                complexity_level=ComplexityLevel.LOW,
                estimated_time_minutes=1
            ),
            research_strategy=ResearchStrategy(
                use_web_search=True,
                use_web_scraping=False,
                use_vector_search=False,
                priority_order=["web_search"]
            ),
            search_queries=["test query"],
            target_websites=[],
            expected_challenges=[]
        )
        
        search_results = [
            SearchResult(
                title="Test Article",
                url="https://test.com/article",
                snippet="Test content",
                credibility_score=8.5,
                source="serpapi"
            )
        ]
        web_search_agent.search.return_value = search_results
        
        # Configure fact checker and summarizer with simple return values
        fact_checker_agent.check_facts.return_value = FactCheckResult(
            verified_facts=[{"fact": "Test fact", "confidence": 0.9, "sources": 1}],
            credibility_scores={"test.com": 8.5},
            contradictions=[],
            cleaned_data=["Test data"]
        )
        
        summarizer_agent.generate_report.return_value = ResearchReport(
            executive_summary="Summary",
            key_findings=["Finding"],
            detailed_analysis="Analysis",
            sources=["Source"],
            recommendations=["Recommendation"],
            metadata={"word_count": 100}
        )
        
        # Execute workflow
        result = orchestrator.research("data flow test")
        
        # Verify workflow completed (may succeed or fail depending on data validation)
        assert isinstance(result, ResearchResult)
        assert result.query == "data flow test"
        
        # Verify agents were called in proper sequence
        assert router_agent.analyze_query.called
        assert web_search_agent.search.called
        assert fact_checker_agent.check_facts.called
        
        # If successful, verify summarizer was also called
        if result.success:
            assert summarizer_agent.generate_report.called

    def test_error_propagation_through_system(self, mock_config):
        """Test that errors are properly propagated and handled"""
        # Create orchestrator with mock agents
        router_agent = Mock(spec=RouterAgent)
        web_search_agent = Mock(spec=WebSearchAgent)
        web_scraper_agent = Mock(spec=WebScraperAgent)
        vector_search_agent = Mock(spec=VectorSearchAgent)
        fact_checker_agent = Mock(spec=FactCheckerAgent)
        summarizer_agent = Mock(spec=SummarizerAgent)
        
        orchestrator = MainOrchestrator(
            router_agent=router_agent,
            web_search_agent=web_search_agent,
            web_scraper_agent=web_scraper_agent,
            vector_search_agent=vector_search_agent,
            fact_checker_agent=fact_checker_agent,
            summarizer_agent=summarizer_agent,
            config=mock_config
        )
        
        # Configure router to fail
        router_agent.analyze_query.side_effect = Exception("Router analysis failed")
        
        # Execute workflow
        result = orchestrator.research("error propagation test")
        
        # Verify error is captured
        assert result.success is False
        assert len(result.metadata['errors']) > 0
        assert 'router' in result.metadata['failed_agents']
        assert result.error_message is not None


class TestSystemResilience:
    """Test system resilience and recovery capabilities"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        api_config = APIConfig()
        api_config.gemini_api_key = "test_key"
        api_config.serpapi_key = "test_key"
        api_config.google_sheets_credentials_path = None
        
        database_config = DatabaseConfig()
        scraping_config = ScrapingConfig()
        research_config = UtilsResearchConfig()
        sheets_config = SheetsConfig()
        
        return AppConfig(
            api=api_config,
            database=database_config,
            scraping=scraping_config,
            research=research_config,
            sheets=sheets_config
        )
    
    def test_recovery_from_partial_failures(self, mock_config):
        """Test system recovers from partial agent failures"""
        # Create orchestrator with mock agents
        router_agent = Mock(spec=RouterAgent)
        web_search_agent = Mock(spec=WebSearchAgent)
        web_scraper_agent = Mock(spec=WebScraperAgent)
        vector_search_agent = Mock(spec=VectorSearchAgent)
        fact_checker_agent = Mock(spec=FactCheckerAgent)
        summarizer_agent = Mock(spec=SummarizerAgent)
        
        orchestrator = MainOrchestrator(
            router_agent=router_agent,
            web_search_agent=web_search_agent,
            web_scraper_agent=web_scraper_agent,
            vector_search_agent=vector_search_agent,
            fact_checker_agent=fact_checker_agent,
            summarizer_agent=summarizer_agent,
            config=mock_config
        )
        
        # Configure mixed success/failure
        router_agent.analyze_query.return_value = ResearchPlan(
            query_analysis=QueryAnalysis(
                topic_category="Test",
                information_type=InformationType.FACTUAL,
                complexity_level=ComplexityLevel.LOW,
                estimated_time_minutes=1
            ),
            research_strategy=ResearchStrategy(
                use_web_search=True,
                use_web_scraping=True,
                use_vector_search=True,
                priority_order=["web_search", "vector_search", "web_scraping"]
            ),
            search_queries=["test"],
            target_websites=["https://test.com"],
            expected_challenges=[]
        )
        
        # Web search succeeds
        web_search_agent.search.return_value = [
            SearchResult("Test", "https://test.com", "snippet", 8.0, "test")
        ]
        
        # Scraper and vector search fail
        web_scraper_agent.scrape_multiple_pages.side_effect = Exception("Scraping failed")
        vector_search_agent.search.side_effect = Exception("Vector search failed")
        
        # Fact checker and summarizer succeed with available data
        fact_checker_agent.check_facts.return_value = FactCheckResult(
            verified_facts=[{"fact": "Test", "confidence": 0.9, "sources": 1}],
            credibility_scores={},
            contradictions=[],
            cleaned_data=[]
        )
        
        summarizer_agent.generate_report.return_value = ResearchReport(
            executive_summary="Partial data summary",
            key_findings=["Finding from available data"],
            detailed_analysis="Analysis with partial data",
            sources=["[1] Test source"],
            recommendations=["Recommendation"],
            metadata={"word_count": 150}
        )
        
        # Execute workflow
        result = orchestrator.research("resilience test")
        
        # Verify system recovered and completed
        assert result.success is True
        assert result.source_count >= 1
        assert 'web_scraper' in result.metadata['failed_agents']
        assert 'vector_search' in result.metadata['failed_agents']
        assert 'web_search' in result.metadata['agents_used']

    def test_graceful_degradation_with_no_data(self, mock_config):
        """Test system handles graceful degradation when no data is collected"""
        # Create orchestrator with mock agents
        router_agent = Mock(spec=RouterAgent)
        web_search_agent = Mock(spec=WebSearchAgent)
        web_scraper_agent = Mock(spec=WebScraperAgent)
        vector_search_agent = Mock(spec=VectorSearchAgent)
        fact_checker_agent = Mock(spec=FactCheckerAgent)
        summarizer_agent = Mock(spec=SummarizerAgent)
        
        orchestrator = MainOrchestrator(
            router_agent=router_agent,
            web_search_agent=web_search_agent,
            web_scraper_agent=web_scraper_agent,
            vector_search_agent=vector_search_agent,
            fact_checker_agent=fact_checker_agent,
            summarizer_agent=summarizer_agent,
            config=mock_config
        )
        
        # Configure all data collection to fail
        router_agent.analyze_query.return_value = ResearchPlan(
            query_analysis=QueryAnalysis(
                topic_category="Test",
                information_type=InformationType.FACTUAL,
                complexity_level=ComplexityLevel.LOW,
                estimated_time_minutes=1
            ),
            research_strategy=ResearchStrategy(
                use_web_search=True,
                use_web_scraping=True,
                use_vector_search=True,
                priority_order=["web_search", "vector_search", "web_scraping"]
            ),
            search_queries=["test"],
            target_websites=["https://test.com"],
            expected_challenges=[]
        )
        
        # All data sources return empty
        web_search_agent.search.return_value = []
        web_scraper_agent.scrape_multiple_pages.return_value = []
        vector_search_agent.search.return_value = []
        
        # Execute workflow
        result = orchestrator.research("no data test")
        
        # Verify graceful failure
        assert result.success is False
        assert "No data collected" in result.error_message or len(result.metadata['errors']) > 0
        assert result.source_count == 0
    
    def test_health_check_functionality(self, mock_config):
        """Test system health check provides accurate status"""
        # Create orchestrator with mock agents
        router_agent = Mock(spec=RouterAgent)
        web_search_agent = Mock(spec=WebSearchAgent)
        web_scraper_agent = Mock(spec=WebScraperAgent)
        vector_search_agent = Mock(spec=VectorSearchAgent)
        fact_checker_agent = Mock(spec=FactCheckerAgent)
        summarizer_agent = Mock(spec=SummarizerAgent)
        
        # Configure health checks
        router_agent.health_check.return_value = {"status": "healthy"}
        web_search_agent.health_check.return_value = {"status": "healthy", "api_available": True}
        web_scraper_agent.health_check.return_value = {"status": "healthy"}
        vector_search_agent.health_check.return_value = {"status": "healthy", "db_connected": True}
        
        orchestrator = MainOrchestrator(
            router_agent=router_agent,
            web_search_agent=web_search_agent,
            web_scraper_agent=web_scraper_agent,
            vector_search_agent=vector_search_agent,
            fact_checker_agent=fact_checker_agent,
            summarizer_agent=summarizer_agent,
            config=mock_config
        )
        
        # Execute health check
        health_status = orchestrator.health_check()
        
        # Verify health check structure
        assert 'orchestrator' in health_status
        assert 'agents' in health_status
        assert 'executor' in health_status
        assert 'current_research_active' in health_status
        
        # Verify agent statuses
        assert 'router' in health_status['agents']
        assert 'web_search' in health_status['agents']
        assert health_status['agents']['web_search']['api_available'] is True
        assert health_status['agents']['vector_search']['db_connected'] is True


class TestCompleteSystemIntegration:
    """Integration tests for the complete system end-to-end"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        api_config = APIConfig()
        api_config.gemini_api_key = "test_key"
        api_config.serpapi_key = "test_key"
        api_config.google_sheets_credentials_path = None
        
        database_config = DatabaseConfig()
        scraping_config = ScrapingConfig()
        research_config = UtilsResearchConfig()
        sheets_config = SheetsConfig()
        
        return AppConfig(
            api=api_config,
            database=database_config,
            scraping=scraping_config,
            research=research_config,
            sheets=sheets_config
        )
    
    def test_full_system_workflow_with_all_components(self, mock_config):
        """Test complete system workflow with all components integrated"""
        # Create orchestrator with mock agents
        router_agent = Mock(spec=RouterAgent)
        web_search_agent = Mock(spec=WebSearchAgent)
        web_scraper_agent = Mock(spec=WebScraperAgent)
        vector_search_agent = Mock(spec=VectorSearchAgent)
        fact_checker_agent = Mock(spec=FactCheckerAgent)
        summarizer_agent = Mock(spec=SummarizerAgent)
        
        progress_updates = []
        
        def progress_callback(status: ProgressStatus):
            progress_updates.append(status)
        
        orchestrator = MainOrchestrator(
            router_agent=router_agent,
            web_search_agent=web_search_agent,
            web_scraper_agent=web_scraper_agent,
            vector_search_agent=vector_search_agent,
            fact_checker_agent=fact_checker_agent,
            summarizer_agent=summarizer_agent,
            config=mock_config,
            progress_callback=progress_callback
        )
        
        # Configure complete successful workflow
        router_agent.analyze_query.return_value = ResearchPlan(
            query_analysis=QueryAnalysis(
                topic_category="Technology",
                information_type=InformationType.ANALYTICAL,
                complexity_level=ComplexityLevel.HIGH,
                estimated_time_minutes=3
            ),
            research_strategy=ResearchStrategy(
                use_web_search=True,
                use_web_scraping=True,
                use_vector_search=True,
                priority_order=["web_search", "vector_search", "web_scraping"]
            ),
            search_queries=["quantum computing applications", "quantum algorithms"],
            target_websites=["https://quantum.org/research"],
            expected_challenges=["Complex technical content"]
        )
        
        web_search_agent.search.return_value = [
            SearchResult("Quantum Computing Advances", "https://quantum.org/advances", 
                        "Recent breakthroughs in quantum computing", 9.0, "serpapi"),
            SearchResult("Quantum Algorithms", "https://algorithms.com/quantum",
                        "New quantum algorithms for optimization", 8.5, "serpapi")
        ]
        
        web_scraper_agent.scrape_multiple_pages.return_value = [
            Mock(success=True, content=ScrapedContent(
                url="https://quantum.org/research",
                title="Quantum Research Overview",
                content="Comprehensive overview of quantum computing research and applications in cryptography and optimization.",
                author="Dr. Quantum",
                publish_date=datetime.now() - timedelta(days=5),
                extraction_method="beautifulsoup"
            ))
        ]
        
        vector_search_agent.search.return_value = [
            Document(
                content="Historical development of quantum computing from theoretical foundations to practical implementations.",
                metadata={'source_url': 'https://history.com/quantum', 'timestamp': datetime.now()},
                similarity_score=0.88,
                credibility_score=9.0
            )
        ]
        
        fact_checker_agent.check_facts.return_value = FactCheckResult(
            verified_facts=[
                {"fact": "Quantum computing shows promise for cryptography", "confidence": 0.92, "sources": 3},
                {"fact": "New quantum algorithms improve optimization", "confidence": 0.88, "sources": 2}
            ],
            credibility_scores={"quantum.org": 9.0, "algorithms.com": 8.5},
            contradictions=[],
            cleaned_data=["Quantum cryptography", "Quantum optimization"]
        )
        
        summarizer_agent.generate_report.return_value = ResearchReport(
            executive_summary="Quantum computing is advancing rapidly with significant applications in cryptography and optimization, supported by new algorithmic developments.",
            key_findings=[
                "Quantum computing breakthroughs enable new cryptographic methods",
                "Quantum algorithms significantly improve optimization problems",
                "Practical implementations are emerging from theoretical foundations"
            ],
            detailed_analysis="The research demonstrates that quantum computing has evolved from theoretical concepts to practical applications. Key developments include advances in quantum algorithms for optimization and cryptography. The field shows strong momentum with increasing real-world implementations.",
            sources=[
                "[1] Quantum Computing Advances - https://quantum.org/advances",
                "[2] Quantum Algorithms - https://algorithms.com/quantum",
                "[3] Quantum Research Overview - https://quantum.org/research"
            ],
            recommendations=[
                "Organizations should monitor quantum computing developments",
                "Investment in quantum-resistant cryptography is advisable",
                "Explore quantum optimization for complex problems"
            ],
            metadata={"word_count": 950, "confidence_level": "high", "technical_depth": "advanced"}
        )
        
        # Execute complete workflow
        query = "What are the latest developments in quantum computing?"
        config = ResearchConfig(
            max_sources=15,
            enable_web_scraping=True,
            enable_vector_search=True,
            report_style=ReportStyle.TECHNICAL,
            report_length=ReportLength.LONG,
            timeout_seconds=180
        )
        
        result = orchestrator.research(query, config)
        
        # Comprehensive verification
        assert result.success is True
        assert result.query == query
        assert result.source_count >= 3
        assert result.execution_time > 0
        
        # Verify report quality
        assert "quantum computing" in result.report.executive_summary.lower()
        assert len(result.report.key_findings) >= 3
        assert len(result.report.sources) >= 3
        assert len(result.report.recommendations) >= 2
        assert result.report.metadata["word_count"] >= 800
        
        # Verify all agents executed
        assert len(result.metadata['agents_used']) >= 5
        assert 'router' in result.metadata['agents_used']
        assert 'web_search' in result.metadata['agents_used']
        assert 'fact_checker' in result.metadata['agents_used']
        assert 'summarizer' in result.metadata['agents_used']
        
        # Verify progress tracking
        assert len(progress_updates) > 0
        stages = [update.current_step for update in progress_updates]
        assert 'planning' in stages
        assert 'data_collection' in stages
        
        # Verify data sources
        assert result.metadata['data_sources']['web_search_results'] >= 2
        assert result.metadata['data_sources']['scraped_pages'] >= 1
        assert result.metadata['data_sources']['vector_documents'] >= 1
        
        # Verify no critical errors
        assert len(result.metadata.get('failed_agents', [])) == 0
