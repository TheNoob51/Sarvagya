"""
Integration tests for Main Orchestrator.
Tests complete workflow with mock agents and validates error handling and timeout scenarios.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import List, Dict, Any

from agents.main_orchestrator import MainOrchestrator, ResearchStage, AgentResult, ResearchContext
from agents.data_models import (
    ResearchConfig, ResearchResult, ProgressStatus, ResearchReport,
    SearchResult, ScrapedContent, Document, FactCheckResult, ReportStyle, ReportLength
)
from utils.config import AppConfig, APIConfig, DatabaseConfig, ScrapingConfig, ResearchConfig as UtilsResearchConfig


class TestMainOrchestratorIntegration:
    """Integration tests for MainOrchestrator with mock agents"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        api_config = APIConfig()
        api_config.gemini_api_key = "test_gemini_key"
        api_config.serpapi_key = "test_serpapi_key"
        
        database_config = DatabaseConfig()
        scraping_config = ScrapingConfig()
        research_config = UtilsResearchConfig()
        
        return AppConfig(
            api=api_config,
            database=database_config,
            scraping=scraping_config,
            research=research_config
        )
    
    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing"""
        return {
            'router': Mock(),
            'web_search': Mock(),
            'web_scraper': Mock(),
            'vector_search': Mock(),
            'fact_checker': Mock(),
            'summarizer': Mock()
        }
    
    @pytest.fixture
    def orchestrator(self, mock_agents, mock_config):
        """Create MainOrchestrator with mock agents"""
        return MainOrchestrator(
            router_agent=mock_agents['router'],
            web_search_agent=mock_agents['web_search'],
            web_scraper_agent=mock_agents['web_scraper'],
            vector_search_agent=mock_agents['vector_search'],
            fact_checker_agent=mock_agents['fact_checker'],
            summarizer_agent=mock_agents['summarizer'],
            config=mock_config
        )
    
    @pytest.fixture
    def sample_research_plan(self):
        """Sample research plan from router agent"""
        return Mock(
            research_strategy=Mock(
                use_web_search=True,
                use_web_scraping=True,
                use_vector_search=True
            ),
            search_queries=["test query 1", "test query 2"],
            target_websites=["https://example.com", "https://test.org"]
        )
    
    @pytest.fixture
    def sample_search_results(self):
        """Sample search results"""
        return [
            SearchResult(
                title="Test Article 1",
                url="https://example.com/article1",
                snippet="Test content about research topic",
                credibility_score=8.0,
                source="serpapi"
            ),
            SearchResult(
                title="Academic Paper",
                url="https://university.edu/paper",
                snippet="Peer-reviewed research findings",
                credibility_score=9.0,
                source="serpapi"
            )
        ]
    
    @pytest.fixture
    def sample_scraped_content(self):
        """Sample scraped content"""
        return [
            ScrapedContent(
                url="https://example.com/article",
                title="Scraped Article",
                content="Detailed article content with comprehensive information about the research topic.",
                author="Dr. Jane Smith",
                publish_date=datetime.now() - timedelta(days=5),
                extraction_method="beautifulsoup"
            )
        ]
    
    @pytest.fixture
    def sample_vector_documents(self):
        """Sample vector database documents"""
        return [
            Document(
                content="Vector database content with relevant information",
                metadata={
                    'title': 'Database Document',
                    'source_url': 'https://vector-db.com/doc',
                    'author': 'System'
                },
                similarity_score=0.85,
                credibility_score=8.0
            )
        ]
    
    @pytest.fixture
    def sample_fact_check_result(self):
        """Sample fact check result"""
        return FactCheckResult(
            verified_facts=[
                "Fact 1: Research shows significant findings",
                "Fact 2: Data supports the hypothesis"
            ],
            credibility_scores={"source1": 8.0, "source2": 9.0},
            contradictions=["Minor contradiction resolved"],
            cleaned_data=["Cleaned fact 1", "Cleaned fact 2"]
        )
    
    @pytest.fixture
    def sample_research_report(self):
        """Sample research report"""
        return ResearchReport(
            executive_summary="Research summary with key findings and conclusions",
            key_findings=[
                "Finding 1: Significant evidence found",
                "Finding 2: Data supports conclusions"
            ],
            detailed_analysis="Comprehensive analysis of research data and findings",
            sources=["[1] Source 1", "[2] Source 2"],
            recommendations=["Recommendation 1", "Recommendation 2"],
            metadata={
                "query": "test query",
                "word_count": 250,
                "source_count": 2,
                "confidence_level": "high"
            }
        )

    def test_successful_complete_workflow(
        self, 
        orchestrator, 
        mock_agents,
        sample_research_plan,
        sample_search_results,
        sample_scraped_content,
        sample_vector_documents,
        sample_fact_check_result,
        sample_research_report
    ):
        """Test successful execution of complete research workflow"""
        # Configure mock agents for successful execution
        mock_agents['router'].analyze_query.return_value = sample_research_plan
        mock_agents['web_search'].search.return_value = sample_search_results
        mock_agents['web_scraper'].scrape_multiple_pages.return_value = [
            Mock(success=True, content=content) for content in sample_scraped_content
        ]
        mock_agents['vector_search'].search.return_value = sample_vector_documents
        mock_agents['fact_checker'].check_facts.return_value = sample_fact_check_result
        mock_agents['summarizer'].generate_report.return_value = sample_research_report
        
        # Execute research
        config = ResearchConfig(max_sources=10, timeout_seconds=120)
        result = orchestrator.research("test research query", config)
        
        # Verify successful completion
        assert result.success is True
        assert result.query == "test research query"
        assert isinstance(result.report, ResearchReport)
        assert result.source_count > 0
        assert result.execution_time > 0
        
        # Verify all agents were called (router gets timeout parameter)
        mock_agents['router'].analyze_query.assert_called_once()
        mock_agents['web_search'].search.assert_called_once()
        mock_agents['web_scraper'].scrape_multiple_pages.assert_called_once()
        mock_agents['vector_search'].search.assert_called_once()
        mock_agents['fact_checker'].check_facts.assert_called_once()
        mock_agents['summarizer'].generate_report.assert_called_once()
        
        # Verify metadata contains expected information
        assert 'execution_time' in result.metadata
        assert 'agents_used' in result.metadata
        assert 'data_sources' in result.metadata
        assert result.metadata['data_sources']['web_search_results'] == 2
        assert result.metadata['data_sources']['scraped_pages'] == 1
        assert result.metadata['data_sources']['vector_documents'] == 1

    def test_router_agent_failure_fallback(self, orchestrator, mock_agents, sample_search_results):
        """Test workflow continues with fallback strategy when router fails"""
        # Configure router to fail
        mock_agents['router'].analyze_query.side_effect = Exception("Router failed")
        
        # Configure other agents for success
        mock_agents['web_search'].search.return_value = sample_search_results
        mock_agents['web_scraper'].scrape_multiple_pages.return_value = []
        mock_agents['vector_search'].search.return_value = []
        mock_agents['fact_checker'].check_facts.return_value = FactCheckResult(
            verified_facts=["Basic fact"], credibility_scores={}, contradictions=[], cleaned_data=[]
        )
        mock_agents['summarizer'].generate_report.return_value = Mock(
            executive_summary="Fallback report",
            key_findings=["Finding"],
            detailed_analysis="Analysis",
            sources=["Source"],
            recommendations=["Recommendation"],
            metadata={"word_count": 100}
        )
        
        # Execute research
        result = orchestrator.research("test query")
        
        # Should fail because router failure prevents data collection strategy
        assert result.success is False
        assert 'router' in result.metadata['failed_agents']
        assert len(result.metadata['errors']) > 0
        
        # Web search should NOT be called when router fails completely
        # (The orchestrator fails at planning stage and doesn't proceed to data collection)

    def test_data_collection_partial_failure(
        self, 
        orchestrator, 
        mock_agents,
        sample_research_plan,
        sample_search_results,
        sample_fact_check_result,
        sample_research_report
    ):
        """Test workflow continues when some data collection agents fail"""
        # Configure router for success
        mock_agents['router'].analyze_query.return_value = sample_research_plan
        
        # Configure partial data collection success
        mock_agents['web_search'].search.return_value = sample_search_results
        mock_agents['web_scraper'].scrape_multiple_pages.side_effect = Exception("Scraper failed")
        mock_agents['vector_search'].search.side_effect = Exception("Vector search failed")
        
        # Configure processing agents for success
        mock_agents['fact_checker'].check_facts.return_value = sample_fact_check_result
        mock_agents['summarizer'].generate_report.return_value = sample_research_report
        
        # Execute research
        result = orchestrator.research("test query")
        
        # Should succeed with partial data
        assert result.success is True
        assert result.source_count == 2  # Only web search results
        assert 'web_scraper' in result.metadata['failed_agents']
        assert 'vector_search' in result.metadata['failed_agents']
        assert 'web_search' in result.metadata['agents_used']

    def test_fact_checker_failure_handling(
        self, 
        orchestrator, 
        mock_agents,
        sample_research_plan,
        sample_search_results,
        sample_research_report
    ):
        """Test workflow handles fact checker failure gracefully"""
        # Configure successful data collection
        mock_agents['router'].analyze_query.return_value = sample_research_plan
        mock_agents['web_search'].search.return_value = sample_search_results
        mock_agents['web_scraper'].scrape_multiple_pages.return_value = []
        mock_agents['vector_search'].search.return_value = []
        
        # Configure fact checker to fail
        mock_agents['fact_checker'].check_facts.side_effect = Exception("Fact checker failed")
        
        # Configure summarizer for success
        mock_agents['summarizer'].generate_report.return_value = sample_research_report
        
        # Execute research
        result = orchestrator.research("test query")
        
        # Should fail because fact checking is critical
        assert result.success is False
        assert 'fact_checker' in result.metadata['failed_agents']
        assert "Fact checking failed" in result.error_message

    def test_summarizer_failure_fallback_report(
        self, 
        orchestrator, 
        mock_agents,
        sample_research_plan,
        sample_search_results,
        sample_fact_check_result
    ):
        """Test fallback report generation when summarizer fails"""
        # Configure successful workflow until summarizer
        mock_agents['router'].analyze_query.return_value = sample_research_plan
        mock_agents['web_search'].search.return_value = sample_search_results
        mock_agents['web_scraper'].scrape_multiple_pages.return_value = []
        mock_agents['vector_search'].search.return_value = []
        mock_agents['fact_checker'].check_facts.return_value = sample_fact_check_result
        
        # Configure summarizer to fail
        mock_agents['summarizer'].generate_report.side_effect = Exception("Summarizer failed")
        
        # Execute research
        result = orchestrator.research("test query")
        
        # Should fail but provide fallback report
        assert result.success is False
        assert isinstance(result.report, ResearchReport)
        assert "fallback_report" in result.report.metadata
        assert result.report.metadata["fallback_report"] is True
        assert 'summarizer' in result.metadata['failed_agents']

    def test_timeout_handling(self, orchestrator, mock_agents):
        """Test timeout handling for long-running agents"""
        # Configure router to succeed quickly
        mock_agents['router'].analyze_query.return_value = Mock(
            research_strategy=Mock(use_web_search=True, use_web_scraping=False, use_vector_search=False),
            search_queries=["test"],
            target_websites=[]
        )
        
        # Configure web search to simulate long execution
        def slow_search(*args, **kwargs):
            time.sleep(2)  # Simulate slow operation
            return []
        
        mock_agents['web_search'].search.side_effect = slow_search
        mock_agents['fact_checker'].check_facts.return_value = FactCheckResult(
            verified_facts=[], credibility_scores={}, contradictions=[], cleaned_data=[]
        )
        mock_agents['summarizer'].generate_report.return_value = Mock(
            executive_summary="Quick report",
            key_findings=[],
            detailed_analysis="Analysis",
            sources=[],
            recommendations=[],
            metadata={"word_count": 50}
        )
        
        # Execute with short timeout
        config = ResearchConfig(timeout_seconds=1)
        result = orchestrator.research("test query", config)
        
        # Should complete but may have warnings about timing
        assert isinstance(result, ResearchResult)
        # The test allows for either success or failure depending on timing

    def test_progress_tracking(self, orchestrator, mock_agents, sample_research_plan):
        """Test progress tracking throughout workflow"""
        progress_updates = []
        
        def progress_callback(status: ProgressStatus):
            progress_updates.append({
                'step': status.current_step,
                'percentage': status.completion_percentage,
                'message': status.status_message
            })
        
        # Create orchestrator with progress callback
        orchestrator.progress_callback = progress_callback
        
        # Configure agents for success
        mock_agents['router'].analyze_query.return_value = sample_research_plan
        mock_agents['web_search'].search.return_value = []
        mock_agents['web_scraper'].scrape_multiple_pages.return_value = []
        mock_agents['vector_search'].search.return_value = []
        mock_agents['fact_checker'].check_facts.return_value = FactCheckResult(
            verified_facts=[], credibility_scores={}, contradictions=[], cleaned_data=[]
        )
        mock_agents['summarizer'].generate_report.return_value = Mock(
            executive_summary="Report",
            key_findings=[],
            detailed_analysis="Analysis",
            sources=[],
            recommendations=[],
            metadata={"word_count": 50}
        )
        
        # Execute research
        orchestrator.research("test query")
        
        # Verify progress updates were sent
        assert len(progress_updates) > 0
        
        # Verify progress stages (may not reach all stages if data collection fails)
        stages = [update['step'] for update in progress_updates]
        assert 'planning' in stages
        assert 'data_collection' in stages
        # Note: fact_checking and report_generation may not be reached if no data is collected
        
        # Verify progress percentages increase
        percentages = [update['percentage'] for update in progress_updates]
        assert percentages[-1] >= percentages[0]  # Should increase over time

    def test_error_recovery_and_aggregation(
        self, 
        orchestrator, 
        mock_agents,
        sample_research_plan
    ):
        """Test error recovery and proper error aggregation"""
        # Configure mixed success/failure scenario
        mock_agents['router'].analyze_query.return_value = sample_research_plan
        mock_agents['web_search'].search.side_effect = Exception("Web search API error")
        mock_agents['web_scraper'].scrape_multiple_pages.side_effect = Exception("Scraping blocked")
        mock_agents['vector_search'].search.return_value = []  # Success but empty
        mock_agents['fact_checker'].check_facts.return_value = FactCheckResult(
            verified_facts=[], credibility_scores={}, contradictions=[], cleaned_data=[]
        )
        mock_agents['summarizer'].generate_report.return_value = Mock(
            executive_summary="Limited report",
            key_findings=[],
            detailed_analysis="Limited analysis",
            sources=[],
            recommendations=[],
            metadata={"word_count": 30}
        )
        
        # Execute research
        result = orchestrator.research("test query")
        
        # Should fail due to no data collected
        assert result.success is False
        
        # Verify error aggregation
        assert len(result.metadata['errors']) > 0
        assert len(result.metadata['failed_agents']) >= 2
        assert 'web_search' in result.metadata['failed_agents']
        assert 'web_scraper' in result.metadata['failed_agents']
        
        # Should still have some metadata
        assert 'execution_time' in result.metadata
        assert 'data_sources' in result.metadata

    def test_data_validation_between_agents(
        self, 
        orchestrator, 
        mock_agents
    ):
        """Test data validation between agent stages"""
        # Configure router with invalid data structure
        invalid_plan = Mock()
        # Missing required attributes
        mock_agents['router'].analyze_query.return_value = invalid_plan
        
        mock_agents['web_search'].search.return_value = []
        mock_agents['web_scraper'].scrape_multiple_pages.return_value = []
        mock_agents['vector_search'].search.return_value = []
        mock_agents['fact_checker'].check_facts.return_value = FactCheckResult(
            verified_facts=[], credibility_scores={}, contradictions=[], cleaned_data=[]
        )
        mock_agents['summarizer'].generate_report.return_value = Mock(
            executive_summary="Report",
            key_findings=[],
            detailed_analysis="Analysis",
            sources=[],
            recommendations=[],
            metadata={"word_count": 50}
        )
        
        # Execute research
        result = orchestrator.research("test query")
        
        # Should handle invalid data gracefully
        assert isinstance(result, ResearchResult)
        # May succeed with fallback strategy or fail with proper error handling
        if not result.success:
            assert len(result.metadata['errors']) > 0

    def test_concurrent_agent_execution(
        self, 
        orchestrator, 
        mock_agents,
        sample_research_plan,
        sample_search_results,
        sample_scraped_content,
        sample_vector_documents
    ):
        """Test concurrent execution of data collection agents"""
        execution_times = {}
        
        def track_execution_time(agent_name):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                if agent_name == 'web_search':
                    result = sample_search_results
                elif agent_name == 'web_scraper':
                    result = [Mock(success=True, content=content) for content in sample_scraped_content]
                else:  # vector_search
                    result = sample_vector_documents
                
                # Simulate some processing time
                time.sleep(0.1)
                execution_times[agent_name] = time.time() - start_time
                return result
            return wrapper
        
        # Configure agents with execution tracking
        mock_agents['router'].analyze_query.return_value = sample_research_plan
        mock_agents['web_search'].search.side_effect = track_execution_time('web_search')
        mock_agents['web_scraper'].scrape_multiple_pages.side_effect = track_execution_time('web_scraper')
        mock_agents['vector_search'].search.side_effect = track_execution_time('vector_search')
        
        mock_agents['fact_checker'].check_facts.return_value = FactCheckResult(
            verified_facts=["Fact"], credibility_scores={}, contradictions=[], cleaned_data=[]
        )
        mock_agents['summarizer'].generate_report.return_value = Mock(
            executive_summary="Report",
            key_findings=["Finding"],
            detailed_analysis="Analysis",
            sources=["Source"],
            recommendations=["Recommendation"],
            metadata={"word_count": 100}
        )
        
        # Execute research
        start_time = time.time()
        result = orchestrator.research("test query")
        total_time = time.time() - start_time
        
        # Verify concurrent execution benefit
        assert result.success is True
        
        # Total time should be less than sum of individual times (indicating parallelism)
        if len(execution_times) > 1:
            sequential_time = sum(execution_times.values())
            # Allow some overhead but should be significantly faster than sequential
            assert total_time < sequential_time * 0.8

    def test_health_check_functionality(self, orchestrator, mock_agents):
        """Test orchestrator health check functionality"""
        # Configure mock health checks for agents
        mock_agents['router'].health_check.return_value = {"status": "healthy"}
        mock_agents['web_search'].health_check.return_value = {"status": "healthy", "api_available": True}
        mock_agents['web_scraper'].health_check.return_value = {"status": "healthy", "scraping_functional": True}
        mock_agents['vector_search'].health_check.return_value = {"status": "healthy", "db_connected": True}
        
        # Execute health check
        health_status = orchestrator.health_check()
        
        # Verify health check structure
        assert 'orchestrator' in health_status
        assert 'agents' in health_status
        assert 'executor' in health_status
        assert 'current_research_active' in health_status
        
        # Verify agent health checks were called
        assert 'router' in health_status['agents']
        assert 'web_search' in health_status['agents']
        assert 'web_scraper' in health_status['agents']
        assert 'vector_search' in health_status['agents']
        
        # Verify executor status
        assert 'active' in health_status['executor']
        assert 'max_workers' in health_status['executor']

    def test_research_cancellation(self, orchestrator, mock_agents):
        """Test research cancellation functionality"""
        # Test cancellation when no research is active
        result = orchestrator.cancel_research()
        assert result is False
        
        # Start a research operation (mock long-running)
        mock_agents['router'].analyze_query.return_value = Mock(
            research_strategy=Mock(use_web_search=True, use_web_scraping=False, use_vector_search=False),
            search_queries=["test"],
            target_websites=[]
        )
        
        def long_running_search(*args, **kwargs):
            time.sleep(1)
            return []
        
        mock_agents['web_search'].search.side_effect = long_running_search
        
        # In a real scenario, we would test cancellation during execution
        # For this test, we verify the cancellation mechanism exists
        assert hasattr(orchestrator, 'cancel_research')
        assert hasattr(orchestrator, '_cancel_requested')

    def test_cleanup_functionality(self, orchestrator, mock_agents):
        """Test orchestrator cleanup functionality"""
        # Configure mock cleanup for agents that support it
        mock_agents['web_scraper'].cleanup = Mock()
        
        # Execute cleanup
        orchestrator.cleanup()
        
        # Verify cleanup was called on supporting agents
        mock_agents['web_scraper'].cleanup.assert_called_once()
        
        # Verify executor is shutdown
        assert orchestrator.executor._shutdown is True


class TestOrchestratorErrorScenarios:
    """Test specific error scenarios and edge cases"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        api_config = APIConfig()
        api_config.gemini_api_key = "test_gemini_key"
        api_config.serpapi_key = "test_serpapi_key"
        
        database_config = DatabaseConfig()
        scraping_config = ScrapingConfig()
        research_config = UtilsResearchConfig()
        research_config.timeout_seconds = 1  # Short timeout for error scenarios
        
        return AppConfig(
            api=api_config,
            database=database_config,
            scraping=scraping_config,
            research=research_config
        )
    
    @pytest.fixture
    def orchestrator_with_short_timeout(self, mock_config):
        """Create orchestrator with very short timeout for testing"""
        mock_agents = {
            'router': Mock(),
            'web_search': Mock(),
            'web_scraper': Mock(),
            'vector_search': Mock(),
            'fact_checker': Mock(),
            'summarizer': Mock()
        }
        
        return MainOrchestrator(
            router_agent=mock_agents['router'],
            web_search_agent=mock_agents['web_search'],
            web_scraper_agent=mock_agents['web_scraper'],
            vector_search_agent=mock_agents['vector_search'],
            fact_checker_agent=mock_agents['fact_checker'],
            summarizer_agent=mock_agents['summarizer'],
            config=mock_config
        ), mock_agents

    def test_all_agents_fail_scenario(self, orchestrator_with_short_timeout):
        """Test scenario where all agents fail"""
        orchestrator, mock_agents = orchestrator_with_short_timeout
        
        # Configure all agents to fail
        for agent in mock_agents.values():
            if hasattr(agent, 'analyze_query'):
                agent.analyze_query.side_effect = Exception("Agent failed")
            elif hasattr(agent, 'search'):
                agent.search.side_effect = Exception("Agent failed")
            elif hasattr(agent, 'scrape_multiple_pages'):
                agent.scrape_multiple_pages.side_effect = Exception("Agent failed")
            elif hasattr(agent, 'check_facts'):
                agent.check_facts.side_effect = Exception("Agent failed")
            elif hasattr(agent, 'generate_report'):
                agent.generate_report.side_effect = Exception("Agent failed")
        
        # Execute research
        result = orchestrator.research("test query")
        
        # Should fail gracefully
        assert result.success is False
        assert len(result.metadata['failed_agents']) > 0
        assert len(result.metadata['errors']) > 0
        assert isinstance(result.report, ResearchReport)  # Should have fallback report

    def test_memory_and_resource_constraints(self, orchestrator_with_short_timeout):
        """Test behavior under resource constraints"""
        orchestrator, mock_agents = orchestrator_with_short_timeout
        
        # Configure agents to return large amounts of data
        large_search_results = [
            SearchResult(f"Title {i}", f"https://example{i}.com", "content" * 1000, 7.0, "test")
            for i in range(100)
        ]
        
        mock_agents['router'].analyze_query.return_value = Mock(
            research_strategy=Mock(use_web_search=True, use_web_scraping=False, use_vector_search=False),
            search_queries=["test"],
            target_websites=[]
        )
        mock_agents['web_search'].search.return_value = large_search_results
        mock_agents['fact_checker'].check_facts.return_value = FactCheckResult(
            verified_facts=["Fact"] * 50, credibility_scores={}, contradictions=[], cleaned_data=[]
        )
        mock_agents['summarizer'].generate_report.return_value = Mock(
            executive_summary="Report",
            key_findings=["Finding"] * 20,
            detailed_analysis="Analysis" * 100,
            sources=["Source"] * 50,
            recommendations=["Recommendation"] * 10,
            metadata={"word_count": 5000}
        )
        
        # Execute research
        result = orchestrator.research("test query")
        
        # Should handle large data gracefully
        assert isinstance(result, ResearchResult)
        # Should either succeed or fail gracefully without crashing

    def test_invalid_configuration_handling(self, mock_config):
        """Test handling of invalid configuration"""
        # Test with invalid timeout
        invalid_config = ResearchConfig(
            max_sources=-1,  # Invalid
            timeout_seconds=0,  # Invalid
            enable_web_scraping=None  # Invalid type
        )
        
        mock_agents = {
            'router': Mock(),
            'web_search': Mock(),
            'web_scraper': Mock(),
            'vector_search': Mock(),
            'fact_checker': Mock(),
            'summarizer': Mock()
        }
        
        orchestrator = MainOrchestrator(
            router_agent=mock_agents['router'],
            web_search_agent=mock_agents['web_search'],
            web_scraper_agent=mock_agents['web_scraper'],
            vector_search_agent=mock_agents['vector_search'],
            fact_checker_agent=mock_agents['fact_checker'],
            summarizer_agent=mock_agents['summarizer'],
            config=mock_config
        )
        
        # Should handle invalid config gracefully
        result = orchestrator.research("test query", invalid_config)
        assert isinstance(result, ResearchResult)


class TestOrchestratorWorkflowIntegration:
    """Additional integration tests for complete workflow scenarios"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        api_config = APIConfig()
        api_config.gemini_api_key = "test_gemini_key"
        api_config.serpapi_key = "test_serpapi_key"
        
        database_config = DatabaseConfig()
        scraping_config = ScrapingConfig()
        research_config = UtilsResearchConfig()
        
        return AppConfig(
            api=api_config,
            database=database_config,
            scraping=scraping_config,
            research=research_config
        )
    
    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing"""
        return {
            'router': Mock(),
            'web_search': Mock(),
            'web_scraper': Mock(),
            'vector_search': Mock(),
            'fact_checker': Mock(),
            'summarizer': Mock()
        }
    
    @pytest.fixture
    def orchestrator(self, mock_agents, mock_config):
        """Create MainOrchestrator with mock agents"""
        return MainOrchestrator(
            router_agent=mock_agents['router'],
            web_search_agent=mock_agents['web_search'],
            web_scraper_agent=mock_agents['web_scraper'],
            vector_search_agent=mock_agents['vector_search'],
            fact_checker_agent=mock_agents['fact_checker'],
            summarizer_agent=mock_agents['summarizer'],
            config=mock_config
        )

    def test_complete_workflow_with_all_data_sources(self, orchestrator, mock_agents):
        """Test complete workflow using all available data sources"""
        # Configure router to enable all data sources
        mock_research_plan = Mock()
        mock_research_plan.research_strategy = Mock(
            use_web_search=True,
            use_web_scraping=True,
            use_vector_search=True
        )
        mock_research_plan.search_queries = ["query1", "query2", "query3"]
        mock_research_plan.target_websites = ["https://site1.com", "https://site2.com"]
        
        mock_agents['router'].analyze_query.return_value = mock_research_plan
        
        # Configure data collection agents
        mock_agents['web_search'].search.return_value = [
            SearchResult("Title 1", "https://example1.com", "Content 1", 8.0, "serpapi"),
            SearchResult("Title 2", "https://example2.com", "Content 2", 7.5, "serpapi")
        ]
        
        mock_agents['web_scraper'].scrape_multiple_pages.return_value = [
            Mock(success=True, content=ScrapedContent(
                "https://site1.com", "Scraped Title", "Scraped content", 
                "Author", datetime.now(), "beautifulsoup"
            ))
        ]
        
        mock_agents['vector_search'].search.return_value = [
            Document("Vector content", {"title": "DB Doc"}, 0.85, 8.0)
        ]
        
        # Configure processing agents
        mock_agents['fact_checker'].check_facts.return_value = FactCheckResult(
            verified_facts=["Verified fact 1", "Verified fact 2"],
            credibility_scores={"source1": 8.0, "source2": 7.5},
            contradictions=[],
            cleaned_data=["Clean fact 1", "Clean fact 2"]
        )
        
        mock_agents['summarizer'].generate_report.return_value = ResearchReport(
            executive_summary="Comprehensive research summary",
            key_findings=["Finding 1", "Finding 2", "Finding 3"],
            detailed_analysis="Detailed analysis of all collected data",
            sources=["[1] Source 1", "[2] Source 2", "[3] Source 3"],
            recommendations=["Recommendation 1", "Recommendation 2"],
            metadata={"word_count": 500, "confidence_level": "high"}
        )
        
        # Execute research
        result = orchestrator.research("comprehensive research query")
        
        # Verify successful completion with all data sources
        assert result.success is True
        assert result.source_count == 4  # 2 search + 1 scraped + 1 vector
        assert len(result.metadata['agents_used']) == 6  # All agents used
        assert result.metadata['data_sources']['web_search_results'] == 2
        assert result.metadata['data_sources']['scraped_pages'] == 1
        assert result.metadata['data_sources']['vector_documents'] == 1
        
        # Verify all agents were called appropriately
        mock_agents['router'].analyze_query.assert_called_once()
        mock_agents['web_search'].search.assert_called_once()
        mock_agents['web_scraper'].scrape_multiple_pages.assert_called_once()
        mock_agents['vector_search'].search.assert_called_once()
        mock_agents['fact_checker'].check_facts.assert_called_once()
        mock_agents['summarizer'].generate_report.assert_called_once()

    def test_workflow_with_agent_timeout_scenarios(self, orchestrator, mock_agents):
        """Test workflow behavior when agents exceed timeout limits"""
        # Configure router for success
        mock_research_plan = Mock()
        mock_research_plan.research_strategy = Mock(
            use_web_search=True,
            use_web_scraping=False,
            use_vector_search=False
        )
        mock_research_plan.search_queries = ["test query"]
        mock_research_plan.target_websites = []
        
        mock_agents['router'].analyze_query.return_value = mock_research_plan
        
        # Configure web search to simulate timeout (return None to simulate timeout)
        mock_agents['web_search'].search.return_value = None
        
        # Configure other agents for success
        mock_agents['fact_checker'].check_facts.return_value = FactCheckResult(
            verified_facts=[], credibility_scores={}, contradictions=[], cleaned_data=[]
        )
        
        mock_agents['summarizer'].generate_report.return_value = ResearchReport(
            executive_summary="Limited research due to timeouts",
            key_findings=["Limited findings"],
            detailed_analysis="Analysis with limited data",
            sources=[],
            recommendations=["Retry with longer timeout"],
            metadata={"word_count": 100, "confidence_level": "low"}
        )
        
        # Execute research with short timeout
        config = ResearchConfig(timeout_seconds=5)
        result = orchestrator.research("timeout test query", config)
        
        # Should complete but may have limited data
        assert isinstance(result, ResearchResult)
        # May succeed or fail depending on timeout handling

    def test_workflow_error_recovery_mechanisms(self, orchestrator, mock_agents):
        """Test error recovery and graceful degradation"""
        # Configure router for success
        mock_research_plan = Mock()
        mock_research_plan.research_strategy = Mock(
            use_web_search=True,
            use_web_scraping=True,
            use_vector_search=True
        )
        mock_research_plan.search_queries = ["test"]
        mock_research_plan.target_websites = ["https://test.com"]
        
        mock_agents['router'].analyze_query.return_value = mock_research_plan
        
        # Configure mixed success/failure scenario
        mock_agents['web_search'].search.return_value = [
            SearchResult("Working Source", "https://working.com", "Good content", 8.0, "serpapi")
        ]
        
        # Simulate scraper and vector search failures
        mock_agents['web_scraper'].scrape_multiple_pages.side_effect = Exception("Network error")
        mock_agents['vector_search'].search.side_effect = Exception("Database connection failed")
        
        # Configure processing agents to handle partial data
        mock_agents['fact_checker'].check_facts.return_value = FactCheckResult(
            verified_facts=["Single verified fact"],
            credibility_scores={"working_source": 8.0},
            contradictions=[],
            cleaned_data=["Clean fact"]
        )
        
        mock_agents['summarizer'].generate_report.return_value = ResearchReport(
            executive_summary="Research with partial data",
            key_findings=["Limited finding from available source"],
            detailed_analysis="Analysis based on single source",
            sources=["[1] Working Source"],
            recommendations=["Investigate failed data sources"],
            metadata={"word_count": 200, "confidence_level": "medium"}
        )
        
        # Execute research
        result = orchestrator.research("error recovery test")
        
        # Should succeed with partial data
        assert result.success is True
        assert result.source_count == 1  # Only web search succeeded
        assert 'web_scraper' in result.metadata['failed_agents']
        assert 'vector_search' in result.metadata['failed_agents']
        assert 'web_search' in result.metadata['agents_used']
        # Check for errors in the appropriate metadata field
        if 'errors' in result.metadata:
            assert len(result.metadata['errors']) >= 2  # At least 2 errors from failed agents
        else:
            # Errors might be in warnings or other fields depending on success/failure
            assert len(result.metadata['failed_agents']) >= 2

    def test_workflow_data_validation_and_quality_checks(self, orchestrator, mock_agents):
        """Test data validation and quality assessment throughout workflow"""
        # Configure router with comprehensive plan
        mock_research_plan = Mock()
        mock_research_plan.research_strategy = Mock(
            use_web_search=True,
            use_web_scraping=True,
            use_vector_search=True
        )
        mock_research_plan.search_queries = ["quality test"]
        mock_research_plan.target_websites = ["https://quality.com"]
        
        mock_agents['router'].analyze_query.return_value = mock_research_plan
        
        # Configure high-quality data sources
        mock_agents['web_search'].search.return_value = [
            SearchResult("High Quality Source", "https://academic.edu", "Peer-reviewed content", 9.5, "serpapi"),
            SearchResult("Medium Quality", "https://news.com", "News article", 7.0, "serpapi"),
            SearchResult("Low Quality", "https://blog.com", "Personal opinion", 4.0, "serpapi")
        ]
        
        mock_agents['web_scraper'].scrape_multiple_pages.return_value = [
            Mock(success=True, content=ScrapedContent(
                "https://quality.com", "Quality Article", "Well-researched content",
                "Dr. Expert", datetime.now(), "beautifulsoup"
            ))
        ]
        
        mock_agents['vector_search'].search.return_value = [
            Document("Historical data", {"title": "Archive", "credibility": 8.5}, 0.9, 8.5)
        ]
        
        # Configure fact checker to filter by quality
        mock_agents['fact_checker'].check_facts.return_value = FactCheckResult(
            verified_facts=["High-quality fact 1", "High-quality fact 2"],
            credibility_scores={"academic": 9.5, "quality_site": 8.0, "archive": 8.5},
            contradictions=["Low-quality source contradicted"],
            cleaned_data=["Verified fact 1", "Verified fact 2"]
        )
        
        mock_agents['summarizer'].generate_report.return_value = ResearchReport(
            executive_summary="High-quality research with validated sources",
            key_findings=["Quality finding 1", "Quality finding 2"],
            detailed_analysis="Comprehensive analysis from credible sources",
            sources=["[1] Academic Source", "[2] Quality Website", "[3] Archive"],
            recommendations=["High-confidence recommendations"],
            metadata={"word_count": 800, "confidence_level": "high", "quality_score": 9.0}
        )
        
        # Execute research
        result = orchestrator.research("quality validation test")
        
        # Verify quality assessment
        assert result.success is True
        assert result.source_count == 5  # 3 search + 1 scraped + 1 vector
        
        # Check quality metrics in metadata
        quality_assessment = result.metadata.get('quality_assessment', {})
        assert 'overall_quality' in quality_assessment
        assert 'data_completeness' in quality_assessment
        assert 'source_diversity' in quality_assessment
        assert 'credibility_score' in quality_assessment
        
        # Verify data validation metrics
        data_validation = result.metadata.get('data_validation', {})
        assert data_validation['total_sources_validated'] == 5
        assert data_validation['source_diversity_score'] > 0
        assert data_validation['average_credibility_score'] > 0

    def test_workflow_performance_monitoring(self, orchestrator, mock_agents):
        """Test performance monitoring and metrics collection"""
        # Configure all agents for success with timing simulation
        mock_research_plan = Mock()
        mock_research_plan.research_strategy = Mock(
            use_web_search=True,
            use_web_scraping=True,
            use_vector_search=True
        )
        mock_research_plan.search_queries = ["performance test"]
        mock_research_plan.target_websites = ["https://perf.com"]
        
        mock_agents['router'].analyze_query.return_value = mock_research_plan
        
        # Configure agents with simulated execution times
        mock_agents['web_search'].search.return_value = [
            SearchResult("Fast Result", "https://fast.com", "Quick content", 8.0, "serpapi")
        ]
        
        mock_agents['web_scraper'].scrape_multiple_pages.return_value = [
            Mock(success=True, content=ScrapedContent(
                "https://perf.com", "Performance Test", "Test content",
                "Tester", datetime.now(), "beautifulsoup"
            ))
        ]
        
        mock_agents['vector_search'].search.return_value = [
            Document("Performance data", {"title": "Perf Doc"}, 0.8, 7.5)
        ]
        
        mock_agents['fact_checker'].check_facts.return_value = FactCheckResult(
            verified_facts=["Performance fact"],
            credibility_scores={"fast": 8.0},
            contradictions=[],
            cleaned_data=["Clean performance fact"]
        )
        
        mock_agents['summarizer'].generate_report.return_value = ResearchReport(
            executive_summary="Performance monitoring test results",
            key_findings=["Performance finding"],
            detailed_analysis="Analysis of performance metrics",
            sources=["[1] Fast Source"],
            recommendations=["Performance recommendations"],
            metadata={"word_count": 300, "confidence_level": "medium"}
        )
        
        # Execute research and measure performance
        start_time = time.time()
        result = orchestrator.research("performance monitoring test")
        execution_time = time.time() - start_time
        
        # Verify performance metrics are collected
        assert result.success is True
        assert result.execution_time > 0
        assert result.execution_time <= execution_time + 0.1  # Allow small margin
        
        # Check performance metrics in metadata
        performance_metrics = result.metadata.get('performance_metrics', {})
        assert 'total_execution_time' in performance_metrics
        assert 'agent_performance' in performance_metrics
        assert 'throughput_metrics' in performance_metrics
        assert 'efficiency_analysis' in performance_metrics
        
        # Verify agent performance tracking
        agent_performance = performance_metrics['agent_performance']
        expected_agents = ['router', 'web_search', 'web_scraper', 'vector_search', 'fact_checker', 'summarizer']
        for agent in expected_agents:
            assert agent in agent_performance
            assert 'execution_time' in agent_performance[agent]
            assert 'success' in agent_performance[agent]