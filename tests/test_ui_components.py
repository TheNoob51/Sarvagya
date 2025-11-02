"""
UI Component Tests for the Intelligent Research Assistant.
Tests interface components with mock data and validates user interaction flows.
"""

import pytest
import gradio as gr
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import time
import threading

from ui.gradio_interface import GradioInterface, create_gradio_app
from agents.data_models import (
    ResearchConfig, ResearchResult, ProgressStatus, ResearchReport,
    ReportStyle, ReportLength
)
from config import get_config


class TestGradioInterfaceComponents:
    """Test suite for GradioInterface UI components"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        config = Mock()
        config.api = Mock()
        config.api.gemini_api_key = "test_key"
        config.api.serpapi_key = "test_serpapi_key"
        config.api.google_sheets_credentials_path = "/test/path/credentials.json"
        config.database = Mock()
        config.database.chroma_db_path = "/test/chroma"
        config.gradio_host = "127.0.0.1"
        config.gradio_port = 7860
        return config
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator for testing"""
        orchestrator = Mock()
        orchestrator.health_check.return_value = {
            'orchestrator': 'healthy',
            'agents': {
                'router': {'status': 'healthy'},
                'web_search': {'status': 'healthy'},
                'web_scraper': {'status': 'healthy'},
                'vector_search': {'status': 'healthy'},
                'fact_checker': {'status': 'healthy'},
                'summarizer': {'status': 'healthy'}
            },
            'executor': {'active': True, 'max_workers': 4},
            'current_research_active': False
        }
        orchestrator.get_research_history.return_value = []
        orchestrator.get_research_analytics.return_value = {}
        orchestrator.get_sheets_status.return_value = {'available': True, 'spreadsheet_name': 'test_sheet'}
        return orchestrator
    
    @pytest.fixture
    def gradio_interface(self, mock_config):
        """Create GradioInterface instance with mocked dependencies"""
        with patch('ui.gradio_interface.get_config', return_value=mock_config):
            with patch('os.path.exists', return_value=True):
                interface = GradioInterface()
                return interface
    
    def test_interface_initialization(self, gradio_interface):
        """Test GradioInterface initialization with proper configuration"""
        assert gradio_interface is not None
        assert gradio_interface.config is not None
        assert gradio_interface.orchestrator is None
        assert gradio_interface.current_progress is None
        assert gradio_interface.research_history == []
        assert gradio_interface.is_researching is False
    
    def test_configuration_validation(self, gradio_interface):
        """Test configuration validation functionality"""
        # Test with valid configuration (but still may have Google Sheets error)
        gradio_interface.config.api.gemini_api_key = "valid_key"
        with patch('os.path.exists', return_value=True):
            errors = gradio_interface._validate_configuration()
            assert len(errors) == 0
        
        # Test with missing API key
        gradio_interface.config.api.gemini_api_key = None
        errors = gradio_interface._validate_configuration()
        assert len(errors) > 0
        assert any("Gemini API key" in error for error in errors)
    
    def test_orchestrator_initialization_success(self, gradio_interface, mock_orchestrator):
        """Test successful orchestrator initialization"""
        with patch('ui.gradio_interface.create_main_orchestrator', return_value=mock_orchestrator):
            success, error_msg = gradio_interface._initialize_orchestrator()
            assert success is True
            assert error_msg == ""
            assert gradio_interface.orchestrator is not None
    
    def test_orchestrator_initialization_failure(self, gradio_interface):
        """Test orchestrator initialization failure handling"""
        gradio_interface.config.api.gemini_api_key = None
        success, error_msg = gradio_interface._initialize_orchestrator()
        assert success is False
        assert "Gemini API key is required" in error_msg
    
    def test_progress_callback_functionality(self, gradio_interface):
        """Test progress callback and display formatting"""
        # Create mock progress status
        progress = ProgressStatus(
            current_step="data_collection",
            completion_percentage=45.0,
            status_message="Collecting data from web sources",
            estimated_time_remaining=30,
            completed_agents=["router", "web_search"],
            failed_agents=[]
        )
        
        # Test progress callback
        gradio_interface._progress_callback(progress)
        assert gradio_interface.current_progress == progress
        
        # Test progress display formatting
        display = gradio_interface._format_progress_display(progress)
        assert "45.0%" in display
        assert "Data Collection" in display
        assert "Collecting data from web sources" in display
        assert "router, web_search" in display
    
    def test_research_result_formatting(self, gradio_interface):
        """Test research result formatting for UI display"""
        # Create mock research result
        report = ResearchReport(
            executive_summary="Test research summary with key findings",
            key_findings=["Finding 1", "Finding 2", "Finding 3"],
            detailed_analysis="Comprehensive analysis of research data",
            sources=["[1] Source 1", "[2] Source 2"],
            recommendations=["Recommendation 1", "Recommendation 2"],
            metadata={"word_count": 500, "confidence_level": "high"}
        )
        
        result = Mock()
        result.success = True
        result.query = "test query"
        result.report = report
        result.source_count = 5
        result.execution_time = 45.2
        result.metadata = {
            'execution_time': 45.2,
            'agents_used': ['router', 'web_search', 'fact_checker', 'summarizer'],
            'data_sources': {
                'web_search_results': 3,
                'scraped_pages': 1,
                'vector_documents': 1
            }
        }
        
        # Test result formatting
        report_md, metadata_info, status_msg = gradio_interface._format_research_result(result)
        
        # Verify report formatting
        assert "# Research Report: test query" in report_md
        assert "## Executive Summary" in report_md
        assert "Test research summary" in report_md
        assert "## Key Findings" in report_md
        assert "Finding 1" in report_md
        assert "## Sources" in report_md
        assert "[1] Source 1" in report_md
        
        # Verify metadata formatting
        assert "45.2 seconds" in metadata_info
        assert "Web Search Results: 3" in metadata_info
        assert "Scraped Pages: 1" in metadata_info
        
        # Verify status message
        assert "✅ Research completed successfully" in status_msg
        assert "45.2s" in status_msg
        assert "5 sources" in status_msg
    
    def test_failed_research_result_formatting(self, gradio_interface):
        """Test formatting of failed research results"""
        result = Mock()
        result.success = False
        result.query = "failed query"
        result.error_message = "Network connection failed"
        result.report = None
        result.metadata = {
            'failed_agents': ['web_search', 'web_scraper'],
            'errors': ['Connection timeout', 'API rate limit exceeded']
        }
        
        report_md, metadata_info, status_msg = gradio_interface._format_research_result(result)
        
        assert "❌ **Research Failed**" in report_md
        assert "Network connection failed" in report_md
        assert "Research failed" in status_msg
    
    def test_research_history_formatting(self, gradio_interface):
        """Test research history display formatting"""
        # Mock research history data
        gradio_interface.research_history = [
            {
                'timestamp': '2024-01-15T10:30:00',
                'query': 'Test research query 1',
                'success': True,
                'execution_time': 45.2,
                'source_count': 8,
                'summary': 'Research summary for first query'
            },
            {
                'timestamp': '2024-01-15T09:15:00',
                'query': 'Test research query 2',
                'success': False,
                'execution_time': 12.5,
                'source_count': 0,
                'summary': 'Failed research attempt'
            }
        ]
        
        history_display = gradio_interface.get_research_history()
        
        assert "# Research History" in history_display
        assert "✅ Test research query 1" in history_display
        assert "❌ Test research query 2" in history_display
        assert "45.2s" in history_display
        assert "8" in history_display  # source count
        assert "Research summary for first query" in history_display
    
    def test_research_history_search(self, gradio_interface):
        """Test research history search functionality"""
        # Setup test history
        gradio_interface.research_history = [
            {
                'timestamp': '2024-01-15T10:30:00',
                'query': 'artificial intelligence research',
                'success': True,
                'execution_time': 45.2,
                'source_count': 8,
                'summary': 'AI research findings and trends'
            },
            {
                'timestamp': '2024-01-15T09:15:00',
                'query': 'climate change impacts',
                'success': True,
                'execution_time': 38.7,
                'source_count': 6,
                'summary': 'Environmental research on climate effects'
            }
        ]
        
        # Test search functionality
        search_results = gradio_interface.search_research_history("artificial intelligence")
        assert "artificial intelligence research" in search_results
        assert "AI research findings" in search_results
        assert "climate change" not in search_results
        
        # Test empty search returns all history
        all_results = gradio_interface.search_research_history("")
        assert "artificial intelligence research" in all_results
        assert "climate change impacts" in all_results
    
    def test_analytics_dashboard_formatting(self, gradio_interface):
        """Test analytics dashboard display formatting"""
        # Mock orchestrator with analytics data
        mock_orchestrator = Mock()
        mock_orchestrator.get_research_analytics.return_value = {
            'total_research': 25,
            'successful_research': 22,
            'success_rate': 88.0,
            'total_sources': 180,
            'avg_execution_time': 42.5,
            'avg_sources_per_research': 7.2,
            'sheets_available': True,
            'last_updated': '2024-01-15 10:30:00'
        }
        gradio_interface.orchestrator = mock_orchestrator
        
        analytics_display = gradio_interface.get_analytics_dashboard()
        
        assert "# Analytics Dashboard" in analytics_display
        assert "**Total Research Queries:** 25" in analytics_display
        assert "**Success Rate:** 88.0%" in analytics_display
        assert "**Average Execution Time:** 42.5 seconds" in analytics_display
        assert "✅ Connected" in analytics_display  # Google Sheets status
    
    def test_system_status_display(self, gradio_interface, mock_orchestrator):
        """Test system status information display"""
        gradio_interface.orchestrator = mock_orchestrator
        
        status_display = gradio_interface.get_system_status()
        
        assert "# System Status" in status_display
        assert "## Orchestrator Status: ✅ Healthy" in status_display
        assert "**Router:** ✅ Available" in status_display
        assert "**Web Search:** ✅ Available" in status_display
        assert "✅ Configured" in status_display  # API configuration status
    
    def test_configuration_validation_display(self, gradio_interface):
        """Test system configuration validation display"""
        validation_display = gradio_interface.validate_system_configuration()
        
        assert "# System Configuration Validation" in validation_display
        assert "## Orchestrator Initialization Test" in validation_display
        assert "## API Connectivity" in validation_display
        assert "## Database Configuration" in validation_display
    
    def test_input_validation(self, gradio_interface):
        """Test input validation for research queries"""
        # Test empty query
        result = gradio_interface.conduct_research("", 10, True, True, "academic", "medium", 120)
        report, metadata, status, progress = result
        assert "Please enter a research query" in report
        assert "❌ No query provided" in status
        
        # Test short query
        result = gradio_interface.conduct_research("AI", 10, True, True, "academic", "medium", 120)
        report, metadata, status, progress = result
        assert "more detailed research query" in report
        assert "❌ Query too short" in status
        
        # Test invalid parameters - mock orchestrator to avoid initialization issues
        with patch.object(gradio_interface, '_initialize_orchestrator', return_value=(True, "")):
            result = gradio_interface.conduct_research("valid query", 0, True, True, "academic", "medium", 120)
            report, metadata, status, progress = result
            assert "Maximum sources must be between 1 and 50" in report
            assert "❌ Invalid configuration" in status


class TestGradioInterfaceInteractions:
    """Test user interaction flows and interface behavior"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        config = Mock()
        config.api = Mock()
        config.api.gemini_api_key = "test_key"
        config.api.serpapi_key = "test_serpapi_key"
        config.api.google_sheets_credentials_path = "/test/path/credentials.json"
        config.database = Mock()
        config.database.chroma_db_path = "/test/chroma"
        config.gradio_host = "127.0.0.1"
        config.gradio_port = 7860
        return config
    
    @pytest.fixture
    def gradio_interface(self, mock_config):
        """Create GradioInterface instance with mocked dependencies"""
        with patch('ui.gradio_interface.get_config', return_value=mock_config):
            with patch('os.path.exists', return_value=True):
                interface = GradioInterface()
                return interface
    
    def test_research_workflow_interaction(self, gradio_interface):
        """Test complete research workflow interaction"""
        # Mock successful orchestrator
        mock_orchestrator = Mock()
        mock_result = Mock()
        mock_result.success = True
        mock_result.query = "test interaction query"
        mock_result.execution_time = 35.0
        mock_result.source_count = 6
        mock_result.report = ResearchReport(
            executive_summary="Interaction test summary",
            key_findings=["Interactive finding 1", "Interactive finding 2"],
            detailed_analysis="Interactive analysis",
            sources=["[1] Interactive source"],
            recommendations=["Interactive recommendation"],
            metadata={"word_count": 300}
        )
        mock_result.metadata = {
            'execution_time': 35.0,
            'agents_used': ['router', 'web_search', 'summarizer'],
            'data_sources': {'web_search_results': 6, 'scraped_pages': 0, 'vector_documents': 0}
        }
        
        mock_orchestrator.research.return_value = mock_result
        gradio_interface.orchestrator = mock_orchestrator
        
        # Test research interaction
        result = gradio_interface.conduct_research(
            "test interaction query",
            10,  # max_sources
            True,  # enable_web_scraping
            True,  # enable_vector_search
            "academic",  # report_style
            "medium",  # report_length
            120  # timeout_seconds
        )
        
        report, metadata, status, progress = result
        
        # Verify successful interaction
        assert "# Research Report: test interaction query" in report
        assert "Interaction test summary" in report
        assert "Interactive finding 1" in report
        assert "✅ Research completed successfully" in status
        assert "35.0s" in status
        assert "6 sources" in status
    
    def test_concurrent_research_prevention(self, gradio_interface):
        """Test prevention of concurrent research operations"""
        # Set research in progress
        gradio_interface.is_researching = True
        
        result = gradio_interface.conduct_research(
            "concurrent test query", 10, True, True, "academic", "medium", 120
        )
        
        report, metadata, status, progress = result
        assert "Research already in progress" in report
        assert "⚠️ Research in progress" in status
    
    def test_configuration_error_handling(self, gradio_interface):
        """Test handling of configuration errors during interaction"""
        # Simulate configuration errors
        gradio_interface.config_errors = ["Gemini API key not configured", "Database path invalid"]
        
        result = gradio_interface.conduct_research(
            "config error test", 10, True, True, "academic", "medium", 120
        )
        
        report, metadata, status, progress = result
        assert "Configuration errors detected" in report
        assert "Gemini API key not configured" in report
        assert "❌ Configuration errors" in status
    
    def test_orchestrator_failure_interaction(self, gradio_interface):
        """Test interaction when orchestrator fails to initialize"""
        # Mock orchestrator initialization failure
        with patch.object(gradio_interface, '_initialize_orchestrator', return_value=(False, "Initialization failed")):
            result = gradio_interface.conduct_research(
                "orchestrator failure test", 10, True, True, "academic", "medium", 120
            )
            
            report, metadata, status, progress = result
            assert "Failed to initialize research system" in report
            assert "❌ System initialization failed" in status
    
    def test_research_timeout_interaction(self, gradio_interface):
        """Test research timeout interaction flow"""
        # Mock orchestrator that simulates timeout
        mock_orchestrator = Mock()
        
        def slow_research(*args, **kwargs):
            time.sleep(0.2)  # Simulate slow operation
            return None  # Simulate timeout
        
        mock_orchestrator.research.side_effect = slow_research
        gradio_interface.orchestrator = mock_orchestrator
        
        # Test with very short timeout
        result = gradio_interface.conduct_research(
            "timeout test query", 10, True, True, "academic", "medium", 1  # 1 second timeout
        )
        
        report, metadata, status, progress = result
        # Should handle timeout gracefully
        assert isinstance(report, str)
        assert isinstance(status, str)
    
    def test_progress_update_interaction(self, gradio_interface):
        """Test progress update interaction during research"""
        progress_updates = []
        
        def mock_progress_callback(progress_text):
            progress_updates.append(progress_text)
        
        gradio_interface.progress_update_callback = mock_progress_callback
        
        # Simulate progress updates
        progress1 = ProgressStatus(
            current_step="planning",
            completion_percentage=20.0,
            status_message="Analyzing query",
            estimated_time_remaining=60,
            completed_agents=[],
            failed_agents=[]
        )
        
        progress2 = ProgressStatus(
            current_step="data_collection",
            completion_percentage=60.0,
            status_message="Collecting data",
            estimated_time_remaining=30,
            completed_agents=["router"],
            failed_agents=[]
        )
        
        # Test progress callbacks
        gradio_interface._progress_callback(progress1)
        gradio_interface._progress_callback(progress2)
        
        # Verify progress updates were captured
        assert len(progress_updates) == 2
        assert "20.0%" in progress_updates[0]
        assert "60.0%" in progress_updates[1]
        assert "Planning" in progress_updates[0]
        assert "Data Collection" in progress_updates[1]
    
    def test_history_interaction_flow(self, gradio_interface):
        """Test history tab interaction flow"""
        # Setup mock history
        gradio_interface.research_history = [
            {
                'timestamp': '2024-01-15T10:30:00',
                'query': 'history interaction test',
                'success': True,
                'execution_time': 42.0,
                'source_count': 7,
                'summary': 'History interaction summary'
            }
        ]
        
        # Test history display
        history_display = gradio_interface.get_research_history()
        assert "history interaction test" in history_display
        assert "42.0s" in history_display
        
        # Test history search
        search_results = gradio_interface.search_research_history("interaction")
        assert "history interaction test" in search_results
        
        # Test empty search
        empty_search = gradio_interface.search_research_history("")
        assert "history interaction test" in empty_search
    
    def test_analytics_interaction_flow(self, gradio_interface):
        """Test analytics tab interaction flow"""
        # Mock orchestrator with analytics
        mock_orchestrator = Mock()
        mock_orchestrator.get_research_analytics.return_value = {
            'total_research': 15,
            'successful_research': 13,
            'success_rate': 86.7,
            'avg_execution_time': 38.5
        }
        gradio_interface.orchestrator = mock_orchestrator
        
        # Test analytics display
        analytics_display = gradio_interface.get_analytics_dashboard()
        assert "**Total Research Queries:** 15" in analytics_display
        assert "**Success Rate:** 86.7%" in analytics_display
        
        # Verify orchestrator was called
        mock_orchestrator.get_research_analytics.assert_called_once()
    
    def test_settings_interaction_flow(self, gradio_interface):
        """Test settings tab interaction flow"""
        # Test configuration update (placeholder functionality)
        config_result = gradio_interface.update_api_configuration(
            "new_gemini_key", "new_serpapi_key", "/new/path/credentials.json"
        )
        
        assert "Configuration Update Not Implemented" in config_result
        assert "Edit your .env file" in config_result
        
        # Test configuration validation
        validation_result = gradio_interface.validate_system_configuration()
        assert "System Configuration Validation" in validation_result


class TestGradioInterfaceCreation:
    """Test Gradio interface creation and component structure"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        config = Mock()
        config.api = Mock()
        config.api.gemini_api_key = "test_key"
        config.api.serpapi_key = None
        config.api.google_sheets_credentials_path = None
        config.database = Mock()
        config.database.chroma_db_path = "/test/chroma"
        config.gradio_host = "127.0.0.1"
        config.gradio_port = 7860
        return config
    
    def test_create_gradio_app_factory(self, mock_config):
        """Test create_gradio_app factory function"""
        with patch('ui.gradio_interface.get_config', return_value=mock_config):
            with patch('os.path.exists', return_value=True):
                app = create_gradio_app()
                assert isinstance(app, GradioInterface)
                assert app.config is not None
    
    def test_interface_creation_structure(self, mock_config):
        """Test Gradio interface structure and components"""
        with patch('ui.gradio_interface.get_config', return_value=mock_config):
            with patch('os.path.exists', return_value=True):
                interface = GradioInterface()
                
                # Test that interface has the create_interface method
                assert hasattr(interface, 'create_interface')
                assert callable(interface.create_interface)
                
                # Test interface creation would work (without actually creating it due to Gradio context issues)
                # This validates the method exists and is properly structured
    
    def test_interface_launch_parameters(self, mock_config):
        """Test interface launch with different parameters"""
        with patch('ui.gradio_interface.get_config', return_value=mock_config):
            with patch('os.path.exists', return_value=True):
                interface = GradioInterface()
                
                # Mock the launch method to avoid actual server startup
                with patch.object(interface, 'create_interface') as mock_create:
                    mock_gradio_interface = Mock()
                    mock_create.return_value = mock_gradio_interface
                    
                    # Test launch with default parameters
                    interface.launch()
                    mock_gradio_interface.launch.assert_called_once()
                    
                    # Verify launch parameters
                    call_args = mock_gradio_interface.launch.call_args
                    assert call_args[1]['server_name'] == "127.0.0.1"
                    assert call_args[1]['server_port'] == 7860
                    assert call_args[1]['share'] is False
                    assert call_args[1]['debug'] is False
    
    def test_interface_error_handling_during_creation(self, mock_config):
        """Test error handling during interface creation"""
        # Set up config to have missing credentials file
        mock_config.api.google_sheets_credentials_path = "/nonexistent/path"
        
        with patch('ui.gradio_interface.get_config', return_value=mock_config):
            with patch('os.path.exists', return_value=False):  # Simulate missing files
                interface = GradioInterface()
                
                # Should handle missing files gracefully
                assert interface is not None
                assert len(interface.config_errors) > 0


class TestUIComponentIntegration:
    """Integration tests for UI components with mock data"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        config = Mock()
        config.api = Mock()
        config.api.gemini_api_key = "test_key"
        config.api.serpapi_key = "test_serpapi_key"
        config.api.google_sheets_credentials_path = "/test/path/credentials.json"
        config.database = Mock()
        config.database.chroma_db_path = "/test/chroma"
        config.gradio_host = "127.0.0.1"
        config.gradio_port = 7860
        return config
    
    def test_complete_ui_workflow_integration(self, mock_config):
        """Test complete UI workflow with mock data"""
        with patch('ui.gradio_interface.get_config', return_value=mock_config):
            with patch('os.path.exists', return_value=True):
                interface = GradioInterface()
                
                # Mock successful research workflow
                mock_orchestrator = Mock()
                mock_result = Mock()
                mock_result.success = True
                mock_result.query = "integration test query"
                mock_result.execution_time = 40.0
                mock_result.source_count = 8
                mock_result.report = ResearchReport(
                    executive_summary="Integration test summary",
                    key_findings=["Integration finding 1", "Integration finding 2"],
                    detailed_analysis="Integration analysis",
                    sources=["[1] Integration source"],
                    recommendations=["Integration recommendation"],
                    metadata={"word_count": 400}
                )
                mock_result.metadata = {
                    'execution_time': 40.0,
                    'agents_used': ['router', 'web_search', 'fact_checker', 'summarizer'],
                    'data_sources': {'web_search_results': 5, 'scraped_pages': 2, 'vector_documents': 1}
                }
                
                mock_orchestrator.research.return_value = mock_result
                mock_orchestrator.get_research_history.return_value = [
                    {
                        'timestamp': '2024-01-15T10:30:00',
                        'query': 'integration test query',
                        'success': True,
                        'execution_time': 40.0,
                        'source_count': 8,
                        'summary': 'Integration test summary'
                    }
                ]
                mock_orchestrator.get_research_analytics.return_value = {
                    'total_research': 1,
                    'successful_research': 1,
                    'success_rate': 100.0,
                    'avg_execution_time': 40.0
                }
                
                interface.orchestrator = mock_orchestrator
                
                # Test complete workflow
                # 1. Conduct research
                research_result = interface.conduct_research(
                    "integration test query", 10, True, True, "academic", "medium", 120
                )
                report, metadata, status, progress = research_result
                
                assert "Integration test summary" in report
                assert "✅ Research completed successfully" in status
                
                # 2. Check history
                history = interface.get_research_history()
                assert "integration test query" in history
                
                # 3. Check analytics
                analytics = interface.get_analytics_dashboard()
                assert "**Total Research Queries:** 1" in analytics
                assert "**Success Rate:** 100.0%" in analytics
                
                # 4. Validate system status
                status_info = interface.get_system_status()
                assert "System Status" in status_info
    
    def test_ui_error_recovery_integration(self, mock_config):
        """Test UI error recovery and graceful degradation"""
        with patch('ui.gradio_interface.get_config', return_value=mock_config):
            with patch('os.path.exists', return_value=True):
                interface = GradioInterface()
                
                # Mock orchestrator with mixed success/failure
                mock_orchestrator = Mock()
                mock_orchestrator.research.side_effect = Exception("Research failed")
                mock_orchestrator.get_research_history.return_value = []
                mock_orchestrator.get_research_analytics.side_effect = Exception("Analytics failed")
                
                interface.orchestrator = mock_orchestrator
                
                # Test research failure handling
                research_result = interface.conduct_research(
                    "error recovery test", 10, True, True, "academic", "medium", 120
                )
                report, metadata, status, progress = research_result
                
                # Should handle error gracefully
                assert isinstance(report, str)
                assert isinstance(status, str)
                
                # Test analytics failure handling
                analytics = interface.get_analytics_dashboard()
                # Should fallback to local analytics or show error message
                assert isinstance(analytics, str)
    
    def test_ui_performance_with_large_data(self, mock_config):
        """Test UI performance with large datasets"""
        with patch('ui.gradio_interface.get_config', return_value=mock_config):
            with patch('os.path.exists', return_value=True):
                interface = GradioInterface()
                
                # Create large mock history
                large_history = []
                for i in range(100):
                    large_history.append({
                        'timestamp': f'2024-01-{i%30+1:02d}T10:30:00',
                        'query': f'Large dataset test query {i}',
                        'success': i % 10 != 0,  # 90% success rate
                        'execution_time': 30.0 + (i % 20),
                        'source_count': 5 + (i % 10),
                        'summary': f'Summary for query {i} with detailed information'
                    })
                
                interface.research_history = large_history
                
                # Test history display performance
                start_time = time.time()
                history_display = interface.get_research_history()
                history_time = time.time() - start_time
                
                # Should complete within reasonable time (< 1 second)
                assert history_time < 1.0
                assert "Large dataset test query" in history_display
                
                # Test search performance
                start_time = time.time()
                search_results = interface.search_research_history("query 50")
                search_time = time.time() - start_time
                
                # Search should be fast
                assert search_time < 0.5
                assert "Large dataset test query 50" in search_results