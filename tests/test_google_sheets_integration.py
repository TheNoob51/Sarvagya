"""
Google Sheets Integration Tests for the Intelligent Research Assistant.
Tests sheet operations with mock Google Sheets API and validates data formatting and error handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json
import os

from utils.google_sheets_handler import GoogleSheetsHandler
from agents.data_models import ResearchResult, ResearchReport, ReportStyle, ReportLength
from gspread.exceptions import APIError, SpreadsheetNotFound, WorksheetNotFound


class TestGoogleSheetsHandler:
    """Test suite for GoogleSheetsHandler class"""
    
    @pytest.fixture
    def mock_credentials_path(self, tmp_path):
        """Create a temporary credentials file for testing"""
        creds_file = tmp_path / "test_credentials.json"
        creds_data = {
            "type": "service_account",
            "project_id": "test-project",
            "private_key_id": "test-key-id",
            "private_key": "-----BEGIN PRIVATE KEY-----\ntest-key\n-----END PRIVATE KEY-----\n",
            "client_email": "test@test-project.iam.gserviceaccount.com",
            "client_id": "123456789",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token"
        }
        creds_file.write_text(json.dumps(creds_data))
        return str(creds_file)
    
    @pytest.fixture
    def sample_research_result(self):
        """Create a sample ResearchResult for testing"""
        report = ResearchReport(
            executive_summary="This is a test executive summary.",
            key_findings=["Finding 1", "Finding 2", "Finding 3"],
            detailed_analysis="This is detailed analysis of the research topic.",
            sources=["Source 1", "Source 2"],
            recommendations=["Recommendation 1", "Recommendation 2"],
            metadata={"quality_score": 8.5}
        )
        
        return ResearchResult(
            query="Test research query",
            report=report,
            metadata={
                "report_style": "academic",
                "report_length": "medium",
                "processing_agents": ["router", "web_search", "summarizer"]
            },
            execution_time=45.2,
            source_count=5,
            success=True,
            error_message=None
        )
    
    def test_initialization_without_credentials(self):
        """Test handler initialization without credentials"""
        handler = GoogleSheetsHandler()
        
        assert handler.credentials_path is None
        assert handler.client is None
        assert handler.spreadsheet is None
        assert handler.worksheet is None
        assert not handler.is_available()
    
    def test_initialization_with_invalid_credentials(self):
        """Test handler initialization with invalid credentials path"""
        handler = GoogleSheetsHandler(credentials_path="/invalid/path.json")
        
        assert handler.credentials_path == "/invalid/path.json"
        assert handler.client is None
        assert not handler.is_available()
    
    @patch('utils.google_sheets_handler.gspread.authorize')
    @patch('utils.google_sheets_handler.Credentials.from_service_account_file')
    def test_successful_authentication(self, mock_credentials, mock_authorize, mock_credentials_path):
        """Test successful authentication with Google Sheets API"""
        # Mock the credentials and client
        mock_creds = Mock()
        mock_credentials.return_value = mock_creds
        mock_client = Mock()
        mock_authorize.return_value = mock_client
        
        handler = GoogleSheetsHandler(credentials_path=mock_credentials_path)
        
        # Verify authentication was called correctly
        mock_credentials.assert_called_once()
        mock_authorize.assert_called_once_with(mock_creds)
        assert handler.client == mock_client
    
    @patch('utils.google_sheets_handler.gspread.authorize')
    @patch('utils.google_sheets_handler.Credentials.from_service_account_file')
    def test_authentication_failure(self, mock_credentials, mock_authorize, mock_credentials_path):
        """Test authentication failure handling"""
        # Mock authentication failure
        mock_credentials.side_effect = Exception("Authentication failed")
        
        handler = GoogleSheetsHandler(credentials_path=mock_credentials_path)
        
        assert handler.client is None
        assert not handler.is_available()
    
    @patch('utils.google_sheets_handler.gspread.authorize')
    @patch('utils.google_sheets_handler.Credentials.from_service_account_file')
    def test_initialize_with_existing_spreadsheet(self, mock_credentials, mock_authorize, mock_credentials_path):
        """Test initialization with existing spreadsheet"""
        # Setup mocks
        mock_client = Mock()
        mock_authorize.return_value = mock_client
        mock_spreadsheet = Mock()
        mock_worksheet = Mock()
        
        mock_client.open.return_value = mock_spreadsheet
        mock_spreadsheet.worksheet.return_value = mock_worksheet
        
        handler = GoogleSheetsHandler(credentials_path=mock_credentials_path)
        result = handler.initialize()
        
        assert result is True
        assert handler.spreadsheet == mock_spreadsheet
        assert handler.worksheet == mock_worksheet
        mock_client.open.assert_called_once_with("Research Assistant Data")
    
    @patch('utils.google_sheets_handler.gspread.authorize')
    @patch('utils.google_sheets_handler.Credentials.from_service_account_file')
    def test_initialize_creates_new_spreadsheet(self, mock_credentials, mock_authorize, mock_credentials_path):
        """Test initialization creates new spreadsheet when it doesn't exist"""
        # Setup mocks
        mock_client = Mock()
        mock_authorize.return_value = mock_client
        mock_spreadsheet = Mock()
        mock_worksheet = Mock()
        
        # Mock spreadsheet not found, then creation
        mock_client.open.side_effect = SpreadsheetNotFound("Spreadsheet not found")
        mock_client.create.return_value = mock_spreadsheet
        mock_spreadsheet.worksheet.side_effect = WorksheetNotFound("Worksheet not found")
        mock_spreadsheet.add_worksheet.return_value = mock_worksheet
        
        handler = GoogleSheetsHandler(credentials_path=mock_credentials_path)
        result = handler.initialize()
        
        assert result is True
        mock_client.create.assert_called_once_with("Research Assistant Data")
        mock_spreadsheet.add_worksheet.assert_called_once()
        mock_worksheet.append_row.assert_called_once_with(handler.COLUMNS)
    
    @patch('utils.google_sheets_handler.gspread.authorize')
    @patch('utils.google_sheets_handler.Credentials.from_service_account_file')
    def test_save_research_success(self, mock_credentials, mock_authorize, mock_credentials_path, sample_research_result):
        """Test successful research data saving"""
        # Setup mocks
        mock_client = Mock()
        mock_authorize.return_value = mock_client
        mock_worksheet = Mock()
        
        handler = GoogleSheetsHandler(credentials_path=mock_credentials_path)
        handler.worksheet = mock_worksheet
        
        result = handler.save_research(sample_research_result)
        
        assert result is True
        mock_worksheet.append_row.assert_called_once()
        
        # Verify the row data format
        call_args = mock_worksheet.append_row.call_args[0][0]
        assert len(call_args) == len(handler.COLUMNS)
        assert call_args[1] == sample_research_result.query  # Query column
        assert call_args[2] == sample_research_result.report.executive_summary  # Summary column
        assert call_args[4] == str(sample_research_result.source_count)  # Source count
        assert call_args[6] == "Yes"  # Success column
    
    def test_save_research_without_initialization(self, sample_research_result):
        """Test saving research without proper initialization"""
        handler = GoogleSheetsHandler()
        
        result = handler.save_research(sample_research_result)
        
        assert result is False
    
    @patch('utils.google_sheets_handler.gspread.authorize')
    @patch('utils.google_sheets_handler.Credentials.from_service_account_file')
    def test_save_research_api_error(self, mock_credentials, mock_authorize, mock_credentials_path, sample_research_result):
        """Test handling API errors during research saving"""
        # Setup mocks
        mock_client = Mock()
        mock_authorize.return_value = mock_client
        mock_worksheet = Mock()
        
        # Create a proper mock response for APIError
        mock_response = Mock()
        mock_response.text = "API quota exceeded"
        mock_response.json.return_value = {"error": "API quota exceeded"}
        mock_worksheet.append_row.side_effect = APIError(mock_response)
        
        handler = GoogleSheetsHandler(credentials_path=mock_credentials_path)
        handler.worksheet = mock_worksheet
        
        result = handler.save_research(sample_research_result)
        
        assert result is False
    
    @patch('utils.google_sheets_handler.gspread.authorize')
    @patch('utils.google_sheets_handler.Credentials.from_service_account_file')
    def test_get_recent_research(self, mock_credentials, mock_authorize, mock_credentials_path):
        """Test retrieving recent research entries"""
        # Setup mocks
        mock_client = Mock()
        mock_authorize.return_value = mock_client
        mock_worksheet = Mock()
        
        # Mock research data
        mock_records = [
            {
                'Timestamp': '2024-01-02 10:00:00',
                'Query': 'Recent query',
                'Summary': 'Recent summary',
                'Success': 'Yes'
            },
            {
                'Timestamp': '2024-01-01 10:00:00',
                'Query': 'Older query',
                'Summary': 'Older summary',
                'Success': 'Yes'
            }
        ]
        mock_worksheet.get_all_records.return_value = mock_records
        
        handler = GoogleSheetsHandler(credentials_path=mock_credentials_path)
        handler.worksheet = mock_worksheet
        
        result = handler.get_recent_research(limit=10)
        
        assert len(result) == 2
        # Should be sorted by timestamp (most recent first)
        assert result[0]['Query'] == 'Recent query'
        assert result[1]['Query'] == 'Older query'
    
    @patch('utils.google_sheets_handler.gspread.authorize')
    @patch('utils.google_sheets_handler.Credentials.from_service_account_file')
    def test_search_research(self, mock_credentials, mock_authorize, mock_credentials_path):
        """Test searching research entries by query"""
        # Setup mocks
        mock_client = Mock()
        mock_authorize.return_value = mock_client
        mock_worksheet = Mock()
        
        # Mock research data
        mock_records = [
            {
                'Timestamp': '2024-01-02 10:00:00',
                'Query': 'Python programming tutorial',
                'Summary': 'Learn Python basics',
                'Success': 'Yes'
            },
            {
                'Timestamp': '2024-01-01 10:00:00',
                'Query': 'JavaScript frameworks',
                'Summary': 'Comparison of JS frameworks',
                'Success': 'Yes'
            },
            {
                'Timestamp': '2024-01-03 10:00:00',
                'Query': 'Machine learning with Python',
                'Summary': 'ML algorithms in Python',
                'Success': 'Yes'
            }
        ]
        mock_worksheet.get_all_records.return_value = mock_records
        
        handler = GoogleSheetsHandler(credentials_path=mock_credentials_path)
        handler.worksheet = mock_worksheet
        
        # Search for "python"
        result = handler.search_research("python", limit=10)
        
        assert len(result) == 2
        # Should find both Python-related entries
        queries = [r['Query'] for r in result]
        assert 'Python programming tutorial' in queries
        assert 'Machine learning with Python' in queries
        assert 'JavaScript frameworks' not in queries
    
    @patch('utils.google_sheets_handler.gspread.authorize')
    @patch('utils.google_sheets_handler.Credentials.from_service_account_file')
    def test_get_analytics(self, mock_credentials, mock_authorize, mock_credentials_path):
        """Test analytics calculation from research data"""
        # Setup mocks
        mock_client = Mock()
        mock_authorize.return_value = mock_client
        mock_worksheet = Mock()
        
        # Mock research data with various scenarios
        mock_records = [
            {
                'Timestamp': '2024-01-01 10:00:00',
                'Query': 'Query 1',
                'Success': 'Yes',
                'Source Count': '5',
                'Processing Time (seconds)': '30.5',
                'Report Style': 'academic',
                'Report Length': 'medium'
            },
            {
                'Timestamp': '2024-01-02 10:00:00',
                'Query': 'Query 2',
                'Success': 'Yes',
                'Source Count': '3',
                'Processing Time (seconds)': '25.0',
                'Report Style': 'casual',
                'Report Length': 'short'
            },
            {
                'Timestamp': '2024-01-03 10:00:00',
                'Query': 'Query 3',
                'Success': 'No',
                'Source Count': '0',
                'Processing Time (seconds)': '0',
                'Report Style': 'academic',
                'Report Length': 'medium'
            }
        ]
        mock_worksheet.get_all_records.return_value = mock_records
        
        handler = GoogleSheetsHandler(credentials_path=mock_credentials_path)
        handler.worksheet = mock_worksheet
        
        result = handler.get_analytics()
        
        assert result['total_researches'] == 3
        assert result['successful_researches'] == 2
        assert result['failed_researches'] == 1
        assert result['success_rate'] == 66.67
        assert result['average_sources'] == 4.0  # (5+3)/2
        assert result['average_processing_time'] == 27.75  # (30.5+25.0)/2
        assert result['research_by_style']['academic'] == 2
        assert result['research_by_style']['casual'] == 1
    
    def test_format_full_report(self, mock_credentials_path):
        """Test research report formatting for Google Sheets storage"""
        handler = GoogleSheetsHandler(credentials_path=mock_credentials_path)
        
        report = ResearchReport(
            executive_summary="Test summary",
            key_findings=["Finding 1", "Finding 2"],
            detailed_analysis="Test analysis",
            sources=["Source 1", "Source 2"],
            recommendations=["Rec 1", "Rec 2"],
            metadata={}
        )
        
        formatted = handler._format_full_report(report)
        
        assert "EXECUTIVE SUMMARY:" in formatted
        assert "Test summary" in formatted
        assert "KEY FINDINGS:" in formatted
        assert "• Finding 1" in formatted
        assert "DETAILED ANALYSIS:" in formatted
        assert "Test analysis" in formatted
        assert "SOURCES & CITATIONS:" in formatted
        assert "[1] Source 1" in formatted
        assert "RECOMMENDATIONS:" in formatted
        assert "• Rec 1" in formatted
    
    def test_format_full_report_empty(self, mock_credentials_path):
        """Test formatting empty or None report"""
        handler = GoogleSheetsHandler(credentials_path=mock_credentials_path)
        
        assert handler._format_full_report(None) == ""
    
    @patch('utils.google_sheets_handler.gspread.authorize')
    @patch('utils.google_sheets_handler.Credentials.from_service_account_file')
    def test_rate_limiting(self, mock_credentials, mock_authorize, mock_credentials_path):
        """Test rate limiting functionality"""
        # Setup mocks
        mock_client = Mock()
        mock_authorize.return_value = mock_client
        
        handler = GoogleSheetsHandler(credentials_path=mock_credentials_path)
        
        # Test that rate limiting doesn't break functionality
        start_time = handler._last_request_time
        handler._rate_limit()
        
        # Request count should increment
        assert handler._request_count > 0
        # Last request time should be updated
        assert handler._last_request_time >= start_time
    
    @patch('utils.google_sheets_handler.gspread.authorize')
    @patch('utils.google_sheets_handler.Credentials.from_service_account_file')
    def test_api_quota_error_handling(self, mock_credentials, mock_authorize, mock_credentials_path):
        """Test handling of API quota exceeded errors"""
        # Setup mocks
        mock_client = Mock()
        mock_authorize.return_value = mock_client
        mock_worksheet = Mock()
        
        # Create a proper mock response for APIError
        mock_response = Mock()
        mock_response.text = "Quota exceeded for quota metric 'Read requests' and limit 'Read requests per minute per user'"
        mock_response.json.return_value = {"error": "Quota exceeded"}
        quota_error = APIError(mock_response)
        mock_worksheet.get_all_records.side_effect = quota_error
        
        handler = GoogleSheetsHandler(credentials_path=mock_credentials_path)
        handler.worksheet = mock_worksheet
        
        # Test that quota errors are handled gracefully
        result = handler.get_recent_research()
        assert result == []
        
        analytics = handler.get_analytics()
        assert analytics == {}
    
    def test_get_status(self, mock_credentials_path):
        """Test status reporting functionality"""
        handler = GoogleSheetsHandler(credentials_path=mock_credentials_path)
        
        status = handler.get_status()
        
        assert 'available' in status
        assert 'authenticated' in status
        assert 'spreadsheet_connected' in status
        assert 'worksheet_ready' in status
        assert 'credentials_path' in status
        assert 'spreadsheet_name' in status
        assert 'request_count' in status
        
        assert status['credentials_path'] == mock_credentials_path
        assert status['spreadsheet_name'] == "Research Assistant Data"
        assert status['available'] is False  # Not fully initialized
    
    @patch('utils.google_sheets_handler.gspread.authorize')
    @patch('utils.google_sheets_handler.Credentials.from_service_account_file')
    def test_error_handling_during_initialization(self, mock_credentials, mock_authorize, mock_credentials_path):
        """Test error handling during various initialization steps"""
        # Test spreadsheet creation failure
        mock_client = Mock()
        mock_authorize.return_value = mock_client
        mock_client.open.side_effect = SpreadsheetNotFound("Not found")
        
        # Create a proper mock response for APIError
        mock_response = Mock()
        mock_response.text = "Creation failed"
        mock_response.json.return_value = {"error": "Creation failed"}
        mock_client.create.side_effect = APIError(mock_response)
        
        handler = GoogleSheetsHandler(credentials_path=mock_credentials_path)
        result = handler.initialize()
        
        assert result is False
        assert handler.spreadsheet is None
    
    def test_save_research_with_failed_result(self, mock_credentials_path):
        """Test saving a failed research result"""
        handler = GoogleSheetsHandler(credentials_path=mock_credentials_path)
        handler.worksheet = Mock()
        
        # Create a failed research result
        failed_result = ResearchResult(
            query="Failed query",
            report=None,
            metadata={},
            execution_time=10.0,
            source_count=0,
            success=False,
            error_message="API timeout error"
        )
        
        result = handler.save_research(failed_result)
        
        # Should still attempt to save even failed results
        handler.worksheet.append_row.assert_called_once()
        
        # Check the saved data format
        call_args = handler.worksheet.append_row.call_args[0][0]
        assert call_args[6] == "No"  # Success column should be "No"
        assert call_args[9] == "API timeout error"  # Error message column