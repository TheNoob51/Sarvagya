"""
Google Sheets Handler for the Intelligent Research Assistant.
Manages research data storage, retrieval, and analytics using Google Sheets API.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import asdict

import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import APIError, SpreadsheetNotFound, WorksheetNotFound

from agents.data_models import ResearchResult, ResearchReport
from utils.logging_config import get_logger

logger = get_logger(__name__)

class GoogleSheetsHandler:
    """
    Handles Google Sheets operations for research data storage and analytics.
    
    Features:
    - Service account authentication
    - Research data saving with structured columns
    - Research history retrieval and search
    - Analytics calculations
    - Auto-creation of sheets and worksheets
    - Error handling for quota limits and connection issues
    """
    
    # Column structure for research data
    COLUMNS = [
        'Timestamp',
        'Query', 
        'Summary',
        'Full Report',
        'Source Count',
        'Processing Time (seconds)',
        'Success',
        'Report Style',
        'Report Length',
        'Error Message'
    ]
    
    def __init__(self, credentials_path: Optional[str] = None, spreadsheet_name: str = "Research Assistant Data"):
        """
        Initialize Google Sheets handler.
        
        Args:
            credentials_path: Path to Google service account JSON file
            spreadsheet_name: Name of the spreadsheet to use
        """
        self.credentials_path = credentials_path
        self.spreadsheet_name = spreadsheet_name
        self.client = None
        self.spreadsheet = None
        self.worksheet = None
        self._last_request_time = 0
        self._request_count = 0
        
        # Rate limiting settings (Google Sheets API limits)
        self.max_requests_per_minute = 100
        self.min_request_interval = 0.6  # seconds between requests
        
        if credentials_path:
            self._authenticate()
    
    def _authenticate(self) -> bool:
        """
        Authenticate with Google Sheets API using service account.
        
        Returns:
            bool: True if authentication successful, False otherwise
        """
        try:
            if not self.credentials_path or not os.path.exists(self.credentials_path):
                logger.warning(f"Google Sheets credentials not found at: {self.credentials_path}")
                return False
            
            # Define the scope for Google Sheets API
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            
            # Load credentials from service account file
            credentials = Credentials.from_service_account_file(
                self.credentials_path, 
                scopes=scope
            )
            
            # Create gspread client
            self.client = gspread.authorize(credentials)
            logger.info("Successfully authenticated with Google Sheets API")
            return True
            
        except Exception as e:
            logger.error(f"Failed to authenticate with Google Sheets: {e}")
            return False
    
    def _rate_limit(self):
        """Apply rate limiting to respect Google Sheets API limits."""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
        self._request_count += 1
    
    def _get_or_create_spreadsheet(self) -> bool:
        """
        Get existing spreadsheet or create new one if it doesn't exist.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            logger.error("Not authenticated with Google Sheets")
            return False
        
        try:
            self._rate_limit()
            
            # Try to open existing spreadsheet
            try:
                self.spreadsheet = self.client.open(self.spreadsheet_name)
                logger.info(f"Opened existing spreadsheet: {self.spreadsheet_name}")
            except SpreadsheetNotFound:
                # Create new spreadsheet
                logger.info(f"Creating new spreadsheet: {self.spreadsheet_name}")
                self.spreadsheet = self.client.create(self.spreadsheet_name)
                
                # Share with service account email (optional, for visibility)
                try:
                    with open(self.credentials_path, 'r') as f:
                        creds_data = json.load(f)
                        service_email = creds_data.get('client_email')
                        if service_email:
                            self.spreadsheet.share(service_email, perm_type='user', role='writer')
                except Exception as e:
                    logger.warning(f"Could not share spreadsheet with service account: {e}")
            
            return True
            
        except APIError as e:
            if 'quota' in str(e).lower():
                logger.error("Google Sheets API quota exceeded")
            else:
                logger.error(f"Google Sheets API error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error accessing spreadsheet: {e}")
            return False
    
    def _get_or_create_worksheet(self, worksheet_name: str = "Research Data") -> bool:
        """
        Get existing worksheet or create new one with proper headers.
        
        Args:
            worksheet_name: Name of the worksheet
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.spreadsheet:
            return False
        
        try:
            self._rate_limit()
            
            # Try to get existing worksheet
            try:
                self.worksheet = self.spreadsheet.worksheet(worksheet_name)
                logger.info(f"Using existing worksheet: {worksheet_name}")
            except WorksheetNotFound:
                # Create new worksheet
                logger.info(f"Creating new worksheet: {worksheet_name}")
                self.worksheet = self.spreadsheet.add_worksheet(
                    title=worksheet_name, 
                    rows=1000, 
                    cols=len(self.COLUMNS)
                )
                
                # Add headers
                self._rate_limit()
                self.worksheet.append_row(self.COLUMNS)
                logger.info("Added column headers to new worksheet")
            
            return True
            
        except APIError as e:
            logger.error(f"Google Sheets API error creating worksheet: {e}")
            return False
        except Exception as e:
            logger.error(f"Error creating worksheet: {e}")
            return False
    
    def initialize(self) -> bool:
        """
        Initialize Google Sheets connection and ensure spreadsheet/worksheet exist.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not self.credentials_path:
            logger.info("Google Sheets credentials not configured, skipping initialization")
            return False
        
        if not self._authenticate():
            return False
        
        if not self._get_or_create_spreadsheet():
            return False
        
        if not self._get_or_create_worksheet():
            return False
        
        logger.info("Google Sheets handler initialized successfully")
        return True
    
    def save_research(self, result: ResearchResult) -> bool:
        """
        Save research result to Google Sheets.
        
        Args:
            result: ResearchResult object containing research data
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        if not self.worksheet:
            logger.warning("Google Sheets not initialized, cannot save research")
            return False
        
        try:
            # Prepare row data according to column structure
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            row_data = [
                timestamp,
                result.query,
                result.report.executive_summary if result.report else "",
                self._format_full_report(result.report) if result.report else "",
                str(result.source_count),
                f"{result.execution_time:.2f}",
                "Yes" if result.success else "No",
                result.metadata.get('report_style', 'academic'),
                result.metadata.get('report_length', 'medium'),
                result.error_message or ""
            ]
            
            self._rate_limit()
            self.worksheet.append_row(row_data)
            
            logger.info(f"Successfully saved research to Google Sheets: {result.query[:50]}...")
            return True
            
        except APIError as e:
            if 'quota' in str(e).lower():
                logger.error("Google Sheets API quota exceeded while saving research")
            else:
                logger.error(f"Google Sheets API error while saving: {e}")
            return False
        except Exception as e:
            logger.error(f"Error saving research to Google Sheets: {e}")
            return False
    
    def _format_full_report(self, report: ResearchReport) -> str:
        """
        Format research report for storage in Google Sheets.
        
        Args:
            report: ResearchReport object
            
        Returns:
            str: Formatted report text
        """
        if not report:
            return ""
        
        formatted_report = f"""EXECUTIVE SUMMARY:
{report.executive_summary}

KEY FINDINGS:
{chr(10).join(f"• {finding}" for finding in report.key_findings)}

DETAILED ANALYSIS:
{report.detailed_analysis}

SOURCES & CITATIONS:
{chr(10).join(f"[{i+1}] {source}" for i, source in enumerate(report.sources))}

RECOMMENDATIONS:
{chr(10).join(f"• {rec}" for rec in report.recommendations)}"""
        
        return formatted_report
    
    def get_recent_research(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieve recent research entries from Google Sheets.
        
        Args:
            limit: Maximum number of entries to retrieve
            
        Returns:
            List of research entries as dictionaries
        """
        if not self.worksheet:
            logger.warning("Google Sheets not initialized, cannot retrieve research")
            return []
        
        try:
            self._rate_limit()
            
            # Get all records
            records = self.worksheet.get_all_records()
            
            # Sort by timestamp (most recent first) and limit
            sorted_records = sorted(
                records, 
                key=lambda x: x.get('Timestamp', ''), 
                reverse=True
            )[:limit]
            
            logger.info(f"Retrieved {len(sorted_records)} recent research entries")
            return sorted_records
            
        except APIError as e:
            logger.error(f"Google Sheets API error retrieving research: {e}")
            return []
        except Exception as e:
            logger.error(f"Error retrieving recent research: {e}")
            return []
    
    def search_research(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search research entries by query text.
        
        Args:
            query: Search query to match against research queries and summaries
            limit: Maximum number of results to return
            
        Returns:
            List of matching research entries
        """
        if not self.worksheet:
            logger.warning("Google Sheets not initialized, cannot search research")
            return []
        
        try:
            self._rate_limit()
            
            # Get all records
            records = self.worksheet.get_all_records()
            
            # Filter records that match the query
            query_lower = query.lower()
            matching_records = []
            
            for record in records:
                # Search in query and summary fields
                query_text = record.get('Query', '').lower()
                summary_text = record.get('Summary', '').lower()
                
                if query_lower in query_text or query_lower in summary_text:
                    matching_records.append(record)
            
            # Sort by timestamp (most recent first) and limit
            sorted_matches = sorted(
                matching_records,
                key=lambda x: x.get('Timestamp', ''),
                reverse=True
            )[:limit]
            
            logger.info(f"Found {len(sorted_matches)} research entries matching query: {query}")
            return sorted_matches
            
        except APIError as e:
            logger.error(f"Google Sheets API error searching research: {e}")
            return []
        except Exception as e:
            logger.error(f"Error searching research: {e}")
            return []
    
    def get_analytics(self) -> Dict[str, Any]:
        """
        Calculate analytics from research data.
        
        Returns:
            Dictionary containing analytics metrics
        """
        if not self.worksheet:
            logger.warning("Google Sheets not initialized, cannot calculate analytics")
            return {}
        
        try:
            self._rate_limit()
            
            # Get all records
            records = self.worksheet.get_all_records()
            
            if not records:
                return {
                    'total_researches': 0,
                    'successful_researches': 0,
                    'failed_researches': 0,
                    'success_rate': 0.0,
                    'average_sources': 0.0,
                    'average_processing_time': 0.0,
                    'total_processing_time': 0.0,
                    'most_recent_research': None,
                    'research_by_style': {},
                    'research_by_length': {}
                }
            
            # Calculate metrics
            total_researches = len(records)
            successful_researches = sum(1 for r in records if r.get('Success') == 'Yes')
            failed_researches = total_researches - successful_researches
            success_rate = (successful_researches / total_researches) * 100 if total_researches > 0 else 0
            
            # Calculate averages for successful researches only
            successful_records = [r for r in records if r.get('Success') == 'Yes']
            
            if successful_records:
                # Average source count
                source_counts = []
                for record in successful_records:
                    try:
                        source_count = int(record.get('Source Count', 0))
                        source_counts.append(source_count)
                    except (ValueError, TypeError):
                        pass
                
                average_sources = sum(source_counts) / len(source_counts) if source_counts else 0
                
                # Average processing time
                processing_times = []
                for record in successful_records:
                    try:
                        proc_time = float(record.get('Processing Time (seconds)', 0))
                        processing_times.append(proc_time)
                    except (ValueError, TypeError):
                        pass
                
                average_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
                total_processing_time = sum(processing_times)
            else:
                average_sources = 0
                average_processing_time = 0
                total_processing_time = 0
            
            # Most recent research
            most_recent = None
            if records:
                sorted_records = sorted(records, key=lambda x: x.get('Timestamp', ''), reverse=True)
                most_recent = sorted_records[0].get('Timestamp')
            
            # Research by style and length
            research_by_style = {}
            research_by_length = {}
            
            for record in records:
                style = record.get('Report Style', 'unknown')
                length = record.get('Report Length', 'unknown')
                
                research_by_style[style] = research_by_style.get(style, 0) + 1
                research_by_length[length] = research_by_length.get(length, 0) + 1
            
            analytics = {
                'total_researches': total_researches,
                'successful_researches': successful_researches,
                'failed_researches': failed_researches,
                'success_rate': round(success_rate, 2),
                'average_sources': round(average_sources, 2),
                'average_processing_time': round(average_processing_time, 2),
                'total_processing_time': round(total_processing_time, 2),
                'most_recent_research': most_recent,
                'research_by_style': research_by_style,
                'research_by_length': research_by_length
            }
            
            logger.info("Successfully calculated research analytics")
            return analytics
            
        except APIError as e:
            logger.error(f"Google Sheets API error calculating analytics: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error calculating analytics: {e}")
            return {}
    
    def delete_old_research(self, days_old: int = 90) -> int:
        """
        Delete research entries older than specified days.
        
        Args:
            days_old: Number of days after which to delete entries
            
        Returns:
            Number of entries deleted
        """
        if not self.worksheet:
            logger.warning("Google Sheets not initialized, cannot delete old research")
            return 0
        
        try:
            self._rate_limit()
            
            # Get all records with row numbers
            all_values = self.worksheet.get_all_values()
            
            if len(all_values) <= 1:  # Only headers or empty
                return 0
            
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            # Find rows to delete (in reverse order to maintain indices)
            rows_to_delete = []
            
            for i, row in enumerate(all_values[1:], start=2):  # Skip header row
                if len(row) > 0:  # Check if row has data
                    try:
                        timestamp_str = row[0]  # First column is timestamp
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                        
                        if timestamp < cutoff_date:
                            rows_to_delete.append(i)
                    except (ValueError, IndexError):
                        # Skip rows with invalid timestamps
                        continue
            
            # Delete rows in reverse order to maintain indices
            deleted_count = 0
            for row_num in reversed(rows_to_delete):
                try:
                    self._rate_limit()
                    self.worksheet.delete_rows(row_num)
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Error deleting row {row_num}: {e}")
            
            logger.info(f"Deleted {deleted_count} old research entries (older than {days_old} days)")
            return deleted_count
            
        except APIError as e:
            logger.error(f"Google Sheets API error deleting old research: {e}")
            return 0
        except Exception as e:
            logger.error(f"Error deleting old research: {e}")
            return 0
    
    def export_research_data(self, format: str = 'csv') -> Optional[str]:
        """
        Export all research data to specified format.
        
        Args:
            format: Export format ('csv' or 'json')
            
        Returns:
            Exported data as string, or None if error
        """
        if not self.worksheet:
            logger.warning("Google Sheets not initialized, cannot export data")
            return None
        
        try:
            self._rate_limit()
            
            if format.lower() == 'csv':
                # Export as CSV
                all_values = self.worksheet.get_all_values()
                csv_lines = []
                
                for row in all_values:
                    # Escape commas and quotes in CSV
                    escaped_row = []
                    for cell in row:
                        if ',' in cell or '"' in cell or '\n' in cell:
                            escaped_cell = '"' + cell.replace('"', '""') + '"'
                        else:
                            escaped_cell = cell
                        escaped_row.append(escaped_cell)
                    csv_lines.append(','.join(escaped_row))
                
                return '\n'.join(csv_lines)
            
            elif format.lower() == 'json':
                # Export as JSON
                records = self.worksheet.get_all_records()
                import json
                return json.dumps(records, indent=2, default=str)
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return None
                
        except APIError as e:
            logger.error(f"Google Sheets API error exporting data: {e}")
            return None
        except Exception as e:
            logger.error(f"Error exporting research data: {e}")
            return None
    
    def is_available(self) -> bool:
        """
        Check if Google Sheets integration is available and working.
        
        Returns:
            bool: True if available, False otherwise
        """
        return (
            self.credentials_path is not None and 
            os.path.exists(self.credentials_path) and 
            self.client is not None and 
            self.worksheet is not None
        )
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of Google Sheets integration.
        
        Returns:
            Dictionary with status information
        """
        status = {
            'available': self.is_available(),
            'authenticated': self.client is not None,
            'spreadsheet_connected': self.spreadsheet is not None,
            'worksheet_ready': self.worksheet is not None,
            'credentials_path': self.credentials_path,
            'spreadsheet_name': self.spreadsheet_name,
            'request_count': self._request_count
        }
        
        if self.spreadsheet:
            try:
                status['spreadsheet_url'] = self.spreadsheet.url
            except:
                pass
        
        return status