"""
Gradio User Interface for the Intelligent Research Assistant.
Provides a web-based interface for submitting research queries and viewing results.
"""

import gradio as gr
import threading
import time
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import json

from agents.main_orchestrator import MainOrchestrator, create_main_orchestrator
from agents.data_models import ResearchConfig, ProgressStatus, ReportStyle, ReportLength
from config import get_config
import structlog

logger = structlog.get_logger(__name__)

class GradioInterface:
    """
    Gradio web interface for the Intelligent Research Assistant.
    
    Features:
    - Query input with configurable parameters
    - Real-time progress display and result formatting
    - Error messaging and status indicators
    - Settings and history tabs
    - Analytics dashboard
    """
    
    def __init__(self):
        """Initialize the Gradio interface"""
        self.config = get_config()
        self.orchestrator: Optional[MainOrchestrator] = None
        self.current_progress: Optional[ProgressStatus] = None
        self.research_history: List[Dict[str, Any]] = []
        self.analytics_data: Dict[str, Any] = {}
        
        # UI state
        self.is_researching = False
        self.progress_thread: Optional[threading.Thread] = None
        self.progress_update_callback = None
        
        # Configuration validation
        self.config_errors = self._validate_configuration()
        
        logger.info("GradioInterface initialized", 
                   config_errors=len(self.config_errors))
    
    def _validate_configuration(self) -> List[str]:
        """
        Validate the current configuration.
        
        Returns:
            List of configuration errors
        """
        errors = []
        
        if not self.config.api.gemini_api_key:
            errors.append("Gemini API key is not configured")
        
        if self.config.api.google_sheets_credentials_path:
            import os
            if not os.path.exists(self.config.api.google_sheets_credentials_path):
                errors.append("Google Sheets credentials file not found")
        
        if not self.config.database.chroma_db_path:
            errors.append("ChromaDB path is not configured")
        
        return errors
    
    def _initialize_orchestrator(self) -> Tuple[bool, str]:
        """
        Initialize the orchestrator with progress callback.
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Check for critical configuration errors
            if not self.config.api.gemini_api_key:
                return False, "Gemini API key is required but not configured. Please check your .env file."
            
            if self.orchestrator is None:
                self.orchestrator = create_main_orchestrator(
                    config=self.config,
                    progress_callback=self._progress_callback
                )
                
                # Test the orchestrator
                health_status = self.orchestrator.health_check()
                if health_status.get('orchestrator') != 'healthy':
                    return False, "Orchestrator health check failed"
            
            return True, ""
        except Exception as e:
            logger.error("Failed to initialize orchestrator", error=str(e))
            return False, f"Failed to initialize research system: {str(e)}"
    
    def _progress_callback(self, progress: ProgressStatus):
        """
        Callback function for progress updates from orchestrator.
        
        Args:
            progress: Progress status from orchestrator
        """
        self.current_progress = progress
        logger.debug("Progress update received", 
                    step=progress.current_step,
                    percentage=progress.completion_percentage)
        
        # Trigger UI update if callback is set
        if self.progress_update_callback:
            try:
                self.progress_update_callback(self._format_progress_display(progress))
            except Exception as e:
                logger.warning("Progress update callback failed", error=str(e))
    
    def _format_progress_display(self, progress: Optional[ProgressStatus]) -> str:
        """
        Format progress status for display in the UI.
        
        Args:
            progress: Progress status object
            
        Returns:
            Formatted progress string
        """
        if not progress:
            return "No active research"
        
        progress_bar = "█" * int(progress.completion_percentage / 5) + "░" * (20 - int(progress.completion_percentage / 5))
        
        return f"""
**Research Progress: {progress.completion_percentage:.1f}%**

{progress_bar}

**Current Step:** {progress.current_step.replace('_', ' ').title()}
**Status:** {progress.status_message}
**Estimated Time Remaining:** {progress.estimated_time_remaining}s

**Completed Agents:** {', '.join(progress.completed_agents) if progress.completed_agents else 'None'}
**Failed Agents:** {', '.join(progress.failed_agents) if progress.failed_agents else 'None'}
"""
    
    def _format_research_result(self, result) -> Tuple[str, str, str]:
        """
        Format research result for display in the UI.
        
        Args:
            result: ResearchResult object
            
        Returns:
            Tuple of (report_markdown, metadata_info, status_message)
        """
        if not result.success:
            error_msg = f"❌ **Research Failed**\n\n**Error:** {result.error_message}\n\n"
            if result.report:
                error_msg += f"**Partial Results Available:**\n{result.report.executive_summary}"
            return error_msg, self._format_metadata(result.metadata), "Research failed"
        
        report = result.report
        
        # Format the main report
        report_markdown = f"""
# Research Report: {result.query}

## Executive Summary
{report.executive_summary}

## Key Findings
"""
        
        for i, finding in enumerate(report.key_findings, 1):
            report_markdown += f"{i}. {finding}\n"
        
        report_markdown += f"""
## Detailed Analysis
{report.detailed_analysis}

## Recommendations
"""
        
        for i, rec in enumerate(report.recommendations, 1):
            report_markdown += f"{i}. {rec}\n"
        
        report_markdown += "\n## Sources\n"
        for i, source in enumerate(report.sources, 1):
            report_markdown += f"{i}. {source}\n"
        
        # Format metadata
        metadata_info = self._format_metadata(result.metadata)
        
        status_message = f"✅ Research completed successfully in {result.execution_time:.1f}s with {result.source_count} sources"
        
        return report_markdown, metadata_info, status_message
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Format metadata for display.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Formatted metadata string
        """
        if not metadata:
            return "No metadata available"
        
        info = f"""
**Execution Time:** {metadata.get('execution_time', 0):.1f} seconds
**Data Sources:**
- Web Search Results: {metadata.get('data_sources', {}).get('web_search_results', 0)}
- Scraped Pages: {metadata.get('data_sources', {}).get('scraped_pages', 0)}
- Vector Documents: {metadata.get('data_sources', {}).get('vector_documents', 0)}

**Agents Used:** {', '.join(metadata.get('agents_used', []))}
"""
        
        if metadata.get('failed_agents'):
            info += f"**Failed Agents:** {', '.join(metadata.get('failed_agents', []))}\n"
        
        if metadata.get('warnings'):
            info += f"**Warnings:** {len(metadata.get('warnings', []))} warnings\n"
        
        quality_assessment = metadata.get('quality_assessment', {})
        if quality_assessment:
            info += f"""
**Quality Assessment:**
- Overall Quality: {quality_assessment.get('overall_quality', 'Unknown')}
- Data Completeness: {quality_assessment.get('data_completeness', 0):.1%}
- Source Diversity: {quality_assessment.get('source_diversity', 0):.1%}
- Credibility Score: {quality_assessment.get('credibility_score', 0):.1%}
"""
        
        return info
    
    def conduct_research(
        self,
        query: str,
        max_sources: int,
        enable_web_scraping: bool,
        enable_vector_search: bool,
        report_style: str,
        report_length: str,
        timeout_seconds: int
    ) -> Tuple[str, str, str, str]:
        """
        Conduct research based on user input with real-time progress updates.
        
        Args:
            query: Research query
            max_sources: Maximum number of sources
            enable_web_scraping: Whether to enable web scraping
            enable_vector_search: Whether to enable vector search
            report_style: Report style (academic, casual, technical)
            report_length: Report length (short, medium, long)
            timeout_seconds: Timeout in seconds
            
        Returns:
            Tuple of (report, metadata, status, progress)
        """
        # Input validation
        if not query.strip():
            return "Please enter a research query.", "", "❌ No query provided", "No active research"
        
        if len(query.strip()) < 10:
            return "Please enter a more detailed research query (at least 10 characters).", "", "❌ Query too short", "No active research"
        
        if self.is_researching:
            return "Research already in progress. Please wait for completion.", "", "⚠️ Research in progress", self._format_progress_display(self.current_progress)
        
        # Configuration validation
        if self.config_errors:
            error_msg = "Configuration errors detected:\n" + "\n".join(f"- {error}" for error in self.config_errors)
            return error_msg, "", "❌ Configuration errors", "Configuration validation failed"
        
        # Initialize orchestrator with validation
        success, error_msg = self._initialize_orchestrator()
        if not success:
            return f"Failed to initialize research system: {error_msg}", "", "❌ System initialization failed", "No active research"
        
        self.is_researching = True
        self.current_progress = None
        
        try:
            # Validate configuration parameters
            if max_sources < 1 or max_sources > 50:
                return "Maximum sources must be between 1 and 50.", "", "❌ Invalid configuration", "No active research"
            
            if timeout_seconds < 30 or timeout_seconds > 600:
                return "Timeout must be between 30 and 600 seconds.", "", "❌ Invalid configuration", "No active research"
            
            # Create research configuration
            try:
                config = ResearchConfig(
                    max_sources=max_sources,
                    enable_web_scraping=enable_web_scraping,
                    enable_vector_search=enable_vector_search,
                    report_style=ReportStyle(report_style),
                    report_length=ReportLength(report_length),
                    timeout_seconds=timeout_seconds
                )
            except ValueError as e:
                return f"Invalid configuration: {str(e)}", "", "❌ Configuration error", "No active research"
            
            logger.info("Starting research", 
                       query=query[:100], 
                       max_sources=max_sources,
                       timeout=timeout_seconds)
            
            # Initialize progress tracking
            initial_progress = "🚀 **Starting Research**\n\nInitializing research system and analyzing query..."
            
            # Start research in a separate thread
            result_container = {}
            exception_container = {}
            
            def research_thread():
                try:
                    result = self.orchestrator.research(query, config)
                    result_container['result'] = result
                except Exception as e:
                    logger.error("Research thread failed", error=str(e))
                    exception_container['error'] = str(e)
                    exception_container['type'] = type(e).__name__
            
            thread = threading.Thread(target=research_thread, daemon=True)
            thread.start()
            
            # Wait for completion with progress monitoring
            start_time = time.time()
            last_progress_update = time.time()
            
            while thread.is_alive():
                thread.join(timeout=2.0)  # Check every 2 seconds
                
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                # Update progress display
                if self.current_progress:
                    progress_display = self._format_progress_display(self.current_progress)
                else:
                    # Fallback progress display
                    progress_display = f"""
🔄 **Research in Progress**

**Elapsed Time:** {elapsed_time:.1f}s
**Status:** Processing your query...
**Timeout:** {timeout_seconds}s

Please wait while we gather and analyze information from multiple sources.
"""
                
                # Check for timeout
                if elapsed_time > timeout_seconds + 15:  # Add 15s buffer for cleanup
                    logger.warning("Research timeout exceeded", elapsed_time=elapsed_time)
                    try:
                        if self.orchestrator:
                            self.orchestrator.cancel_research()
                    except:
                        pass
                    break
            
            # Process results
            if 'result' in result_container:
                result = result_container['result']
                
                # Add to local history
                history_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'query': query,
                    'success': result.success,
                    'execution_time': result.execution_time,
                    'source_count': result.source_count,
                    'summary': result.report.executive_summary[:200] + "..." if result.report and len(result.report.executive_summary) > 200 else result.report.executive_summary if result.report else "No summary available"
                }
                
                self.research_history.insert(0, history_entry)
                self.research_history = self.research_history[:20]  # Keep only last 20
                
                # Format result for display
                report_md, metadata_info, status_msg = self._format_research_result(result)
                
                logger.info("Research completed successfully", 
                           execution_time=result.execution_time,
                           source_count=result.source_count,
                           success=result.success)
                
                return report_md, metadata_info, status_msg, "✅ Research completed successfully"
                
            elif 'error' in exception_container:
                error_type = exception_container.get('type', 'Unknown')
                error_msg = exception_container['error']
                
                logger.error("Research failed with exception", 
                           error_type=error_type, 
                           error_msg=error_msg)
                
                # Create detailed error message
                detailed_error = f"""
# Research Failed

**Error Type:** {error_type}
**Error Message:** {error_msg}

**Possible Solutions:**
- Check your internet connection
- Verify API keys are configured correctly
- Try a simpler query
- Reduce the number of sources or timeout
- Check system logs for more details

**Query:** {query}
"""
                
                return detailed_error, "", f"❌ Research failed ({error_type})", "Research failed"
            else:
                # Timeout case
                logger.warning("Research timed out", 
                             elapsed_time=time.time() - start_time,
                             timeout=timeout_seconds)
                
                timeout_msg = f"""
# Research Timed Out

The research operation exceeded the {timeout_seconds} second timeout limit.

**Suggestions:**
- Increase the timeout duration
- Simplify your research query
- Reduce the maximum number of sources
- Try again with a more specific query

**Query:** {query}
**Elapsed Time:** {time.time() - start_time:.1f} seconds
"""
                
                return timeout_msg, "", "⚠️ Research timeout", "Research timed out"
                
        except Exception as e:
            logger.error("Research execution failed with unexpected error", error=str(e))
            
            error_msg = f"""
# Unexpected Error

An unexpected error occurred during research execution.

**Error:** {str(e)}
**Query:** {query}

Please try again or contact support if the problem persists.
"""
            
            return error_msg, "", "❌ Unexpected error", "Research failed"
        
        finally:
            self.is_researching = False
            self.current_progress = None
            self.progress_update_callback = None
    
    def get_research_history(self) -> str:
        """
        Get formatted research history for display.
        
        Returns:
            Formatted history string
        """
        # Try to get history from Google Sheets if available
        if self.orchestrator:
            try:
                sheets_history = self.orchestrator.get_research_history(20)
                if sheets_history:
                    self.research_history = sheets_history
            except Exception as e:
                logger.warning("Failed to load history from Google Sheets", error=str(e))
        
        if not self.research_history:
            return "No research history available."
        
        history_md = "# Research History\n\n"
        
        for i, entry in enumerate(self.research_history, 1):
            # Handle both local and Google Sheets format
            if isinstance(entry, dict):
                status_icon = "✅" if entry.get('success', True) else "❌"
                timestamp_str = entry.get('timestamp', '')
                if timestamp_str:
                    try:
                        if 'T' in timestamp_str:  # ISO format
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        else:  # Assume it's already formatted
                            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                        timestamp_display = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        timestamp_display = timestamp_str
                else:
                    timestamp_display = "Unknown"
                
                query = entry.get('query', 'Unknown query')
                execution_time = entry.get('execution_time', 0)
                source_count = entry.get('source_count', 0)
                summary = entry.get('summary', 'No summary available')
                
                history_md += f"""
## {i}. {status_icon} {query[:100]}{'...' if len(query) > 100 else ''}

**Time:** {timestamp_display}  
**Duration:** {execution_time:.1f}s  
**Sources:** {source_count}  
**Summary:** {summary}

---
"""
        
        return history_md
    
    def search_research_history(self, search_query: str) -> str:
        """
        Search research history by query.
        
        Args:
            search_query: Search term
            
        Returns:
            Formatted search results
        """
        if not search_query.strip():
            return self.get_research_history()
        
        # Try to search in Google Sheets if available
        if self.orchestrator:
            try:
                search_results = self.orchestrator.search_research_history(search_query, 10)
                if search_results:
                    history_md = f"# Search Results for '{search_query}'\n\n"
                    
                    for i, entry in enumerate(search_results, 1):
                        status_icon = "✅" if entry.get('success', True) else "❌"
                        timestamp_str = entry.get('timestamp', '')
                        try:
                            if 'T' in timestamp_str:
                                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            else:
                                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                            timestamp_display = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            timestamp_display = timestamp_str
                        
                        query = entry.get('query', 'Unknown query')
                        execution_time = entry.get('execution_time', 0)
                        source_count = entry.get('source_count', 0)
                        summary = entry.get('summary', 'No summary available')
                        
                        history_md += f"""
## {i}. {status_icon} {query}

**Time:** {timestamp_display}  
**Duration:** {execution_time:.1f}s  
**Sources:** {source_count}  
**Summary:** {summary}

---
"""
                    return history_md
            except Exception as e:
                logger.warning("Failed to search history in Google Sheets", error=str(e))
        
        # Fallback to local search
        search_term = search_query.lower()
        filtered_history = [
            entry for entry in self.research_history
            if search_term in entry.get('query', '').lower() or 
               search_term in entry.get('summary', '').lower()
        ]
        
        if not filtered_history:
            return f"No results found for '{search_query}'"
        
        history_md = f"# Search Results for '{search_query}'\n\n"
        
        for i, entry in enumerate(filtered_history, 1):
            status_icon = "✅" if entry.get('success', True) else "❌"
            timestamp_str = entry.get('timestamp', '')
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                timestamp_display = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            except:
                timestamp_display = timestamp_str
            
            query = entry.get('query', 'Unknown query')
            execution_time = entry.get('execution_time', 0)
            source_count = entry.get('source_count', 0)
            summary = entry.get('summary', 'No summary available')
            
            history_md += f"""
## {i}. {status_icon} {query}

**Time:** {timestamp_display}  
**Duration:** {execution_time:.1f}s  
**Sources:** {source_count}  
**Summary:** {summary}

---
"""
        
        return history_md
    
    def get_analytics_dashboard(self) -> str:
        """
        Get analytics dashboard for display.
        
        Returns:
            Formatted analytics string
        """
        # Try to get analytics from Google Sheets if available
        analytics_data = {}
        if self.orchestrator:
            try:
                analytics_data = self.orchestrator.get_research_analytics()
                if analytics_data:
                    self.analytics_data = analytics_data
            except Exception as e:
                logger.warning("Failed to load analytics from Google Sheets", error=str(e))
        
        # Use Google Sheets analytics if available, otherwise calculate from local history
        if analytics_data:
            analytics_md = f"""
# Analytics Dashboard

## Overview Statistics
- **Total Research Queries:** {analytics_data.get('total_research', 0)}
- **Successful Completions:** {analytics_data.get('successful_research', 0)}
- **Success Rate:** {analytics_data.get('success_rate', 0):.1f}%
- **Total Sources Processed:** {analytics_data.get('total_sources', 0)}

## Performance Metrics
- **Average Execution Time:** {analytics_data.get('avg_execution_time', 0):.1f} seconds
- **Average Sources per Research:** {analytics_data.get('avg_sources_per_research', 0):.1f}
- **Average Processing Time:** {analytics_data.get('avg_processing_time', 0):.1f} seconds

## Data Source Breakdown
- **Web Search Results:** {analytics_data.get('total_web_search_results', 0)}
- **Scraped Pages:** {analytics_data.get('total_scraped_pages', 0)}
- **Vector Database Documents:** {analytics_data.get('total_vector_documents', 0)}

## System Health
- **Google Sheets Integration:** {'✅ Connected' if analytics_data.get('sheets_available', False) else '❌ Not Available'}
- **Last Updated:** {analytics_data.get('last_updated', 'Unknown')}
"""
        else:
            # Fallback to local analytics
            if not self.research_history:
                return "No analytics data available. Conduct some research first."
            
            # Calculate analytics from local history
            total_research = len(self.research_history)
            successful_research = sum(1 for entry in self.research_history if entry.get('success', True))
            success_rate = (successful_research / total_research) * 100 if total_research > 0 else 0
            
            avg_execution_time = sum(entry.get('execution_time', 0) for entry in self.research_history) / total_research
            avg_sources = sum(entry.get('source_count', 0) for entry in self.research_history) / total_research
            
            total_sources = sum(entry.get('source_count', 0) for entry in self.research_history)
            
            analytics_md = f"""
# Analytics Dashboard (Local Data)

## Overview Statistics
- **Total Research Queries:** {total_research}
- **Successful Completions:** {successful_research}
- **Success Rate:** {success_rate:.1f}%
- **Total Sources Processed:** {total_sources}

## Performance Metrics
- **Average Execution Time:** {avg_execution_time:.1f} seconds
- **Average Sources per Research:** {avg_sources:.1f}
- **Sources per Second:** {total_sources / sum(entry.get('execution_time', 1) for entry in self.research_history):.2f}

## Recent Activity
"""
            
            # Add recent activity
            recent_entries = self.research_history[:5]
            for entry in recent_entries:
                status_icon = "✅" if entry.get('success', True) else "❌"
                timestamp_str = entry.get('timestamp', '')
                try:
                    timestamp = datetime.fromisoformat(timestamp_str).strftime("%m-%d %H:%M")
                except:
                    timestamp = timestamp_str
                query = entry.get('query', 'Unknown query')
                analytics_md += f"- {status_icon} {timestamp}: {query[:50]}{'...' if len(query) > 50 else ''}\n"
        
        return analytics_md
    
    def validate_system_configuration(self) -> str:
        """
        Validate system configuration and return status.
        
        Returns:
            Formatted validation status
        """
        validation_md = "# System Configuration Validation\n\n"
        
        # Re-validate configuration
        self.config_errors = self._validate_configuration()
        
        if not self.config_errors:
            validation_md += "✅ **All configuration checks passed!**\n\n"
        else:
            validation_md += "❌ **Configuration issues detected:**\n\n"
            for error in self.config_errors:
                validation_md += f"- {error}\n"
            validation_md += "\n"
        
        # Test orchestrator initialization
        validation_md += "## Orchestrator Initialization Test\n"
        try:
            success, error_msg = self._initialize_orchestrator()
            if success:
                validation_md += "✅ Orchestrator initialized successfully\n\n"
            else:
                validation_md += f"❌ Orchestrator initialization failed: {error_msg}\n\n"
        except Exception as e:
            validation_md += f"❌ Orchestrator test failed: {str(e)}\n\n"
        
        # API connectivity tests
        validation_md += "## API Connectivity\n"
        
        if self.config.api.gemini_api_key:
            validation_md += "✅ Gemini API key configured\n"
        else:
            validation_md += "❌ Gemini API key not configured\n"
        
        if self.config.api.serpapi_key:
            validation_md += "✅ SerpAPI key configured\n"
        else:
            validation_md += "⚠️ SerpAPI key not configured (will use DuckDuckGo fallback)\n"
        
        if self.config.api.google_sheets_credentials_path:
            import os
            if os.path.exists(self.config.api.google_sheets_credentials_path):
                validation_md += "✅ Google Sheets credentials file found\n"
            else:
                validation_md += "❌ Google Sheets credentials file not found\n"
        else:
            validation_md += "⚠️ Google Sheets not configured\n"
        
        # Database checks
        validation_md += "\n## Database Configuration\n"
        import os
        
        if self.config.database.chroma_db_path:
            if os.path.exists(self.config.database.chroma_db_path):
                validation_md += f"✅ ChromaDB directory exists: {self.config.database.chroma_db_path}\n"
            else:
                validation_md += f"⚠️ ChromaDB directory will be created: {self.config.database.chroma_db_path}\n"
        else:
            validation_md += "❌ ChromaDB path not configured\n"
        
        return validation_md
    
    def get_system_status(self) -> str:
        """
        Get system status information.
        
        Returns:
            Formatted system status string
        """
        status_md = "# System Status\n\n"
        
        # Check orchestrator status
        if self.orchestrator:
            try:
                health_status = self.orchestrator.health_check()
                status_md += "## Orchestrator Status: ✅ Healthy\n\n"
                
                # Agent status
                status_md += "### Agent Status\n"
                for agent_name, agent_status in health_status.get('agents', {}).items():
                    if isinstance(agent_status, dict) and 'error' in agent_status:
                        status_md += f"- **{agent_name.replace('_', ' ').title()}:** ❌ Error - {agent_status['error']}\n"
                    else:
                        status_md += f"- **{agent_name.replace('_', ' ').title()}:** ✅ Available\n"
                
                # Executor status
                executor_status = health_status.get('executor', {})
                status_md += f"\n### Executor Status\n"
                status_md += f"- **Active:** {'✅ Yes' if executor_status.get('active', False) else '❌ No'}\n"
                status_md += f"- **Max Workers:** {executor_status.get('max_workers', 'Unknown')}\n"
                
                # Current research status
                status_md += f"- **Current Research Active:** {'✅ Yes' if health_status.get('current_research_active', False) else '❌ No'}\n"
                
            except Exception as e:
                status_md += f"## Orchestrator Status: ❌ Error - {str(e)}\n\n"
        else:
            status_md += "## Orchestrator Status: ❌ Not Initialized\n\n"
        
        # Google Sheets status
        if self.orchestrator:
            try:
                sheets_status = self.orchestrator.get_sheets_status()
                status_md += "### Google Sheets Integration\n"
                status_md += f"- **Available:** {'✅ Yes' if sheets_status.get('available', False) else '❌ No'}\n"
                status_md += f"- **Spreadsheet:** {sheets_status.get('spreadsheet_name', 'Not configured')}\n"
                if sheets_status.get('error'):
                    status_md += f"- **Error:** {sheets_status['error']}\n"
            except Exception as e:
                status_md += f"### Google Sheets Integration: ❌ Error - {str(e)}\n"
        
        # Configuration status
        status_md += "\n### Configuration\n"
        status_md += f"- **Gemini API:** {'✅ Configured' if self.config.api.gemini_api_key else '❌ Not configured'}\n"
        status_md += f"- **SerpAPI:** {'✅ Configured' if self.config.api.serpapi_key else '❌ Not configured (will use DuckDuckGo)'}\n"
        status_md += f"- **Google Sheets:** {'✅ Configured' if self.config.api.google_sheets_credentials_path else '❌ Not configured'}\n"
        status_md += f"- **Database Path:** {self.config.database.chroma_db_path}\n"
        
        return status_md
    
    def update_api_configuration(self, gemini_key: str, serpapi_key: str, sheets_path: str) -> str:
        """
        Update API configuration (placeholder for future implementation).
        
        Args:
            gemini_key: Gemini API key
            serpapi_key: SerpAPI key
            sheets_path: Google Sheets credentials path
            
        Returns:
            Status message
        """
        # This is a placeholder - in a full implementation, this would update the .env file
        # or configuration and reinitialize the orchestrator
        return """
⚠️ **Configuration Update Not Implemented**

To update API keys and configuration:

1. **Edit your .env file** with the new values:
   ```
   GEMINI_API_KEY=your_new_key
   SERPAPI_KEY=your_new_key
   GOOGLE_SHEETS_CREDENTIALS_PATH=path/to/credentials.json
   ```

2. **Restart the application** for changes to take effect.

**Current Configuration:**
- Gemini API: {'✅ Configured' if self.config.api.gemini_api_key else '❌ Not configured'}
- SerpAPI: {'✅ Configured' if self.config.api.serpapi_key else '❌ Not configured'}
- Google Sheets: {'✅ Configured' if self.config.api.google_sheets_credentials_path else '❌ Not configured'}
"""
    
    def create_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(
            title="Intelligent Research Assistant",
            theme=gr.themes.Soft(),
            css="""
            .research-container { padding: 20px; }
            .progress-display { background-color: #f0f0f0; padding: 15px; border-radius: 5px; }
            .status-success { color: #28a745; }
            .status-error { color: #dc3545; }
            .status-warning { color: #ffc107; }
            """
        ) as interface:
            
            gr.Markdown("""
            # 🔬 Intelligent Research Assistant
            
            Welcome to the AI-powered research assistant! Submit your research query below and get comprehensive, 
            well-sourced reports generated automatically from multiple data sources.
            """)
            
            with gr.Tabs():
                # Main Research Tab
                with gr.Tab("🔍 Research", elem_id="research-tab"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            # Query input
                            query_input = gr.Textbox(
                                label="Research Query",
                                placeholder="Enter your research question or topic here...",
                                lines=3,
                                max_lines=5
                            )
                            
                            # Configuration options
                            with gr.Accordion("Research Configuration", open=False):
                                with gr.Row():
                                    max_sources = gr.Slider(
                                        minimum=5,
                                        maximum=50,
                                        value=10,
                                        step=5,
                                        label="Maximum Sources"
                                    )
                                    timeout_seconds = gr.Slider(
                                        minimum=60,
                                        maximum=300,
                                        value=120,
                                        step=30,
                                        label="Timeout (seconds)"
                                    )
                                
                                with gr.Row():
                                    enable_web_scraping = gr.Checkbox(
                                        label="Enable Web Scraping",
                                        value=True
                                    )
                                    enable_vector_search = gr.Checkbox(
                                        label="Enable Vector Database Search",
                                        value=True
                                    )
                                
                                with gr.Row():
                                    report_style = gr.Dropdown(
                                        choices=["academic", "casual", "technical"],
                                        value="academic",
                                        label="Report Style"
                                    )
                                    report_length = gr.Dropdown(
                                        choices=["short", "medium", "long"],
                                        value="medium",
                                        label="Report Length"
                                    )
                            
                            # Research button
                            research_btn = gr.Button(
                                "🚀 Start Research",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=1):
                            # Progress display
                            progress_display = gr.Markdown(
                                "No active research",
                                elem_classes=["progress-display"]
                            )
                            
                            # Status indicator
                            status_display = gr.Markdown(
                                "Ready to start research",
                                elem_classes=["status-display"]
                            )
                    
                    # Results section
                    with gr.Row():
                        with gr.Column():
                            # Research report
                            report_output = gr.Markdown(
                                "Research results will appear here...",
                                label="Research Report"
                            )
                    
                    with gr.Row():
                        with gr.Column():
                            # Metadata and details
                            metadata_output = gr.Markdown(
                                "Research metadata will appear here...",
                                label="Research Details"
                            )
                
                # History Tab
                with gr.Tab("📚 History", elem_id="history-tab"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            history_search = gr.Textbox(
                                label="Search History",
                                placeholder="Enter search terms to filter history...",
                                lines=1
                            )
                        with gr.Column(scale=1):
                            search_history_btn = gr.Button("🔍 Search")
                            refresh_history_btn = gr.Button("🔄 Refresh All")
                    
                    history_display = gr.Markdown(
                        "Research history will appear here...",
                        label="Research History"
                    )
                
                # Analytics Tab
                with gr.Tab("📊 Analytics", elem_id="analytics-tab"):
                    with gr.Row():
                        refresh_analytics_btn = gr.Button("🔄 Refresh Analytics")
                        system_status_btn = gr.Button("🔧 System Status")
                    
                    analytics_display = gr.Markdown(
                        "Analytics data will appear here...",
                        label="Analytics Dashboard"
                    )
                
                # Settings Tab
                with gr.Tab("⚙️ Settings", elem_id="settings-tab"):
                    gr.Markdown("## API Configuration")
                    
                    with gr.Accordion("API Keys & Credentials", open=False):
                        gr.Markdown("""
                        **Note:** API key updates require application restart to take effect.
                        Current configuration is loaded from your .env file.
                        """)
                        
                        with gr.Row():
                            gemini_key_input = gr.Textbox(
                                label="Gemini API Key",
                                placeholder="Enter your Gemini API key...",
                                type="password",
                                value="***configured***" if self.config.api.gemini_api_key else ""
                            )
                            serpapi_key_input = gr.Textbox(
                                label="SerpAPI Key (Optional)",
                                placeholder="Enter your SerpAPI key...",
                                type="password",
                                value="***configured***" if self.config.api.serpapi_key else ""
                            )
                        
                        sheets_path_input = gr.Textbox(
                            label="Google Sheets Credentials Path",
                            placeholder="Path to Google Sheets service account JSON file...",
                            value=self.config.api.google_sheets_credentials_path or ""
                        )
                        
                        with gr.Row():
                            update_config_btn = gr.Button("💾 Update Configuration")
                            validate_config_btn = gr.Button("🔍 Validate Configuration")
                        
                        config_status = gr.Markdown("Configuration status will appear here...")
                    
                    gr.Markdown("## Default Research Parameters")
                    
                    with gr.Accordion("Default Settings", open=True):
                        with gr.Row():
                            default_max_sources = gr.Slider(
                                minimum=5,
                                maximum=50,
                                value=10,
                                step=5,
                                label="Default Maximum Sources"
                            )
                            default_timeout = gr.Slider(
                                minimum=60,
                                maximum=300,
                                value=120,
                                step=30,
                                label="Default Timeout (seconds)"
                            )
                        
                        with gr.Row():
                            default_web_scraping = gr.Checkbox(
                                label="Enable Web Scraping by Default",
                                value=True
                            )
                            default_vector_search = gr.Checkbox(
                                label="Enable Vector Search by Default",
                                value=True
                            )
                        
                        with gr.Row():
                            default_report_style = gr.Dropdown(
                                choices=["academic", "casual", "technical"],
                                value="academic",
                                label="Default Report Style"
                            )
                            default_report_length = gr.Dropdown(
                                choices=["short", "medium", "long"],
                                value="medium",
                                label="Default Report Length"
                            )
                        
                        gr.Markdown("""
                        **Note:** Default parameter changes are not persistent and will reset when the application restarts.
                        To make permanent changes, modify your configuration files.
                        """)
                    
                    gr.Markdown("## System Information")
                    system_info_display = gr.Markdown("Click 'System Status' in Analytics tab to view system information.")
            
            # Event handlers
            def start_research_with_progress(*args):
                """Wrapper to handle research with progress updates"""
                # Set up progress callback
                def update_progress(progress_text):
                    return progress_text
                
                self.progress_update_callback = update_progress
                
                # Start research
                return self.conduct_research(*args)
            
            research_btn.click(
                fn=start_research_with_progress,
                inputs=[
                    query_input,
                    max_sources,
                    enable_web_scraping,
                    enable_vector_search,
                    report_style,
                    report_length,
                    timeout_seconds
                ],
                outputs=[
                    report_output,
                    metadata_output,
                    status_display,
                    progress_display
                ]
            )
            
            # History tab event handlers
            refresh_history_btn.click(
                fn=self.get_research_history,
                outputs=history_display
            )
            
            search_history_btn.click(
                fn=self.search_research_history,
                inputs=history_search,
                outputs=history_display
            )
            
            # Analytics tab event handlers
            refresh_analytics_btn.click(
                fn=self.get_analytics_dashboard,
                outputs=analytics_display
            )
            
            system_status_btn.click(
                fn=self.get_system_status,
                outputs=analytics_display
            )
            
            # Settings tab event handlers
            update_config_btn.click(
                fn=self.update_api_configuration,
                inputs=[gemini_key_input, serpapi_key_input, sheets_path_input],
                outputs=config_status
            )
            
            validate_config_btn.click(
                fn=self.validate_system_configuration,
                outputs=config_status
            )
        
        return interface
    
    def launch(
        self,
        host: str = None,
        port: int = None,
        share: bool = False,
        debug: bool = False
    ):
        """
        Launch the Gradio interface.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            share: Whether to create a public link
            debug: Whether to enable debug mode
        """
        # Use config defaults if not specified
        host = host or self.config.gradio_host
        port = port or self.config.gradio_port
        
        interface = self.create_interface()
        
        logger.info("Launching Gradio interface", 
                   host=host, 
                   port=port, 
                   share=share, 
                   debug=debug)
        
        interface.launch(
            share=share,
            debug=debug,
            show_error=True,
            quiet=False
        )


def create_gradio_app() -> GradioInterface:
    """
    Factory function to create a GradioInterface instance.
    
    Returns:
        Configured GradioInterface instance
    """
    return GradioInterface()


if __name__ == "__main__":
    # For testing the interface directly
    app = create_gradio_app()
    app.launch(debug=True)