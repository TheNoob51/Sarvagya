"""
Main Orchestrator for the Intelligent Research Assistant.
Coordinates workflow between all agents with parallel execution, progress tracking, and error recovery.
"""

import asyncio
import time
import threading
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from enum import Enum
import statistics

import structlog
from agents.data_models import (
    ResearchConfig, ResearchResult, ProgressStatus, ResearchReport,
    SearchResult, ScrapedContent, Document, FactCheckResult, ReportStyle, ReportLength
)
from agents.router_agent import RouterAgent
from agents.web_search_agent import WebSearchAgent
from agents.web_scraper_agent import WebScraperAgent
from agents.vector_search_agent import VectorSearchAgent
from agents.fact_checker_agent import (
    FactCheckerAgent, InformationSource,
    create_information_source_from_search_result,
    create_information_source_from_scraped_content,
    create_information_source_from_document
)
from agents.summarizer_agent import (
    SummarizerAgent, SourceInfo, ReportConfig,
    create_source_info_from_search_result,
    create_source_info_from_scraped_content,
    create_source_info_from_document
)
from utils.config import AppConfig
from utils.google_sheets_handler import GoogleSheetsHandler

logger = structlog.get_logger(__name__)

class ResearchStage(Enum):
    """Stages of the research process"""
    INITIALIZING = "initializing"
    PLANNING = "planning"
    DATA_COLLECTION = "data_collection"
    FACT_CHECKING = "fact_checking"
    REPORT_GENERATION = "report_generation"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AgentResult:
    """Result from an individual agent execution"""
    agent_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResearchContext:
    """Context object that tracks the entire research process"""
    query: str
    config: ResearchConfig
    start_time: float
    current_stage: ResearchStage = ResearchStage.INITIALIZING
    progress_percentage: float = 0.0
    
    # Agent results
    router_result: Optional[AgentResult] = None
    web_search_result: Optional[AgentResult] = None
    scraper_result: Optional[AgentResult] = None
    vector_search_result: Optional[AgentResult] = None
    fact_check_result: Optional[AgentResult] = None
    summarizer_result: Optional[AgentResult] = None
    
    # Aggregated data
    all_search_results: List[SearchResult] = field(default_factory=list)
    all_scraped_content: List[ScrapedContent] = field(default_factory=list)
    all_vector_documents: List[Document] = field(default_factory=list)
    information_sources: List[InformationSource] = field(default_factory=list)
    source_infos: List[SourceInfo] = field(default_factory=list)
    
    # Progress tracking
    completed_agents: List[str] = field(default_factory=list)
    failed_agents: List[str] = field(default_factory=list)
    status_message: str = "Initializing research..."
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class MainOrchestrator:
    """
    Main Orchestrator that coordinates the research workflow between all agents.
    
    Features:
    - Parallel agent execution where possible
    - Real-time progress tracking and status updates
    - Timeout handling and error recovery
    - Result aggregation and data flow management
    - Performance monitoring and logging
    """
    
    # Stage completion percentages for progress tracking
    STAGE_PROGRESS = {
        ResearchStage.INITIALIZING: 5,
        ResearchStage.PLANNING: 15,
        ResearchStage.DATA_COLLECTION: 60,
        ResearchStage.FACT_CHECKING: 80,
        ResearchStage.REPORT_GENERATION: 95,
        ResearchStage.COMPLETED: 100,
        ResearchStage.FAILED: 0
    }
    
    def __init__(
        self,
        router_agent: RouterAgent,
        web_search_agent: WebSearchAgent,
        web_scraper_agent: WebScraperAgent,
        vector_search_agent: VectorSearchAgent,
        fact_checker_agent: FactCheckerAgent,
        summarizer_agent: SummarizerAgent,
        config: AppConfig,
        progress_callback: Optional[Callable[[ProgressStatus], None]] = None
    ):
        """
        Initialize the Main Orchestrator with all required agents.
        
        Args:
            router_agent: Router agent for query analysis
            web_search_agent: Web search agent
            web_scraper_agent: Web scraper agent
            vector_search_agent: Vector search agent
            fact_checker_agent: Fact checker agent
            summarizer_agent: Summarizer agent
            config: Application configuration
            progress_callback: Optional callback for progress updates
        """
        self.router_agent = router_agent
        self.web_search_agent = web_search_agent
        self.web_scraper_agent = web_scraper_agent
        self.vector_search_agent = vector_search_agent
        self.fact_checker_agent = fact_checker_agent
        self.summarizer_agent = summarizer_agent
        self.config = config
        self.progress_callback = progress_callback
        
        # Initialize Google Sheets handler
        self.sheets_handler = GoogleSheetsHandler(
            credentials_path=config.api.google_sheets_credentials_path,
            spreadsheet_name=config.sheets.spreadsheet_name
        )
        
        # Try to initialize Google Sheets (non-blocking)
        try:
            self.sheets_handler.initialize()
        except Exception as e:
            logger.warning("Google Sheets initialization failed", error=str(e))
        
        # Execution control
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.current_research: Optional[ResearchContext] = None
        self._cancel_requested = False
        
        logger.info("MainOrchestrator initialized", 
                   max_workers=4,
                   has_progress_callback=progress_callback is not None,
                   google_sheets_available=self.sheets_handler.is_available())
    
    def _save_to_sheets(self, result: ResearchResult) -> None:
        """
        Save research result to Google Sheets in a separate thread.
        
        Args:
            result: ResearchResult to save
        """
        def save_async():
            try:
                if self.sheets_handler.is_available():
                    success = self.sheets_handler.save_research(result)
                    if success:
                        logger.info("Research saved to Google Sheets", query=result.query[:50])
                    else:
                        logger.warning("Failed to save research to Google Sheets")
                else:
                    logger.debug("Google Sheets not available, skipping save")
            except Exception as e:
                logger.error("Error saving to Google Sheets", error=str(e))
        
        # Run in background thread to avoid blocking
        threading.Thread(target=save_async, daemon=True).start()
    
    def get_research_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent research history from Google Sheets.
        
        Args:
            limit: Maximum number of entries to retrieve
            
        Returns:
            List of research entries
        """
        if not self.sheets_handler.is_available():
            logger.warning("Google Sheets not available for history retrieval")
            return []
        
        try:
            return self.sheets_handler.get_recent_research(limit)
        except Exception as e:
            logger.error("Error retrieving research history", error=str(e))
            return []
    
    def search_research_history(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search research history by query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching research entries
        """
        if not self.sheets_handler.is_available():
            logger.warning("Google Sheets not available for history search")
            return []
        
        try:
            return self.sheets_handler.search_research(query, limit)
        except Exception as e:
            logger.error("Error searching research history", error=str(e))
            return []
    
    def get_research_analytics(self) -> Dict[str, Any]:
        """
        Get research analytics from Google Sheets.
        
        Returns:
            Dictionary containing analytics data
        """
        if not self.sheets_handler.is_available():
            logger.warning("Google Sheets not available for analytics")
            return {}
        
        try:
            return self.sheets_handler.get_analytics()
        except Exception as e:
            logger.error("Error retrieving research analytics", error=str(e))
            return {}
    
    def get_sheets_status(self) -> Dict[str, Any]:
        """
        Get Google Sheets integration status.
        
        Returns:
            Dictionary with status information
        """
        return self.sheets_handler.get_status()
    
    def research(self, query: str, research_config: ResearchConfig = None) -> ResearchResult:
        """
        Execute the complete research workflow.
        
        Args:
            query: Research query to process
            research_config: Optional research configuration
            
        Returns:
            ResearchResult with complete findings and metadata
        """
        # Use default config if none provided
        if research_config is None:
            research_config = ResearchConfig()
        
        # Initialize research context
        context = ResearchContext(
            query=query,
            config=research_config,
            start_time=time.time()
        )
        self.current_research = context
        self._cancel_requested = False
        
        logger.info("Starting research workflow", 
                   query=query[:100],
                   timeout_seconds=research_config.timeout_seconds)
        
        try:
            # Stage 1: Query Analysis and Planning
            self._update_progress(context, ResearchStage.PLANNING, "Analyzing query and creating research plan...")
            if not self._execute_planning_stage(context):
                return self._create_failure_result(context, "Planning stage failed")
            
            # Stage 2: Parallel Data Collection
            self._update_progress(context, ResearchStage.DATA_COLLECTION, "Collecting data from multiple sources...")
            if not self._execute_data_collection_stage(context):
                return self._create_failure_result(context, "Data collection failed")
            
            # Stage 3: Fact Checking and Validation
            self._update_progress(context, ResearchStage.FACT_CHECKING, "Validating information and checking facts...")
            if not self._execute_fact_checking_stage(context):
                return self._create_failure_result(context, "Fact checking failed")
            
            # Stage 4: Report Generation
            self._update_progress(context, ResearchStage.REPORT_GENERATION, "Generating comprehensive research report...")
            if not self._execute_report_generation_stage(context):
                return self._create_failure_result(context, "Report generation failed")
            
            # Stage 5: Result Validation and Quality Assessment
            self._update_progress(context, ResearchStage.COMPLETED, "Validating results and generating analytics...")
            
            # Perform comprehensive result validation
            validation_results = self._validate_agent_results(context)
            context.warnings.extend([f"Quality issue: {issue}" for issue in validation_results["validation_issues"]])
            
            # Monitor performance metrics
            performance_metrics = self._monitor_performance(context)
            
            # Log quality and performance summary
            logger.info("Research workflow completed with quality assessment",
                       overall_quality=validation_results["overall_quality"],
                       data_completeness=validation_results["data_completeness"],
                       total_execution_time=performance_metrics["total_execution_time"],
                       successful_agents=len(context.completed_agents),
                       failed_agents=len(context.failed_agents))
            
            # Stage 6: Final Completion
            self._update_progress(context, ResearchStage.COMPLETED, "Research completed successfully!")
            return self._create_success_result(context)
            
        except Exception as e:
            logger.error("Research workflow failed with unexpected error", error=str(e))
            context.errors.append(f"Unexpected error: {str(e)}")
            return self._create_failure_result(context, f"Unexpected error: {str(e)}")
        
        finally:
            self.current_research = None
    
    def _execute_planning_stage(self, context: ResearchContext) -> bool:
        """
        Execute the planning stage using the Router Agent.
        
        Args:
            context: Research context
            
        Returns:
            True if successful, False otherwise
        """
        try:
            start_time = time.time()
            
            # Execute router agent with timeout
            future = self.executor.submit(
                self._execute_with_timeout,
                self.router_agent.analyze_query,
                context.query,
                30  # 30 second timeout for planning
            )
            
            research_plan = future.result(timeout=35)
            
            if research_plan is None:
                context.errors.append("Router agent returned no plan")
                return False
            
            # Store result
            context.router_result = AgentResult(
                agent_name="router",
                success=True,
                data=research_plan,
                execution_time=time.time() - start_time
            )
            
            context.completed_agents.append("router")
            logger.info("Planning stage completed successfully", 
                       execution_time=context.router_result.execution_time)
            
            return True
            
        except Exception as e:
            logger.error("Planning stage failed", error=str(e))
            context.errors.append(f"Planning failed: {str(e)}")
            context.failed_agents.append("router")
            return False
    
    def _validate_data_flow(self, from_agent: str, to_agent: str, data: Any, context: ResearchContext) -> bool:
        """
        Validate data flow between agents with proper error handling.
        
        Args:
            from_agent: Source agent name
            to_agent: Destination agent name
            data: Data being passed
            context: Research context
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            if data is None:
                logger.warning("Data flow validation failed: None data", 
                             from_agent=from_agent, to_agent=to_agent)
                context.warnings.append(f"No data passed from {from_agent} to {to_agent}")
                return False
            
            # Validate specific data types based on agent flow
            if from_agent == "router" and to_agent in ["web_search", "web_scraper", "vector_search"]:
                if not hasattr(data, 'search_queries') or not hasattr(data, 'target_websites'):
                    logger.error("Invalid router data structure", from_agent=from_agent, to_agent=to_agent)
                    context.errors.append(f"Invalid data structure from {from_agent}")
                    return False
                
                if not data.search_queries and not data.target_websites:
                    logger.warning("Empty router data", from_agent=from_agent, to_agent=to_agent)
                    context.warnings.append(f"Empty search queries and target websites from {from_agent}")
                    return False
            
            elif to_agent == "fact_checker":
                # Validate that we have information sources
                if isinstance(data, list) and len(data) == 0:
                    logger.warning("No information sources for fact checking", from_agent=from_agent)
                    context.warnings.append("No information sources available for fact checking")
                    return False
            
            elif to_agent == "summarizer":
                # Validate that we have verified facts and sources
                if isinstance(data, tuple) and len(data) >= 2:
                    verified_facts, sources = data[0], data[1]
                    if not verified_facts and not sources:
                        logger.warning("No verified facts or sources for report generation")
                        context.warnings.append("No verified facts or sources available for report generation")
                        return False
                else:
                    logger.error("Invalid data format for summarizer", data_type=type(data))
                    context.errors.append("Invalid data format passed to summarizer")
                    return False
            
            logger.debug("Data flow validation passed", 
                        from_agent=from_agent, 
                        to_agent=to_agent,
                        data_type=type(data).__name__)
            return True
            
        except Exception as e:
            logger.error("Data flow validation error", 
                        from_agent=from_agent, 
                        to_agent=to_agent, 
                        error=str(e))
            context.errors.append(f"Data validation error between {from_agent} and {to_agent}: {str(e)}")
            return False
    
    def _execute_data_collection_stage(self, context: ResearchContext) -> bool:
        """
        Execute parallel data collection from multiple sources.
        
        Args:
            context: Research context
            
        Returns:
            True if at least one data source succeeded, False if all failed
        """
        if not context.router_result or not context.router_result.success:
            logger.warning("No research plan available, using default data collection strategy")
            # Use default strategy if planning failed
            use_web_search = True
            use_web_scraping = context.config.enable_web_scraping
            use_vector_search = context.config.enable_vector_search
            search_queries = [context.query]
            target_websites = []
        else:
            # Validate router data before using it
            if not self._validate_data_flow("router", "data_collection", context.router_result.data, context):
                logger.error("Router data validation failed, using fallback strategy")
                use_web_search = True
                use_web_scraping = context.config.enable_web_scraping
                use_vector_search = context.config.enable_vector_search
                search_queries = [context.query]
                target_websites = []
            else:
                research_plan = context.router_result.data
                use_web_search = research_plan.research_strategy.use_web_search
                use_web_scraping = research_plan.research_strategy.use_web_scraping and context.config.enable_web_scraping
                use_vector_search = research_plan.research_strategy.use_vector_search and context.config.enable_vector_search
                search_queries = research_plan.search_queries
                target_websites = research_plan.target_websites
        
        # Submit parallel tasks
        futures = {}
        
        if use_web_search:
            future = self.executor.submit(
                self._execute_web_search,
                search_queries,
                context.config.max_sources
            )
            futures['web_search'] = future
        
        if use_web_scraping and target_websites:
            future = self.executor.submit(
                self._execute_web_scraping,
                target_websites
            )
            futures['web_scraping'] = future
        
        if use_vector_search:
            future = self.executor.submit(
                self._execute_vector_search,
                context.query,
                min(context.config.max_sources // 2, 5)  # Limit vector results
            )
            futures['vector_search'] = future
        
        # Wait for results with timeout
        timeout_per_agent = min(context.config.timeout_seconds // 3, 45)  # Distribute timeout
        successful_agents = 0
        
        for agent_name, future in futures.items():
            try:
                result = future.result(timeout=timeout_per_agent)
                
                if agent_name == 'web_search' and result:
                    context.web_search_result = result
                    context.all_search_results.extend(result.data or [])
                    if result.success:
                        successful_agents += 1
                        context.completed_agents.append("web_search")
                    else:
                        context.failed_agents.append("web_search")
                
                elif agent_name == 'web_scraping' and result:
                    context.scraper_result = result
                    context.all_scraped_content.extend(result.data or [])
                    if result.success:
                        successful_agents += 1
                        context.completed_agents.append("web_scraper")
                    else:
                        context.failed_agents.append("web_scraper")
                
                elif agent_name == 'vector_search' and result:
                    context.vector_search_result = result
                    context.all_vector_documents.extend(result.data or [])
                    if result.success:
                        successful_agents += 1
                        context.completed_agents.append("vector_search")
                    else:
                        context.failed_agents.append("vector_search")
                
            except Exception as e:
                logger.error(f"{agent_name} failed", error=str(e))
                context.errors.append(f"{agent_name} failed: {str(e)}")
                context.failed_agents.append(agent_name)
        
        # Check if we have any data
        total_sources = len(context.all_search_results) + len(context.all_scraped_content) + len(context.all_vector_documents)
        
        if total_sources == 0:
            logger.error("No data collected from any source")
            context.errors.append("No data collected from any source")
            return False
        
        logger.info("Data collection completed", 
                   successful_agents=successful_agents,
                   total_sources=total_sources,
                   search_results=len(context.all_search_results),
                   scraped_content=len(context.all_scraped_content),
                   vector_documents=len(context.all_vector_documents))
        
        return True
    
    def _execute_web_search(self, queries: List[str], max_results: int) -> AgentResult:
        """Execute web search agent"""
        try:
            start_time = time.time()
            results = self.web_search_agent.search(queries, max_results)
            
            return AgentResult(
                agent_name="web_search",
                success=len(results) > 0,
                data=results,
                execution_time=time.time() - start_time,
                metadata={"result_count": len(results)}
            )
        except Exception as e:
            return AgentResult(
                agent_name="web_search",
                success=False,
                error=str(e),
                execution_time=time.time() - start_time if 'start_time' in locals() else 0
            )
    
    def _execute_web_scraping(self, urls: List[str]) -> AgentResult:
        """Execute web scraper agent"""
        try:
            start_time = time.time()
            scrape_results = self.web_scraper_agent.scrape_multiple_pages(urls)
            
            # Extract successful scrapes
            successful_content = [
                result.content for result in scrape_results 
                if result.success and result.content
            ]
            
            return AgentResult(
                agent_name="web_scraper",
                success=len(successful_content) > 0,
                data=successful_content,
                execution_time=time.time() - start_time,
                metadata={
                    "attempted_urls": len(urls),
                    "successful_scrapes": len(successful_content)
                }
            )
        except Exception as e:
            return AgentResult(
                agent_name="web_scraper",
                success=False,
                error=str(e),
                execution_time=time.time() - start_time if 'start_time' in locals() else 0
            )
    
    def _execute_vector_search(self, query: str, max_results: int) -> AgentResult:
        """Execute vector search agent"""
        try:
            start_time = time.time()
            documents = self.vector_search_agent.search(
                query=query,
                top_k=max_results,
                similarity_threshold=0.6
            )
            
            return AgentResult(
                agent_name="vector_search",
                success=len(documents) > 0,
                data=documents,
                execution_time=time.time() - start_time,
                metadata={"result_count": len(documents)}
            )
        except Exception as e:
            return AgentResult(
                agent_name="vector_search",
                success=False,
                error=str(e),
                execution_time=time.time() - start_time if 'start_time' in locals() else 0
            )    

    def _execute_fact_checking_stage(self, context: ResearchContext) -> bool:
        """
        Execute fact checking and validation stage.
        
        Args:
            context: Research context
            
        Returns:
            True if successful, False otherwise
        """
        try:
            start_time = time.time()
            
            # Convert all collected data to InformationSource objects
            information_sources = []
            
            # Add search results
            for search_result in context.all_search_results:
                info_source = create_information_source_from_search_result(search_result)
                information_sources.append(info_source)
            
            # Add scraped content
            for scraped_content in context.all_scraped_content:
                info_source = create_information_source_from_scraped_content(scraped_content)
                information_sources.append(info_source)
            
            # Add vector documents
            for document in context.all_vector_documents:
                info_source = create_information_source_from_document(document)
                information_sources.append(info_source)
            
            if not information_sources:
                logger.warning("No information sources available for fact checking")
                context.warnings.append("No information sources available for fact checking")
                return True  # Continue with empty data
            
            # Validate data flow to fact checker
            if not self._validate_data_flow("data_collection", "fact_checker", information_sources, context):
                logger.error("Data validation failed for fact checker input")
                return False
            
            # Execute fact checking with timeout
            future = self.executor.submit(
                self._execute_with_timeout,
                self.fact_checker_agent.check_facts,
                information_sources,
                30  # 30 second timeout
            )
            
            fact_check_result = future.result(timeout=35)
            
            # Store result
            context.fact_check_result = AgentResult(
                agent_name="fact_checker",
                success=fact_check_result is not None,
                data=fact_check_result,
                execution_time=time.time() - start_time,
                metadata={
                    "sources_analyzed": len(information_sources),
                    "verified_facts": len(fact_check_result.verified_facts) if fact_check_result else 0
                }
            )
            
            # Store processed information sources for report generation
            context.information_sources = information_sources
            
            if fact_check_result:
                context.completed_agents.append("fact_checker")
                logger.info("Fact checking completed successfully",
                           sources_analyzed=len(information_sources),
                           verified_facts=len(fact_check_result.verified_facts))
                return True
            else:
                context.failed_agents.append("fact_checker")
                context.errors.append("Fact checking returned no results")
                return False
            
        except Exception as e:
            logger.error("Fact checking stage failed", error=str(e))
            context.errors.append(f"Fact checking failed: {str(e)}")
            context.failed_agents.append("fact_checker")
            return False
    
    def _execute_report_generation_stage(self, context: ResearchContext) -> bool:
        """
        Execute report generation stage.
        
        Args:
            context: Research context
            
        Returns:
            True if successful, False otherwise
        """
        try:
            start_time = time.time()
            
            # Prepare data for report generation
            verified_facts = []
            if context.fact_check_result and context.fact_check_result.success:
                verified_facts = context.fact_check_result.data.verified_facts
            
            # Validate data flow to summarizer
            summarizer_input = (verified_facts, context.source_infos)
            if not self._validate_data_flow("fact_checker", "summarizer", summarizer_input, context):
                logger.warning("Data validation failed for summarizer, proceeding with available data")
                # Continue anyway as we can still generate a basic report
            
            # Create SourceInfo objects for citations
            source_infos = []
            
            # Add from search results
            for search_result in context.all_search_results:
                source_info = create_source_info_from_search_result(search_result)
                source_infos.append(source_info)
            
            # Add from scraped content
            for scraped_content in context.all_scraped_content:
                source_info = create_source_info_from_scraped_content(scraped_content)
                source_infos.append(source_info)
            
            # Add from vector documents
            for document in context.all_vector_documents:
                source_info = create_source_info_from_document(document)
                source_infos.append(source_info)
            
            context.source_infos = source_infos
            
            # Create report configuration
            report_config = ReportConfig(
                style=ReportStyle(context.config.report_style),
                length=ReportLength(context.config.report_length),
                max_citations=min(len(source_infos), 20)
            )
            
            # Execute report generation with timeout
            future = self.executor.submit(
                self._execute_with_timeout,
                self.summarizer_agent.generate_report,
                context.query,
                verified_facts,
                source_infos,
                report_config,
                30  # 30 second timeout
            )
            
            research_report = future.result(timeout=35)
            
            # Store result
            context.summarizer_result = AgentResult(
                agent_name="summarizer",
                success=research_report is not None,
                data=research_report,
                execution_time=time.time() - start_time,
                metadata={
                    "word_count": research_report.metadata.get('word_count', 0) if research_report else 0,
                    "citation_count": len(source_infos)
                }
            )
            
            if research_report:
                context.completed_agents.append("summarizer")
                logger.info("Report generation completed successfully",
                           word_count=research_report.metadata.get('word_count', 0),
                           citations=len(source_infos))
                return True
            else:
                context.failed_agents.append("summarizer")
                context.errors.append("Report generation returned no results")
                return False
            
        except Exception as e:
            logger.error("Report generation stage failed", error=str(e))
            context.errors.append(f"Report generation failed: {str(e)}")
            context.failed_agents.append("summarizer")
            return False
    
    def _execute_with_timeout(self, func, *args, timeout_seconds: int = 30):
        """
        Execute a function with timeout handling.
        
        Args:
            func: Function to execute
            *args: Arguments for the function
            timeout_seconds: Timeout in seconds
            
        Returns:
            Function result or None if timeout/error
        """
        try:
            # Simple timeout implementation - the function should handle its own timeouts
            return func(*args)
        except Exception as e:
            logger.error("Function execution failed", function=func.__name__, error=str(e))
            return None
    
    def _update_progress(self, context: ResearchContext, stage: ResearchStage, message: str):
        """
        Update progress status and notify callback if available.
        
        Args:
            context: Research context
            stage: Current research stage
            message: Status message
        """
        context.current_stage = stage
        context.progress_percentage = self.STAGE_PROGRESS[stage]
        context.status_message = message
        
        if self.progress_callback:
            progress_status = ProgressStatus(
                current_step=stage.value,
                completion_percentage=context.progress_percentage,
                estimated_time_remaining=self._estimate_remaining_time(context),
                completed_agents=context.completed_agents.copy(),
                failed_agents=context.failed_agents.copy(),
                status_message=message
            )
            
            try:
                self.progress_callback(progress_status)
            except Exception as e:
                logger.warning("Progress callback failed", error=str(e))
        
        logger.info("Progress updated", 
                   stage=stage.value, 
                   percentage=context.progress_percentage,
                   message=message)
    
    def _estimate_remaining_time(self, context: ResearchContext) -> int:
        """
        Estimate remaining time based on current progress.
        
        Args:
            context: Research context
            
        Returns:
            Estimated remaining time in seconds
        """
        elapsed_time = time.time() - context.start_time
        
        if context.progress_percentage <= 0:
            return context.config.timeout_seconds
        
        # Simple linear estimation
        total_estimated_time = elapsed_time / (context.progress_percentage / 100)
        remaining_time = total_estimated_time - elapsed_time
        
        return max(0, int(remaining_time))
    
    def _create_success_result(self, context: ResearchContext) -> ResearchResult:
        """
        Create a successful ResearchResult from context.
        
        Args:
            context: Research context
            
        Returns:
            ResearchResult with success status
        """
        execution_time = time.time() - context.start_time
        
        # Get the research report
        research_report = None
        if context.summarizer_result and context.summarizer_result.success:
            research_report = context.summarizer_result.data
        
        # Fallback report if summarizer failed
        if not research_report:
            research_report = self._create_fallback_report(context)
        
        # Count total sources
        total_sources = (
            len(context.all_search_results) + 
            len(context.all_scraped_content) + 
            len(context.all_vector_documents)
        )
        
        # Get comprehensive analytics
        analytics_data = self._aggregate_research_data(context)
        
        # Create metadata with enhanced information
        metadata = {
            "execution_time": execution_time,
            "stages_completed": [stage.value for stage in [
                ResearchStage.PLANNING, ResearchStage.DATA_COLLECTION,
                ResearchStage.FACT_CHECKING, ResearchStage.REPORT_GENERATION
            ] if stage.value in [agent.replace("_", "") for agent in context.completed_agents]],
            "agents_used": context.completed_agents,
            "failed_agents": context.failed_agents,
            "data_sources": {
                "web_search_results": len(context.all_search_results),
                "scraped_pages": len(context.all_scraped_content),
                "vector_documents": len(context.all_vector_documents)
            },
            "warnings": context.warnings,
            "research_strategy": context.router_result.data if context.router_result else None,
            "quality_assessment": analytics_data.get("quality_assessment", {}),
            "performance_metrics": analytics_data.get("performance_metrics", {}),
            "data_validation": {
                "total_sources_validated": total_sources,
                "source_diversity_score": analytics_data.get("quality_assessment", {}).get("source_diversity", 0.0),
                "average_credibility_score": analytics_data.get("quality_assessment", {}).get("credibility_score", 0.0),
                "data_completeness_score": analytics_data.get("quality_assessment", {}).get("data_completeness", 0.0)
            }
        }
        
        result = ResearchResult(
            query=context.query,
            report=research_report,
            metadata=metadata,
            execution_time=execution_time,
            source_count=total_sources,
            success=True
        )
        
        # Save to Google Sheets (non-blocking)
        self._save_to_sheets(result)
        
        return result
    
    def _create_failure_result(self, context: ResearchContext, error_message: str) -> ResearchResult:
        """
        Create a failed ResearchResult from context.
        
        Args:
            context: Research context
            error_message: Primary error message
            
        Returns:
            ResearchResult with failure status
        """
        execution_time = time.time() - context.start_time
        
        # Create a minimal fallback report
        fallback_report = self._create_fallback_report(context)
        
        # Count sources collected before failure
        total_sources = (
            len(context.all_search_results) + 
            len(context.all_scraped_content) + 
            len(context.all_vector_documents)
        )
        
        # Create metadata with error information
        metadata = {
            "execution_time": execution_time,
            "failure_stage": context.current_stage.value,
            "completed_agents": context.completed_agents,
            "failed_agents": context.failed_agents,
            "errors": context.errors,
            "warnings": context.warnings,
            "partial_data_collected": total_sources > 0,
            "data_sources": {
                "web_search_results": len(context.all_search_results),
                "scraped_pages": len(context.all_scraped_content),
                "vector_documents": len(context.all_vector_documents)
            }
        }
        
        result = ResearchResult(
            query=context.query,
            report=fallback_report,
            metadata=metadata,
            execution_time=execution_time,
            source_count=total_sources,
            success=False,
            error_message=error_message
        )
        
        # Save to Google Sheets (non-blocking)
        self._save_to_sheets(result)
        
        return result
    
    def _create_fallback_report(self, context: ResearchContext) -> ResearchReport:
        """
        Create a fallback report when normal report generation fails.
        
        Args:
            context: Research context
            
        Returns:
            Basic ResearchReport with available information
        """
        # Collect any available information
        findings = []
        
        # Add search result snippets
        for result in context.all_search_results[:5]:
            if result.snippet:
                findings.append(f"From {result.title}: {result.snippet}")
        
        # Add scraped content summaries
        for content in context.all_scraped_content[:3]:
            if content.content:
                summary = content.content[:200] + "..." if len(content.content) > 200 else content.content
                findings.append(f"From {content.title}: {summary}")
        
        # Add vector document content
        for doc in context.all_vector_documents[:3]:
            if doc.content:
                summary = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                findings.append(f"From database: {summary}")
        
        # Create basic citations
        citations = []
        all_sources = (
            [(r.title, r.url) for r in context.all_search_results] +
            [(c.title, c.url) for c in context.all_scraped_content] +
            [(d.metadata.get('title', 'Database Document'), d.metadata.get('source_url', 'vector_database')) 
             for d in context.all_vector_documents]
        )
        
        for i, (title, url) in enumerate(all_sources[:10]):
            citations.append(f"[{i+1}] {title}: {url}")
        
        return ResearchReport(
            executive_summary=f"Research was conducted on '{context.query}' but encountered technical difficulties. Some information was collected from {len(all_sources)} sources.",
            key_findings=findings[:5] if findings else ["Limited information available due to technical issues"],
            detailed_analysis=f"The research on '{context.query}' was partially completed. " + 
                            (f"Information was gathered from {len(all_sources)} sources including web search results, scraped content, and database documents. " if all_sources else "") +
                            "However, the full analysis could not be completed due to technical issues.",
            sources=citations,
            recommendations=["Manual review of collected sources recommended", "Consider re-running the research with adjusted parameters"],
            metadata={
                "query": context.query,
                "word_count": 150,
                "source_count": len(all_sources),
                "citation_count": len(citations),
                "confidence_level": "low",
                "research_completeness": "partial",
                "generated_at": datetime.now().isoformat(),
                "fallback_report": True,
                "errors": context.errors
            }
        )
    
    def get_progress(self) -> Optional[ProgressStatus]:
        """
        Get current progress status.
        
        Returns:
            ProgressStatus if research is active, None otherwise
        """
        if not self.current_research:
            return None
        
        context = self.current_research
        
        return ProgressStatus(
            current_step=context.current_stage.value,
            completion_percentage=context.progress_percentage,
            estimated_time_remaining=self._estimate_remaining_time(context),
            completed_agents=context.completed_agents.copy(),
            failed_agents=context.failed_agents.copy(),
            status_message=context.status_message
        )
    
    def cancel_research(self) -> bool:
        """
        Cancel the current research operation.
        
        Returns:
            True if cancellation was successful, False otherwise
        """
        if not self.current_research:
            logger.warning("No active research to cancel")
            return False
        
        self._cancel_requested = True
        logger.info("Research cancellation requested")
        
        # Note: In a more sophisticated implementation, we would need to
        # interrupt running agents. For now, we just set the flag.
        return True
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the orchestrator and all agents.
        
        Returns:
            Dictionary with health status of all components
        """
        health_status = {
            "orchestrator": "healthy",
            "agents": {},
            "executor": {
                "active": not self.executor._shutdown,
                "max_workers": self.executor._max_workers
            },
            "current_research_active": self.current_research is not None
        }
        
        # Check each agent
        try:
            health_status["agents"]["router"] = self.router_agent.health_check()
        except Exception as e:
            health_status["agents"]["router"] = {"error": str(e)}
        
        try:
            health_status["agents"]["web_search"] = self.web_search_agent.health_check()
        except Exception as e:
            health_status["agents"]["web_search"] = {"error": str(e)}
        
        try:
            health_status["agents"]["web_scraper"] = self.web_scraper_agent.health_check()
        except Exception as e:
            health_status["agents"]["web_scraper"] = {"error": str(e)}
        
        try:
            health_status["agents"]["vector_search"] = self.vector_search_agent.health_check()
        except Exception as e:
            health_status["agents"]["vector_search"] = {"error": str(e)}
        
        # Fact checker and summarizer don't have health_check methods in the current implementation
        health_status["agents"]["fact_checker"] = "available"
        health_status["agents"]["summarizer"] = "available"
        
        logger.info("Health check completed", status=health_status)
        return health_status
    
    def cleanup(self):
        """Clean up resources used by the orchestrator"""
        try:
            # Cancel any active research
            if self.current_research:
                self.cancel_research()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Cleanup agents that support it
            if hasattr(self.web_scraper_agent, 'cleanup'):
                self.web_scraper_agent.cleanup()
            
            logger.info("MainOrchestrator cleanup completed")
            
        except Exception as e:
            logger.error("Error during cleanup", error=str(e))
    
    def _validate_agent_results(self, context: ResearchContext) -> Dict[str, Any]:
        """
        Validate and assess quality of results from all agents.
        
        Args:
            context: Research context with agent results
            
        Returns:
            Dictionary with validation results and quality metrics
        """
        validation_results = {
            "overall_quality": "unknown",
            "data_completeness": 0.0,
            "source_diversity": 0.0,
            "credibility_score": 0.0,
            "validation_issues": [],
            "quality_metrics": {}
        }
        
        try:
            # Validate router results
            if context.router_result and context.router_result.success:
                router_data = context.router_result.data
                if not router_data or not hasattr(router_data, 'search_queries'):
                    validation_results["validation_issues"].append("Router agent returned invalid plan structure")
            else:
                validation_results["validation_issues"].append("Router agent failed or returned no results")
            
            # Validate data collection results
            total_sources = len(context.all_search_results) + len(context.all_scraped_content) + len(context.all_vector_documents)
            
            if total_sources == 0:
                validation_results["validation_issues"].append("No data collected from any source")
                validation_results["data_completeness"] = 0.0
            else:
                # Calculate data completeness (0.0 to 1.0)
                max_expected_sources = context.config.max_sources
                validation_results["data_completeness"] = min(1.0, total_sources / max_expected_sources)
            
            # Assess source diversity
            source_types = set()
            if context.all_search_results:
                source_types.add("web_search")
            if context.all_scraped_content:
                source_types.add("web_scraping")
            if context.all_vector_documents:
                source_types.add("vector_database")
            
            validation_results["source_diversity"] = len(source_types) / 3.0  # Max 3 source types
            
            # Calculate average credibility score
            credibility_scores = []
            
            # Add search result credibility scores
            credibility_scores.extend([result.credibility_score for result in context.all_search_results])
            
            # Add vector document credibility scores
            credibility_scores.extend([doc.credibility_score for doc in context.all_vector_documents])
            
            # Add default credibility for scraped content (7.0)
            credibility_scores.extend([7.0] * len(context.all_scraped_content))
            
            if credibility_scores:
                validation_results["credibility_score"] = statistics.mean(credibility_scores) / 10.0  # Normalize to 0-1
            
            # Validate fact checking results
            if context.fact_check_result and context.fact_check_result.success:
                fact_data = context.fact_check_result.data
                if not fact_data or not hasattr(fact_data, 'verified_facts'):
                    validation_results["validation_issues"].append("Fact checker returned invalid results structure")
                elif len(fact_data.verified_facts) == 0:
                    validation_results["validation_issues"].append("No facts were verified by fact checker")
            else:
                validation_results["validation_issues"].append("Fact checking failed")
            
            # Validate report generation results
            if context.summarizer_result and context.summarizer_result.success:
                report_data = context.summarizer_result.data
                if not report_data or not hasattr(report_data, 'executive_summary'):
                    validation_results["validation_issues"].append("Report generation returned invalid structure")
                elif len(report_data.executive_summary) < 50:
                    validation_results["validation_issues"].append("Generated report is too short")
            else:
                validation_results["validation_issues"].append("Report generation failed")
            
            # Calculate overall quality score
            quality_factors = [
                validation_results["data_completeness"],
                validation_results["source_diversity"],
                validation_results["credibility_score"]
            ]
            
            # Penalty for validation issues
            issue_penalty = min(0.5, len(validation_results["validation_issues"]) * 0.1)
            
            overall_score = statistics.mean(quality_factors) - issue_penalty
            overall_score = max(0.0, min(1.0, overall_score))
            
            if overall_score >= 0.8:
                validation_results["overall_quality"] = "excellent"
            elif overall_score >= 0.6:
                validation_results["overall_quality"] = "good"
            elif overall_score >= 0.4:
                validation_results["overall_quality"] = "fair"
            else:
                validation_results["overall_quality"] = "poor"
            
            # Add detailed quality metrics
            validation_results["quality_metrics"] = {
                "overall_score": overall_score,
                "total_sources": total_sources,
                "source_breakdown": {
                    "web_search": len(context.all_search_results),
                    "web_scraping": len(context.all_scraped_content),
                    "vector_database": len(context.all_vector_documents)
                },
                "average_credibility": statistics.mean(credibility_scores) if credibility_scores else 0.0,
                "successful_agents": len(context.completed_agents),
                "failed_agents": len(context.failed_agents)
            }
            
            logger.info("Result validation completed",
                       overall_quality=validation_results["overall_quality"],
                       data_completeness=validation_results["data_completeness"],
                       source_diversity=validation_results["source_diversity"],
                       issues_count=len(validation_results["validation_issues"]))
            
        except Exception as e:
            logger.error("Result validation failed", error=str(e))
            validation_results["validation_issues"].append(f"Validation process error: {str(e)}")
            validation_results["overall_quality"] = "unknown"
        
        return validation_results
    
    def _monitor_performance(self, context: ResearchContext) -> Dict[str, Any]:
        """
        Monitor and analyze performance metrics for the research process.
        
        Args:
            context: Research context with timing and result data
            
        Returns:
            Dictionary with performance metrics and analysis
        """
        performance_metrics = {
            "total_execution_time": 0.0,
            "agent_performance": {},
            "throughput_metrics": {},
            "efficiency_analysis": {},
            "bottlenecks": [],
            "recommendations": []
        }
        
        try:
            # Calculate total execution time
            performance_metrics["total_execution_time"] = time.time() - context.start_time
            
            # Analyze individual agent performance
            agent_results = [
                ("router", context.router_result),
                ("web_search", context.web_search_result),
                ("web_scraper", context.scraper_result),
                ("vector_search", context.vector_search_result),
                ("fact_checker", context.fact_check_result),
                ("summarizer", context.summarizer_result)
            ]
            
            for agent_name, result in agent_results:
                if result:
                    performance_metrics["agent_performance"][agent_name] = {
                        "execution_time": result.execution_time,
                        "success": result.success,
                        "error": result.error,
                        "metadata": result.metadata
                    }
                    
                    # Identify slow agents (> 30 seconds)
                    if result.execution_time > 30:
                        performance_metrics["bottlenecks"].append(f"{agent_name} took {result.execution_time:.1f}s")
                else:
                    performance_metrics["agent_performance"][agent_name] = {
                        "execution_time": 0.0,
                        "success": False,
                        "error": "Agent not executed",
                        "metadata": {}
                    }
            
            # Calculate throughput metrics
            total_sources = len(context.all_search_results) + len(context.all_scraped_content) + len(context.all_vector_documents)
            
            if performance_metrics["total_execution_time"] > 0:
                performance_metrics["throughput_metrics"] = {
                    "sources_per_second": total_sources / performance_metrics["total_execution_time"],
                    "sources_per_minute": total_sources / (performance_metrics["total_execution_time"] / 60),
                    "average_source_processing_time": performance_metrics["total_execution_time"] / max(1, total_sources)
                }
            
            # Efficiency analysis
            successful_agents = len(context.completed_agents)
            total_agents = 6  # Total number of agents in the system
            
            performance_metrics["efficiency_analysis"] = {
                "agent_success_rate": successful_agents / total_agents,
                "data_collection_efficiency": total_sources / max(1, context.config.max_sources),
                "time_efficiency": min(1.0, context.config.timeout_seconds / performance_metrics["total_execution_time"]),
                "parallel_execution_benefit": self._calculate_parallel_benefit(context)
            }
            
            # Generate performance recommendations
            if performance_metrics["total_execution_time"] > context.config.timeout_seconds * 0.8:
                performance_metrics["recommendations"].append("Consider increasing timeout or optimizing slow agents")
            
            if total_sources < context.config.max_sources * 0.5:
                performance_metrics["recommendations"].append("Low source collection rate - check data source availability")
            
            if len(context.failed_agents) > 2:
                performance_metrics["recommendations"].append("Multiple agent failures detected - check system health")
            
            if performance_metrics["efficiency_analysis"]["agent_success_rate"] < 0.7:
                performance_metrics["recommendations"].append("Agent reliability issues - review error logs")
            
            logger.info("Performance monitoring completed",
                       total_time=performance_metrics["total_execution_time"],
                       successful_agents=successful_agents,
                       total_sources=total_sources,
                       bottlenecks=len(performance_metrics["bottlenecks"]))
            
        except Exception as e:
            logger.error("Performance monitoring failed", error=str(e))
            performance_metrics["error"] = str(e)
        
        return performance_metrics
    
    def _calculate_parallel_benefit(self, context: ResearchContext) -> float:
        """
        Calculate the benefit gained from parallel execution.
        
        Args:
            context: Research context
            
        Returns:
            Parallel execution benefit ratio (0.0 to 1.0)
        """
        try:
            # Sum individual agent execution times
            sequential_time = 0.0
            
            if context.router_result:
                sequential_time += context.router_result.execution_time
            if context.web_search_result:
                sequential_time += context.web_search_result.execution_time
            if context.scraper_result:
                sequential_time += context.scraper_result.execution_time
            if context.vector_search_result:
                sequential_time += context.vector_search_result.execution_time
            if context.fact_check_result:
                sequential_time += context.fact_check_result.execution_time
            if context.summarizer_result:
                sequential_time += context.summarizer_result.execution_time
            
            actual_time = time.time() - context.start_time
            
            if sequential_time > 0 and actual_time > 0:
                # Calculate time saved through parallelization
                time_saved = sequential_time - actual_time
                benefit_ratio = time_saved / sequential_time
                return max(0.0, min(1.0, benefit_ratio))
            
            return 0.0
            
        except Exception as e:
            logger.warning("Failed to calculate parallel benefit", error=str(e))
            return 0.0
    
    def _aggregate_research_data(self, context: ResearchContext) -> Dict[str, Any]:
        """
        Aggregate and structure all research data for final processing.
        
        Args:
            context: Research context with all collected data
            
        Returns:
            Dictionary with aggregated and structured research data
        """
        aggregated_data = {
            "query_analysis": None,
            "collected_sources": [],
            "verified_information": [],
            "final_report": None,
            "metadata": {},
            "quality_assessment": {},
            "performance_metrics": {}
        }
        
        try:
            # Aggregate query analysis
            if context.router_result and context.router_result.success:
                aggregated_data["query_analysis"] = {
                    "research_plan": context.router_result.data,
                    "execution_time": context.router_result.execution_time,
                    "strategy_used": context.router_result.data.research_strategy if hasattr(context.router_result.data, 'research_strategy') else None
                }
            
            # Aggregate all collected sources with metadata
            source_id = 1
            
            # Add search results
            for result in context.all_search_results:
                aggregated_data["collected_sources"].append({
                    "id": source_id,
                    "type": "web_search",
                    "title": result.title,
                    "url": result.url,
                    "content": result.snippet,
                    "credibility_score": result.credibility_score,
                    "source_provider": result.source,
                    "collection_method": "search_api"
                })
                source_id += 1
            
            # Add scraped content
            for content in context.all_scraped_content:
                aggregated_data["collected_sources"].append({
                    "id": source_id,
                    "type": "web_scraping",
                    "title": content.title,
                    "url": content.url,
                    "content": content.content[:1000] + "..." if len(content.content) > 1000 else content.content,
                    "author": content.author,
                    "publish_date": content.publish_date.isoformat() if content.publish_date else None,
                    "extraction_method": content.extraction_method,
                    "collection_method": "web_scraping"
                })
                source_id += 1
            
            # Add vector database documents
            for doc in context.all_vector_documents:
                aggregated_data["collected_sources"].append({
                    "id": source_id,
                    "type": "vector_database",
                    "title": doc.metadata.get('title', 'Database Document'),
                    "url": doc.metadata.get('source_url', 'vector_database'),
                    "content": doc.content[:1000] + "..." if len(doc.content) > 1000 else doc.content,
                    "similarity_score": doc.similarity_score,
                    "credibility_score": doc.credibility_score,
                    "metadata": doc.metadata,
                    "collection_method": "vector_search"
                })
                source_id += 1
            
            # Aggregate verified information
            if context.fact_check_result and context.fact_check_result.success:
                fact_data = context.fact_check_result.data
                aggregated_data["verified_information"] = {
                    "verified_facts": fact_data.verified_facts,
                    "credibility_scores": fact_data.credibility_scores,
                    "contradictions": fact_data.contradictions,
                    "processing_time": context.fact_check_result.execution_time
                }
            
            # Add final report
            if context.summarizer_result and context.summarizer_result.success:
                aggregated_data["final_report"] = context.summarizer_result.data
            
            # Add comprehensive metadata
            aggregated_data["metadata"] = {
                "research_query": context.query,
                "research_config": {
                    "max_sources": context.config.max_sources,
                    "enable_web_scraping": context.config.enable_web_scraping,
                    "enable_vector_search": context.config.enable_vector_search,
                    "report_style": context.config.report_style,
                    "report_length": context.config.report_length,
                    "timeout_seconds": context.config.timeout_seconds
                },
                "execution_summary": {
                    "start_time": datetime.fromtimestamp(context.start_time).isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "total_execution_time": time.time() - context.start_time,
                    "completed_stages": [stage.value for stage in [
                        ResearchStage.PLANNING, ResearchStage.DATA_COLLECTION,
                        ResearchStage.FACT_CHECKING, ResearchStage.REPORT_GENERATION
                    ] if stage != context.current_stage or context.current_stage == ResearchStage.COMPLETED],
                    "successful_agents": context.completed_agents,
                    "failed_agents": context.failed_agents,
                    "warnings": context.warnings,
                    "errors": context.errors
                },
                "data_summary": {
                    "total_sources_collected": len(aggregated_data["collected_sources"]),
                    "source_type_breakdown": {
                        "web_search": len(context.all_search_results),
                        "web_scraping": len(context.all_scraped_content),
                        "vector_database": len(context.all_vector_documents)
                    },
                    "verified_facts_count": len(aggregated_data["verified_information"].get("verified_facts", [])),
                    "contradictions_found": len(aggregated_data["verified_information"].get("contradictions", []))
                }
            }
            
            # Add quality assessment
            aggregated_data["quality_assessment"] = self._validate_agent_results(context)
            
            # Add performance metrics
            aggregated_data["performance_metrics"] = self._monitor_performance(context)
            
            logger.info("Research data aggregation completed",
                       total_sources=len(aggregated_data["collected_sources"]),
                       verified_facts=len(aggregated_data["verified_information"].get("verified_facts", [])),
                       overall_quality=aggregated_data["quality_assessment"]["overall_quality"])
            
        except Exception as e:
            logger.error("Data aggregation failed", error=str(e))
            aggregated_data["error"] = str(e)
        
        return aggregated_data
    
    def get_research_analytics(self) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive analytics for the current or last research session.
        
        Returns:
            Dictionary with detailed analytics or None if no research data available
        """
        if not self.current_research:
            logger.warning("No research data available for analytics")
            return None
        
        return self._aggregate_research_data(self.current_research)
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass


# Factory function for easy instantiation
def create_main_orchestrator(
    config: AppConfig,
    progress_callback: Optional[Callable[[ProgressStatus], None]] = None
) -> MainOrchestrator:
    """
    Factory function to create a MainOrchestrator with all required agents.
    
    Args:
        config: Application configuration
        progress_callback: Optional callback for progress updates
        
    Returns:
        Configured MainOrchestrator instance
    """
    from utils.gemini_client import GeminiClient, GeminiConfig
    from utils.chroma_manager import ChromaManager
    
    # Create Gemini client
    gemini_config = GeminiConfig(api_key=config.api.gemini_api_key)
    gemini_client = GeminiClient(gemini_config)
    
    # Create ChromaDB manager
    chroma_manager = ChromaManager(
        db_path=config.database.chroma_db_path,
        collection_name=config.database.collection_name,
        gemini_client=gemini_client
    )
    
    # Create all agents
    from agents.router_agent import RouterAgent
    from agents.web_search_agent import WebSearchAgent
    from agents.web_scraper_agent import WebScraperAgent
    from agents.vector_search_agent import VectorSearchAgent
    from agents.fact_checker_agent import FactCheckerAgent
    from agents.summarizer_agent import SummarizerAgent
    
    router_agent = RouterAgent(gemini_client)
    web_search_agent = WebSearchAgent(config)
    web_scraper_agent = WebScraperAgent(config)
    vector_search_agent = VectorSearchAgent(chroma_manager)
    fact_checker_agent = FactCheckerAgent(gemini_client)
    summarizer_agent = SummarizerAgent(gemini_client)
    
    # Create orchestrator
    orchestrator = MainOrchestrator(
        router_agent=router_agent,
        web_search_agent=web_search_agent,
        web_scraper_agent=web_scraper_agent,
        vector_search_agent=vector_search_agent,
        fact_checker_agent=fact_checker_agent,
        summarizer_agent=summarizer_agent,
        config=config,
        progress_callback=progress_callback
    )
    
    return orchestrator