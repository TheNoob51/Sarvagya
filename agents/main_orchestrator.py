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

import structlog
from agents.data_models import (
    ResearchConfig, ResearchResult, ProgressStatus, ResearchReport,
    SearchResult, ScrapedContent, Document, FactCheckResult
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
        
        # Execution control
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.current_research: Optional[ResearchContext] = None
        self._cancel_requested = False
        
        logger.info("MainOrchestrator initialized", 
                   max_workers=4,
                   has_progress_callback=progress_callback is not None)
    
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
            
            # Stage 5: Completion
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
            
            # Create SourceInfo objects for citations
            source_infos = []
            
            # Add from search results
            for search_result in context.all_search_results:
                source_info = create_source_info_from_search_result(search_result)
                source_infos.append(source_info)
            
            # Add from scraped content
            for scraped_con