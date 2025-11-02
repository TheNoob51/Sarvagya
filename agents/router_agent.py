"""
Router Agent for query analysis and research strategy planning.
Analyzes research queries using Gemini API and generates optimal research plans.
"""

import json
import logging
from typing import Dict, Any, List
import structlog

from utils.gemini_client import GeminiClient, GeminiConfig
from utils.prompt_templates import PromptTemplates, AgentType
from agents.data_models import (
    ResearchPlan, QueryAnalysis, ResearchStrategy, 
    ComplexityLevel, InformationType
)

logger = structlog.get_logger(__name__)

class RouterAgent:
    """
    Router Agent that analyzes research queries and creates optimal research strategies.
    
    This agent uses Gemini API to:
    1. Analyze the research query to understand topic, complexity, and information type
    2. Determine which data sources should be used (web search, scraping, vector DB)
    3. Generate optimized search queries for web search agents
    4. Suggest authoritative websites for targeted scraping
    5. Return a structured ResearchPlan with all strategy decisions
    """
    
    def __init__(self, gemini_client: GeminiClient):
        """
        Initialize Router Agent with Gemini client.
        
        Args:
            gemini_client: Configured GeminiClient instance
        """
        self.gemini_client = gemini_client
        self.agent_config = PromptTemplates.get_agent_config(AgentType.ROUTER)
        
        logger.info("RouterAgent initialized", 
                   temperature=self.agent_config.temperature,
                   max_tokens=self.agent_config.max_tokens)
    
    def analyze_query(self, query: str) -> ResearchPlan:
        """
        Analyze research query and create comprehensive research plan.
        
        Args:
            query: The research query to analyze
            
        Returns:
            ResearchPlan with complete strategy and recommendations
            
        Raises:
            Exception: If query analysis fails or returns invalid data
        """
        logger.info("Starting query analysis", query=query[:100])
        
        try:
            # Generate the router prompt
            prompt = PromptTemplates.get_router_prompt(query)
            
            # Get response from Gemini API
            response = self.gemini_client.generate_json(
                prompt=prompt,
                temperature=self.agent_config.temperature,
                system_instruction=self.agent_config.system_instruction
            )
            
            # Validate response structure
            self._validate_response(response)
            
            # Parse response into ResearchPlan
            research_plan = self._parse_response(response)
            
            logger.info("Query analysis completed successfully",
                       complexity=research_plan.query_analysis.complexity_level.value,
                       search_queries_count=len(research_plan.search_queries),
                       target_websites_count=len(research_plan.target_websites))
            
            return research_plan
            
        except Exception as e:
            logger.error("Query analysis failed", query=query[:100], error=str(e))
            raise Exception(f"Router Agent failed to analyze query: {str(e)}")
    
    def generate_search_queries(self, query: str) -> List[str]:
        """
        Generate optimized search queries for the given research query.
        
        Args:
            query: The original research query
            
        Returns:
            List of 3-5 optimized search queries
        """
        research_plan = self.analyze_query(query)
        return research_plan.search_queries
    
    def suggest_websites(self, query: str) -> List[str]:
        """
        Suggest authoritative websites for targeted scraping.
        
        Args:
            query: The original research query
            
        Returns:
            List of 3-5 suggested website URLs
        """
        research_plan = self.analyze_query(query)
        return research_plan.target_websites
    
    def _validate_response(self, response: Dict[str, Any]) -> None:
        """
        Validate that the Gemini API response has the expected structure.
        
        Args:
            response: The parsed JSON response from Gemini API
            
        Raises:
            ValueError: If response structure is invalid
        """
        required_keys = [
            "query_analysis", "research_strategy", 
            "search_queries", "target_websites", "expected_challenges"
        ]
        
        for key in required_keys:
            if key not in response:
                raise ValueError(f"Missing required key in response: {key}")
        
        # Validate query_analysis structure
        query_analysis = response["query_analysis"]
        required_analysis_keys = [
            "topic_category", "information_type", 
            "complexity_level", "estimated_time_minutes"
        ]
        
        for key in required_analysis_keys:
            if key not in query_analysis:
                raise ValueError(f"Missing required key in query_analysis: {key}")
        
        # Validate research_strategy structure
        research_strategy = response["research_strategy"]
        required_strategy_keys = [
            "use_web_search", "use_web_scraping", 
            "use_vector_search", "priority_order"
        ]
        
        for key in required_strategy_keys:
            if key not in research_strategy:
                raise ValueError(f"Missing required key in research_strategy: {key}")
        
        # Validate data types
        if not isinstance(response["search_queries"], list):
            raise ValueError("search_queries must be a list")
        
        if not isinstance(response["target_websites"], list):
            raise ValueError("target_websites must be a list")
        
        if not isinstance(response["expected_challenges"], list):
            raise ValueError("expected_challenges must be a list")
        
        # Validate search queries count (3-5 required)
        if len(response["search_queries"]) < 3 or len(response["search_queries"]) > 5:
            raise ValueError("Must provide 3-5 search queries")
        
        # Validate target websites count (3-5 required)
        if len(response["target_websites"]) < 3 or len(response["target_websites"]) > 5:
            raise ValueError("Must provide 3-5 target websites")
        
        logger.debug("Response validation passed")
    
    def _parse_response(self, response: Dict[str, Any]) -> ResearchPlan:
        """
        Parse the validated Gemini API response into a ResearchPlan object.
        
        Args:
            response: The validated JSON response from Gemini API
            
        Returns:
            ResearchPlan object with parsed data
            
        Raises:
            ValueError: If data cannot be parsed into expected types
        """
        try:
            # Parse query analysis
            query_analysis_data = response["query_analysis"]
            
            # Convert string enums to enum types
            try:
                # Handle multiple information types - take the first valid one
                info_type_str = query_analysis_data["information_type"]
                if "," in info_type_str:
                    # Split and try each type
                    for type_part in info_type_str.split(","):
                        type_part = type_part.strip()
                        try:
                            information_type = InformationType(type_part)
                            break
                        except ValueError:
                            continue
                    else:
                        # If none matched, use fallback
                        information_type = InformationType.FACTUAL
                        logger.warning("No valid information_type found, using FACTUAL as fallback",
                                     provided_type=info_type_str)
                else:
                    information_type = InformationType(info_type_str)
            except ValueError:
                # Fallback to FACTUAL if invalid type provided
                information_type = InformationType.FACTUAL
                logger.warning("Invalid information_type, using FACTUAL as fallback",
                             provided_type=query_analysis_data["information_type"])
            
            try:
                complexity_level = ComplexityLevel(query_analysis_data["complexity_level"])
            except ValueError:
                # Fallback to MEDIUM if invalid complexity provided
                complexity_level = ComplexityLevel.MEDIUM
                logger.warning("Invalid complexity_level, using MEDIUM as fallback",
                             provided_complexity=query_analysis_data["complexity_level"])
            
            query_analysis = QueryAnalysis(
                topic_category=query_analysis_data["topic_category"],
                information_type=information_type,
                complexity_level=complexity_level,
                estimated_time_minutes=int(query_analysis_data["estimated_time_minutes"])
            )
            
            # Parse research strategy
            strategy_data = response["research_strategy"]
            research_strategy = ResearchStrategy(
                use_web_search=bool(strategy_data["use_web_search"]),
                use_web_scraping=bool(strategy_data["use_web_scraping"]),
                use_vector_search=bool(strategy_data["use_vector_search"]),
                priority_order=list(strategy_data["priority_order"])
            )
            
            # Create ResearchPlan
            research_plan = ResearchPlan(
                query_analysis=query_analysis,
                research_strategy=research_strategy,
                search_queries=list(response["search_queries"]),
                target_websites=list(response["target_websites"]),
                expected_challenges=list(response["expected_challenges"])
            )
            
            logger.debug("Response parsed successfully into ResearchPlan")
            return research_plan
            
        except (KeyError, ValueError, TypeError) as e:
            logger.error("Failed to parse response into ResearchPlan", error=str(e))
            raise ValueError(f"Failed to parse response: {str(e)}")
    
    def health_check(self) -> bool:
        """
        Perform a health check on the Router Agent.
        
        Returns:
            True if agent is functioning properly, False otherwise
        """
        try:
            # Test with a simple query
            test_query = "What is artificial intelligence?"
            test_plan = self.analyze_query(test_query)
            
            # Verify we got a valid plan
            is_healthy = (
                test_plan is not None and
                len(test_plan.search_queries) >= 3 and
                len(test_plan.target_websites) >= 3 and
                test_plan.query_analysis is not None and
                test_plan.research_strategy is not None
            )
            
            logger.info("Router Agent health check completed", is_healthy=is_healthy)
            return is_healthy
            
        except Exception as e:
            logger.error("Router Agent health check failed", error=str(e))
            return False


# Factory function for easy instantiation
def create_router_agent(gemini_config: GeminiConfig) -> RouterAgent:
    """
    Factory function to create a RouterAgent with proper configuration.
    
    Args:
        gemini_config: Configuration for Gemini API client
        
    Returns:
        Configured RouterAgent instance
    """
    gemini_client = GeminiClient(gemini_config)
    return RouterAgent(gemini_client)