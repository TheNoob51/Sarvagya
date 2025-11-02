"""
Prompt templates for different agents in the Intelligent Research Assistant.
Provides structured prompts with JSON output specifications and temperature settings.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

class AgentType(Enum):
    """Types of agents with different prompt requirements"""
    ROUTER = "router"
    FACT_CHECKER = "fact_checker"
    SUMMARIZER = "summarizer"

@dataclass
class PromptConfig:
    """Configuration for prompt generation"""
    temperature: float
    max_tokens: int
    system_instruction: Optional[str] = None

class PromptTemplates:
    """
    Collection of prompt templates for different agents.
    Each template includes structured prompts, JSON output specifications, and optimal settings.
    """
    
    # Temperature settings optimized for different agent types
    AGENT_CONFIGS = {
        AgentType.ROUTER: PromptConfig(
            temperature=0.3,  # Low temperature for consistent analysis
            max_tokens=2048,
            system_instruction="You are a research strategy analyst. Analyze queries and create optimal research plans."
        ),
        AgentType.FACT_CHECKER: PromptConfig(
            temperature=0.2,  # Very low temperature for factual accuracy
            max_tokens=4096,
            system_instruction="You are a fact-checking expert. Verify information accuracy and identify contradictions."
        ),
        AgentType.SUMMARIZER: PromptConfig(
            temperature=0.5,  # Moderate temperature for natural writing
            max_tokens=6144,
            system_instruction="You are a professional research writer. Create comprehensive, well-structured reports."
        )
    }
    
    @classmethod
    def get_router_prompt(cls, query: str) -> str:
        """
        Generate prompt for Router Agent to analyze query and create research strategy.
        
        Args:
            query: The research query to analyze
            
        Returns:
            Formatted prompt for router agent
        """
        return f"""Analyze the following research query and create an optimal research strategy.

RESEARCH QUERY: "{query}"

Your task is to determine the best approach for researching this query. Consider:
1. What type of information is needed (factual, analytical, current events, historical, etc.)
2. Which data sources would be most valuable
3. What search queries would yield the best results
4. Which websites might have authoritative information

Respond with a JSON object containing:

{{
    "query_analysis": {{
        "topic_category": "string (e.g., 'technology', 'health', 'politics', 'science', etc.)",
        "information_type": "string - SINGLE VALUE ONLY (choose one: 'factual', 'analytical', 'current_events', 'historical')",
        "complexity_level": "string - SINGLE VALUE ONLY (choose one: 'low', 'medium', 'high')",
        "estimated_time_minutes": "integer (1-10)"
    }},
    "research_strategy": {{
        "use_web_search": "boolean",
        "use_web_scraping": "boolean", 
        "use_vector_search": "boolean",
        "priority_order": ["string array of data source priorities"]
    }},
    "search_queries": [
        "string - optimized search query 1",
        "string - optimized search query 2", 
        "string - optimized search query 3",
        "string - optimized search query 4 (optional)",
        "string - optimized search query 5 (optional)"
    ],
    "target_websites": [
        "string - authoritative website URL 1",
        "string - authoritative website URL 2",
        "string - authoritative website URL 3",
        "string - authoritative website URL 4 (optional)",
        "string - authoritative website URL 5 (optional)"
    ],
    "expected_challenges": [
        "string - potential challenge 1",
        "string - potential challenge 2 (optional)"
    ]
}}

Guidelines:
- Generate 3-5 diverse search queries that approach the topic from different angles
- Suggest authoritative websites likely to have relevant information
- Consider both primary sources and reputable secondary sources
- Factor in recency requirements (use web search for current events, vector search for established knowledge)
- Be realistic about complexity and time estimates"""

    @classmethod
    def get_fact_checker_prompt(cls, information_sources: List[Dict[str, Any]]) -> str:
        """
        Generate prompt for Fact Checker Agent to validate information and identify contradictions.
        
        Args:
            information_sources: List of information sources with content and metadata
            
        Returns:
            Formatted prompt for fact checker agent
        """
        sources_text = ""
        for i, source in enumerate(information_sources, 1):
            source_url = source.get('url', 'Unknown source')
            source_content = source.get('content', '')[:2000]  # Limit content length
            source_type = source.get('type', 'web')
            
            sources_text += f"""
SOURCE {i} ({source_type}):
URL: {source_url}
CONTENT: {source_content}
---
"""
        
        return f"""Analyze the following information sources for accuracy, credibility, and contradictions.

{sources_text}

Your task is to:
1. Identify and remove duplicate or near-duplicate information
2. Detect contradictions between sources
3. Assess the credibility of each source
4. Extract only verified, high-quality facts
5. Flag any questionable or unverifiable claims

Respond with a JSON object containing:

{{
    "analysis_summary": {{
        "total_sources_analyzed": "integer",
        "duplicates_found": "integer", 
        "contradictions_found": "integer",
        "high_credibility_sources": "integer",
        "overall_reliability_score": "float (0.0-1.0)"
    }},
    "source_credibility": [
        {{
            "source_index": "integer (1-based)",
            "credibility_score": "float (1.0-10.0)",
            "credibility_factors": [
                "string - factor 1 (e.g., 'authoritative domain')",
                "string - factor 2 (e.g., 'recent publication')"
            ],
            "concerns": [
                "string - concern 1 (optional)",
                "string - concern 2 (optional)"
            ]
        }}
    ],
    "contradictions": [
        {{
            "topic": "string - what the contradiction is about",
            "source_indices": ["integer array - which sources contradict"],
            "description": "string - description of the contradiction",
            "resolution": "string - how to resolve or which source is more credible"
        }}
    ],
    "verified_facts": [
        {{
            "fact": "string - verified factual statement",
            "supporting_sources": ["integer array - source indices that support this"],
            "confidence_level": "string (high/medium/low)"
        }}
    ],
    "removed_content": [
        {{
            "reason": "string (duplicate/low_credibility/contradiction/unverifiable)",
            "content_preview": "string - first 100 chars of removed content",
            "source_index": "integer"
        }}
    ]
}}

Guidelines:
- Be conservative: when in doubt, flag as questionable rather than accept
- Consider source authority, recency, and corroboration
- Look for bias indicators and promotional content
- Prioritize primary sources over secondary sources
- Cross-reference claims across multiple sources when possible"""

    @classmethod
    def get_summarizer_prompt(
        cls, 
        query: str, 
        verified_facts: List[Dict[str, Any]], 
        sources: List[Dict[str, Any]],
        report_style: str = "academic",
        report_length: str = "medium"
    ) -> str:
        """
        Generate prompt for Summarizer Agent to create research report.
        
        Args:
            query: Original research query
            verified_facts: List of verified facts from fact checker
            sources: List of source information for citations
            report_style: Style of report (academic, casual, technical)
            report_length: Length of report (short, medium, long)
            
        Returns:
            Formatted prompt for summarizer agent
        """
        facts_text = ""
        for i, fact in enumerate(verified_facts, 1):
            fact_content = fact.get('fact', '')
            confidence = fact.get('confidence_level', 'medium')
            supporting_sources = fact.get('supporting_sources', [])
            
            facts_text += f"""
FACT {i} (Confidence: {confidence}):
{fact_content}
Supporting sources: {supporting_sources}
---
"""
        
        sources_text = ""
        for i, source in enumerate(sources, 1):
            source_url = source.get('url', 'Unknown source')
            source_title = source.get('title', 'Untitled')
            source_type = source.get('type', 'web')
            
            sources_text += f"""
[{i}] {source_title} ({source_type})
    URL: {source_url}
"""
        
        # Style-specific instructions
        style_instructions = {
            "academic": "Use formal academic language, include methodology notes, and emphasize evidence-based conclusions.",
            "casual": "Use conversational tone, explain technical terms, and focus on practical implications.",
            "technical": "Use precise technical language, include detailed specifications, and focus on implementation details."
        }
        
        # Length-specific word counts
        length_specs = {
            "short": "400-600 words",
            "medium": "800-1000 words", 
            "long": "1200-1500 words"
        }
        
        return f"""Create a comprehensive research report based on the verified facts and sources provided.

ORIGINAL QUERY: "{query}"

VERIFIED FACTS:
{facts_text}

AVAILABLE SOURCES FOR CITATION:
{sources_text}

REPORT REQUIREMENTS:
- Style: {report_style} ({style_instructions.get(report_style, '')})
- Length: {report_length} ({length_specs.get(report_length, '800-1000 words')})
- Include numbered citations [1], [2], etc.
- Structure with clear sections as specified below

Respond with a JSON object containing:

{{
    "report": {{
        "executive_summary": "string - 3-4 sentence overview of key findings",
        "key_findings": [
            "string - key finding 1 with citation [X]",
            "string - key finding 2 with citation [X]",
            "string - key finding 3 with citation [X]",
            "string - key finding 4 with citation [X] (optional)",
            "string - key finding 5 with citation [X] (optional)"
        ],
        "detailed_analysis": "string - comprehensive analysis with multiple paragraphs and citations",
        "methodology_notes": "string - brief note on research approach and source evaluation",
        "limitations": "string - acknowledgment of any limitations or gaps in available information",
        "recommendations": [
            "string - actionable recommendation 1 (if applicable)",
            "string - actionable recommendation 2 (if applicable)",
            "string - actionable recommendation 3 (if applicable)"
        ]
    }},
    "citations": [
        {{
            "number": "integer - citation number",
            "title": "string - source title", 
            "url": "string - source URL",
            "type": "string - source type (web, academic, news, etc.)",
            "access_date": "string - when information was accessed"
        }}
    ],
    "metadata": {{
        "word_count": "integer - approximate word count",
        "source_count": "integer - number of sources cited",
        "confidence_level": "string (high/medium/low) - overall confidence in findings",
        "research_completeness": "string (comprehensive/adequate/limited)",
        "last_updated": "string - current date"
    }}
}}

Guidelines:
- Ensure all factual claims are supported by citations
- Maintain objectivity and acknowledge uncertainty where appropriate
- Use clear, logical structure with smooth transitions between sections
- Include specific details and examples where available
- Balance comprehensiveness with readability
- Cite sources appropriately throughout the text using [1], [2] format
- Ensure the analysis directly addresses the original research query"""

    @classmethod
    def get_agent_config(cls, agent_type: AgentType) -> PromptConfig:
        """
        Get the optimal configuration for a specific agent type.
        
        Args:
            agent_type: The type of agent
            
        Returns:
            PromptConfig with temperature, max_tokens, and system instruction
        """
        return cls.AGENT_CONFIGS[agent_type]
    
    @classmethod
    def get_embedding_query_prompt(cls, text: str) -> str:
        """
        Generate prompt for creating embedding-optimized queries.
        
        Args:
            text: Original text to create embedding query for
            
        Returns:
            Optimized text for embedding generation
        """
        # For embeddings, we typically want clean, focused text
        # This is a simple implementation - could be enhanced with more sophisticated processing
        return text.strip()
    
    @classmethod
    def validate_json_response(cls, response: str, expected_keys: List[str]) -> bool:
        """
        Validate that a JSON response contains expected keys.
        
        Args:
            response: JSON response string
            expected_keys: List of required top-level keys
            
        Returns:
            True if response is valid JSON with expected keys
        """
        try:
            import json
            parsed = json.loads(response)
            
            if not isinstance(parsed, dict):
                return False
                
            for key in expected_keys:
                if key not in parsed:
                    return False
                    
            return True
            
        except (json.JSONDecodeError, TypeError):
            return False

# Example usage and testing functions
def _test_router_prompt():
    """Test function for router prompt generation"""
    test_query = "What are the latest developments in artificial intelligence safety research?"
    prompt = PromptTemplates.get_router_prompt(test_query)
    config = PromptTemplates.get_agent_config(AgentType.ROUTER)
    
    print("Router Prompt Test:")
    print(f"Temperature: {config.temperature}")
    print(f"Max Tokens: {config.max_tokens}")
    print(f"System Instruction: {config.system_instruction}")
    print(f"Prompt Length: {len(prompt)} characters")
    print("✓ Router prompt generated successfully")

def _test_fact_checker_prompt():
    """Test function for fact checker prompt generation"""
    test_sources = [
        {
            "url": "https://example.com/ai-safety",
            "content": "Recent research shows that AI alignment is a critical challenge...",
            "type": "web"
        },
        {
            "url": "https://arxiv.org/example",
            "content": "Our study demonstrates that current safety measures are insufficient...",
            "type": "academic"
        }
    ]
    
    prompt = PromptTemplates.get_fact_checker_prompt(test_sources)
    config = PromptTemplates.get_agent_config(AgentType.FACT_CHECKER)
    
    print("Fact Checker Prompt Test:")
    print(f"Temperature: {config.temperature}")
    print(f"Max Tokens: {config.max_tokens}")
    print(f"Prompt Length: {len(prompt)} characters")
    print("✓ Fact checker prompt generated successfully")

def _test_summarizer_prompt():
    """Test function for summarizer prompt generation"""
    test_query = "AI safety research developments"
    test_facts = [
        {
            "fact": "AI alignment research has increased 300% in the last two years",
            "confidence_level": "high",
            "supporting_sources": [1, 2]
        }
    ]
    test_sources = [
        {
            "url": "https://example.com/source1",
            "title": "AI Safety Report 2024",
            "type": "web"
        }
    ]
    
    prompt = PromptTemplates.get_summarizer_prompt(
        test_query, test_facts, test_sources, "academic", "medium"
    )
    config = PromptTemplates.get_agent_config(AgentType.SUMMARIZER)
    
    print("Summarizer Prompt Test:")
    print(f"Temperature: {config.temperature}")
    print(f"Max Tokens: {config.max_tokens}")
    print(f"Prompt Length: {len(prompt)} characters")
    print("✓ Summarizer prompt generated successfully")

if __name__ == "__main__":
    print("Testing Prompt Templates...")
    _test_router_prompt()
    print()
    _test_fact_checker_prompt()
    print()
    _test_summarizer_prompt()
    print("\n✓ All prompt template tests passed!")