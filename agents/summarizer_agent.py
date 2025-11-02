"""
Summarizer Agent for the Intelligent Research Assistant.
Generates professional research reports from verified information with structured sections and citations.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import re

import structlog
from utils.gemini_client import GeminiClient, GeminiConfig
from utils.prompt_templates import PromptTemplates, AgentType
from agents.data_models import ResearchReport, ReportStyle, ReportLength

logger = structlog.get_logger(__name__)

@dataclass
class ReportConfig:
    """Configuration for report generation"""
    style: ReportStyle = ReportStyle.ACADEMIC
    length: ReportLength = ReportLength.MEDIUM
    include_methodology: bool = True
    include_limitations: bool = True
    include_recommendations: bool = True
    max_citations: int = 20
    min_word_count: int = 400
    max_word_count: int = 1500

@dataclass
class SourceInfo:
    """Information about a source for citation"""
    title: str
    url: str
    source_type: str  # "web", "academic", "news", "government", etc.
    author: Optional[str] = None
    publish_date: Optional[datetime] = None
    credibility_score: float = 5.0

class SummarizerAgent:
    """
    Agent responsible for generating professional research reports from verified information.
    
    Key responsibilities:
    - Generate structured reports with all required sections
    - Format citations properly with numbered references
    - Manage word count according to specified length
    - Apply appropriate writing style (academic, casual, technical)
    - Validate report quality and completeness
    """
    
    # Word count targets for different report lengths
    WORD_COUNT_TARGETS = {
        ReportLength.SHORT: {"min": 400, "target": 500, "max": 600},
        ReportLength.MEDIUM: {"min": 800, "target": 900, "max": 1000},
        ReportLength.LONG: {"min": 1200, "target": 1350, "max": 1500}
    }
    
    # Style-specific writing guidelines
    STYLE_GUIDELINES = {
        ReportStyle.ACADEMIC: {
            "tone": "formal and objective",
            "language": "precise academic terminology",
            "structure": "rigorous with clear methodology",
            "citations": "frequent and detailed",
            "recommendations": "evidence-based conclusions"
        },
        ReportStyle.CASUAL: {
            "tone": "conversational and accessible",
            "language": "plain language with explanations",
            "structure": "clear and engaging",
            "citations": "integrated naturally",
            "recommendations": "practical and actionable"
        },
        ReportStyle.TECHNICAL: {
            "tone": "precise and detailed",
            "language": "technical terminology and specifications",
            "structure": "systematic with implementation focus",
            "citations": "technical sources prioritized",
            "recommendations": "implementation-focused"
        }
    }
    
    def __init__(self, gemini_client: GeminiClient):
        """
        Initialize the Summarizer Agent.
        
        Args:
            gemini_client: Configured Gemini API client
        """
        self.gemini_client = gemini_client
        self.prompt_config = PromptTemplates.get_agent_config(AgentType.SUMMARIZER)
        
        logger.info("SummarizerAgent initialized")
    
    def generate_report(
        self, 
        query: str,
        verified_facts: List[Dict[str, Any]], 
        sources: List[SourceInfo],
        config: ReportConfig = None
    ) -> ResearchReport:
        """
        Generate a comprehensive research report from verified facts and sources.
        
        Args:
            query: Original research query
            verified_facts: List of verified facts from fact checker
            sources: List of source information for citations
            config: Report configuration (optional)
            
        Returns:
            ResearchReport with all sections and metadata
        """
        config = config or ReportConfig()
        
        logger.info("Starting report generation", 
                   query=query,
                   facts_count=len(verified_facts),
                   sources_count=len(sources),
                   style=config.style.value,
                   length=config.length.value)
        
        try:
            # Step 1: Prepare and validate input data
            processed_facts = self._process_verified_facts(verified_facts)
            processed_sources = self._process_sources(sources, config.max_citations)
            
            # Step 2: Generate report using Gemini
            report_data = self._generate_with_gemini(
                query, processed_facts, processed_sources, config
            )
            
            # Step 3: Validate and enhance the report
            validated_report = self._validate_and_enhance_report(report_data, config)
            
            # Step 4: Create final ResearchReport object
            research_report = self._create_research_report(
                validated_report, query, len(sources)
            )
            
            logger.info("Report generation completed successfully",
                       word_count=research_report.metadata.get('word_count', 0),
                       citations_count=research_report.metadata.get('source_count', 0))
            
            return research_report
            
        except Exception as e:
            logger.error("Report generation failed", error=str(e))
            # Return minimal report on failure
            return self._create_fallback_report(query, verified_facts, sources)
    
    def _process_verified_facts(self, verified_facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process and organize verified facts for report generation.
        
        Args:
            verified_facts: Raw verified facts from fact checker
            
        Returns:
            Processed facts with enhanced metadata
        """
        processed = []
        
        for fact in verified_facts:
            if isinstance(fact, dict) and 'fact' in fact:
                processed_fact = {
                    'fact': fact['fact'],
                    'confidence_level': fact.get('confidence_level', 'medium'),
                    'supporting_sources': fact.get('supporting_sources', []),
                    'category': self._categorize_fact(fact['fact'])
                }
                processed.append(processed_fact)
        
        # Sort by confidence level and category
        processed.sort(key=lambda x: (
            {'high': 3, 'medium': 2, 'low': 1}[x['confidence_level']],
            x['category']
        ), reverse=True)
        
        return processed
    
    def _categorize_fact(self, fact_text: str) -> str:
        """
        Categorize a fact based on its content.
        
        Args:
            fact_text: The factual statement
            
        Returns:
            Category string
        """
        fact_lower = fact_text.lower()
        
        # Simple keyword-based categorization
        if any(word in fact_lower for word in ['statistic', 'percent', 'number', 'rate', 'data']):
            return 'statistical'
        elif any(word in fact_lower for word in ['research', 'study', 'found', 'discovered']):
            return 'research_finding'
        elif any(word in fact_lower for word in ['definition', 'means', 'refers to', 'defined as']):
            return 'definition'
        elif any(word in fact_lower for word in ['trend', 'increase', 'decrease', 'growth']):
            return 'trend'
        else:
            return 'general'
    
    def _process_sources(self, sources: List[SourceInfo], max_citations: int) -> List[Dict[str, Any]]:
        """
        Process and prioritize sources for citation.
        
        Args:
            sources: List of source information
            max_citations: Maximum number of citations to include
            
        Returns:
            Processed sources sorted by priority
        """
        # Sort sources by credibility score and type priority
        type_priority = {
            'academic': 5,
            'government': 4,
            'news': 3,
            'web': 2,
            'vector_db': 2,
            'social': 1
        }
        
        processed_sources = []
        for i, source in enumerate(sources):
            processed_source = {
                'index': i + 1,
                'title': source.title,
                'url': source.url,
                'type': source.source_type,
                'author': source.author,
                'publish_date': source.publish_date,
                'credibility_score': source.credibility_score,
                'priority_score': (
                    source.credibility_score * 0.7 + 
                    type_priority.get(source.source_type, 1) * 0.3
                )
            }
            processed_sources.append(processed_source)
        
        # Sort by priority and limit to max_citations
        processed_sources.sort(key=lambda x: x['priority_score'], reverse=True)
        return processed_sources[:max_citations]
    
    def _generate_with_gemini(
        self, 
        query: str, 
        facts: List[Dict[str, Any]], 
        sources: List[Dict[str, Any]], 
        config: ReportConfig
    ) -> Dict[str, Any]:
        """
        Generate report content using Gemini API.
        
        Args:
            query: Research query
            facts: Processed verified facts
            sources: Processed sources
            config: Report configuration
            
        Returns:
            Generated report data
        """
        try:
            # Generate the prompt
            prompt = PromptTemplates.get_summarizer_prompt(
                query=query,
                verified_facts=facts,
                sources=sources,
                report_style=config.style.value,
                report_length=config.length.value
            )
            
            # Call Gemini API
            response = self.gemini_client.generate_json(
                prompt=prompt,
                temperature=self.prompt_config.temperature,
                system_instruction=self.prompt_config.system_instruction
            )
            
            logger.info("Gemini report generation completed successfully")
            return response
            
        except Exception as e:
            logger.error("Gemini report generation failed", error=str(e))
            # Return minimal structure on failure
            return self._create_minimal_report_structure(query, facts, sources)
    
    def _create_minimal_report_structure(
        self, 
        query: str, 
        facts: List[Dict[str, Any]], 
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a minimal report structure when Gemini API fails.
        
        Args:
            query: Research query
            facts: Verified facts
            sources: Source information
            
        Returns:
            Minimal report structure
        """
        # Extract key facts for summary
        key_facts = [fact['fact'] for fact in facts[:5]]
        
        return {
            "report": {
                "executive_summary": f"Research on '{query}' yielded {len(facts)} verified facts from {len(sources)} sources. Key findings include information about the requested topic.",
                "key_findings": key_facts,
                "detailed_analysis": f"Based on the available sources, the research on '{query}' reveals several important aspects. " + " ".join(key_facts[:3]),
                "methodology_notes": f"This research analyzed {len(sources)} sources using automated fact-checking and credibility assessment.",
                "limitations": "This report was generated using automated analysis and may require human review for completeness.",
                "recommendations": ["Further research may be needed for comprehensive understanding."]
            },
            "citations": [
                {
                    "number": i + 1,
                    "title": source['title'],
                    "url": source['url'],
                    "type": source['type'],
                    "access_date": datetime.now().strftime("%Y-%m-%d")
                }
                for i, source in enumerate(sources[:10])
            ],
            "metadata": {
                "word_count": 200,
                "source_count": len(sources),
                "confidence_level": "medium",
                "research_completeness": "limited",
                "last_updated": datetime.now().strftime("%Y-%m-%d")
            }
        }
    
    def _validate_and_enhance_report(
        self, 
        report_data: Dict[str, Any], 
        config: ReportConfig
    ) -> Dict[str, Any]:
        """
        Validate and enhance the generated report.
        
        Args:
            report_data: Raw report data from Gemini
            config: Report configuration
            
        Returns:
            Validated and enhanced report data
        """
        report = report_data.get('report', {})
        
        # Validate required sections
        required_sections = ['executive_summary', 'key_findings', 'detailed_analysis']
        for section in required_sections:
            if not report.get(section):
                logger.warning(f"Missing required section: {section}")
                report[section] = f"[{section.replace('_', ' ').title()} not available]"
        
        # Validate word count
        word_count = self._count_words(report)
        target_counts = self.WORD_COUNT_TARGETS[config.length]
        
        if word_count < target_counts['min']:
            logger.warning("Report below minimum word count", 
                          actual=word_count, 
                          minimum=target_counts['min'])
        elif word_count > target_counts['max']:
            logger.warning("Report exceeds maximum word count", 
                          actual=word_count, 
                          maximum=target_counts['max'])
        
        # Enhance metadata
        metadata = report_data.get('metadata', {})
        metadata.update({
            'actual_word_count': word_count,
            'target_word_count': target_counts['target'],
            'style_applied': config.style.value,
            'length_category': config.length.value,
            'generation_timestamp': datetime.now().isoformat()
        })
        
        report_data['metadata'] = metadata
        
        # Validate citations
        citations = report_data.get('citations', [])
        if not citations:
            logger.warning("No citations found in report")
        
        return report_data
    
    def _count_words(self, report: Dict[str, Any]) -> int:
        """
        Count words in the main report sections.
        
        Args:
            report: Report dictionary
            
        Returns:
            Total word count
        """
        text_sections = [
            report.get('executive_summary', ''),
            ' '.join(report.get('key_findings', [])),
            report.get('detailed_analysis', ''),
            report.get('methodology_notes', ''),
            report.get('limitations', ''),
            ' '.join(report.get('recommendations', []))
        ]
        
        total_text = ' '.join(text_sections)
        words = re.findall(r'\b\w+\b', total_text)
        return len(words)
    
    def _create_research_report(
        self, 
        report_data: Dict[str, Any], 
        query: str, 
        source_count: int
    ) -> ResearchReport:
        """
        Create final ResearchReport object from validated data.
        
        Args:
            report_data: Validated report data
            query: Original research query
            source_count: Number of sources used
            
        Returns:
            ResearchReport object
        """
        report = report_data.get('report', {})
        citations = report_data.get('citations', [])
        metadata = report_data.get('metadata', {})
        
        # Format citations as strings
        citation_strings = []
        for citation in citations:
            citation_str = f"[{citation.get('number', '?')}] {citation.get('title', 'Unknown Title')}"
            if citation.get('author'):
                citation_str += f" by {citation['author']}"
            citation_str += f" ({citation.get('type', 'web')}): {citation.get('url', '')}"
            if citation.get('access_date'):
                citation_str += f" (accessed {citation['access_date']})"
            citation_strings.append(citation_str)
        
        # Create metadata dictionary
        final_metadata = {
            'query': query,
            'word_count': metadata.get('actual_word_count', 0),
            'source_count': source_count,
            'citation_count': len(citations),
            'confidence_level': metadata.get('confidence_level', 'medium'),
            'research_completeness': metadata.get('research_completeness', 'adequate'),
            'style': metadata.get('style_applied', 'academic'),
            'length': metadata.get('length_category', 'medium'),
            'generated_at': metadata.get('generation_timestamp', datetime.now().isoformat()),
            'last_updated': metadata.get('last_updated', datetime.now().strftime("%Y-%m-%d"))
        }
        
        return ResearchReport(
            executive_summary=report.get('executive_summary', ''),
            key_findings=report.get('key_findings', []),
            detailed_analysis=report.get('detailed_analysis', ''),
            sources=citation_strings,
            recommendations=report.get('recommendations', []),
            metadata=final_metadata
        )
    
    def _create_fallback_report(
        self, 
        query: str, 
        verified_facts: List[Dict[str, Any]], 
        sources: List[SourceInfo]
    ) -> ResearchReport:
        """
        Create a fallback report when generation fails.
        
        Args:
            query: Research query
            verified_facts: Verified facts
            sources: Source information
            
        Returns:
            Basic ResearchReport
        """
        logger.warning("Creating fallback report due to generation failure")
        
        # Extract basic information
        fact_texts = [fact.get('fact', str(fact)) for fact in verified_facts[:5]]
        
        return ResearchReport(
            executive_summary=f"Research was conducted on '{query}' but report generation encountered issues. {len(verified_facts)} facts were verified from {len(sources)} sources.",
            key_findings=fact_texts,
            detailed_analysis=f"The research on '{query}' yielded information from multiple sources. However, detailed analysis could not be completed due to technical issues.",
            sources=[f"[{i+1}] {source.title}: {source.url}" for i, source in enumerate(sources[:10])],
            recommendations=["Manual review of sources recommended", "Additional research may be needed"],
            metadata={
                'query': query,
                'word_count': 100,
                'source_count': len(sources),
                'citation_count': min(len(sources), 10),
                'confidence_level': 'low',
                'research_completeness': 'limited',
                'generated_at': datetime.now().isoformat(),
                'error': 'Report generation failed'
            }
        )
    
    def create_executive_summary(self, content: str, max_sentences: int = 4) -> str:
        """
        Create an executive summary from content (standalone method).
        
        Args:
            content: Full content to summarize
            max_sentences: Maximum number of sentences
            
        Returns:
            Executive summary string
        """
        # Simple extractive summarization - take first few sentences
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        summary_sentences = sentences[:max_sentences]
        return '. '.join(summary_sentences) + '.'
    
    def generate_citations(self, sources: List[SourceInfo]) -> List[str]:
        """
        Generate formatted citations from sources (standalone method).
        
        Args:
            sources: List of source information
            
        Returns:
            List of formatted citation strings
        """
        citations = []
        for i, source in enumerate(sources):
            citation = f"[{i+1}] {source.title}"
            if source.author:
                citation += f" by {source.author}"
            citation += f" ({source.source_type}): {source.url}"
            if source.publish_date:
                citation += f" ({source.publish_date.strftime('%Y-%m-%d')})"
            citations.append(citation)
        
        return citations

# Utility functions for creating SourceInfo objects

def create_source_info_from_search_result(search_result) -> SourceInfo:
    """Create SourceInfo from SearchResult data model"""
    return SourceInfo(
        title=search_result.title,
        url=search_result.url,
        source_type="web",
        credibility_score=search_result.credibility_score
    )

def create_source_info_from_scraped_content(scraped_content) -> SourceInfo:
    """Create SourceInfo from ScrapedContent data model"""
    return SourceInfo(
        title=scraped_content.title,
        url=scraped_content.url,
        source_type="web",
        author=scraped_content.author,
        publish_date=scraped_content.publish_date,
        credibility_score=7.0  # Default for scraped content
    )

def create_source_info_from_document(document) -> SourceInfo:
    """Create SourceInfo from Document (vector database) data model"""
    metadata = document.metadata
    return SourceInfo(
        title=metadata.get('title', 'Vector Database Document'),
        url=metadata.get('source_url', 'vector_database'),
        source_type="vector_db",
        author=metadata.get('author'),
        publish_date=metadata.get('timestamp'),
        credibility_score=document.credibility_score
    )