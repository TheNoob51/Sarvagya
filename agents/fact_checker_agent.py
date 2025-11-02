"""
Fact Checker Agent for the Intelligent Research Assistant.
Validates information accuracy, identifies contradictions, scores credibility, and removes duplicates.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib
import re
from urllib.parse import urlparse

import structlog
from utils.gemini_client import GeminiClient, GeminiConfig
from utils.prompt_templates import PromptTemplates, AgentType
from agents.data_models import FactCheckResult

logger = structlog.get_logger(__name__)

@dataclass
class InformationSource:
    """Represents a source of information for fact checking"""
    content: str
    url: str
    title: str
    source_type: str  # "web", "academic", "news", "vector_db"
    author: Optional[str] = None
    publish_date: Optional[datetime] = None
    extraction_method: Optional[str] = None

@dataclass
class CredibilityFactors:
    """Factors that influence source credibility"""
    domain_authority: float  # 0.0-1.0
    content_quality: float   # 0.0-1.0
    recency: float          # 0.0-1.0
    author_credibility: float  # 0.0-1.0
    source_type_weight: float  # 0.0-1.0

class FactCheckerAgent:
    """
    Agent responsible for validating information accuracy and identifying contradictions.
    
    Key responsibilities:
    - Remove duplicate and near-duplicate content
    - Detect contradictions between sources
    - Assess credibility of each source
    - Extract verified facts only
    - Filter sources based on credibility thresholds
    """
    
    # Domain authority scores for common domains
    DOMAIN_AUTHORITY = {
        # Academic and research institutions
        'edu': 0.9, 'ac.uk': 0.9, 'arxiv.org': 0.85, 'pubmed.ncbi.nlm.nih.gov': 0.9,
        'nature.com': 0.9, 'science.org': 0.9, 'ieee.org': 0.85,
        
        # Government sources
        'gov': 0.85, 'gov.uk': 0.85, 'europa.eu': 0.8, 'who.int': 0.85,
        
        # Reputable news organizations
        'reuters.com': 0.8, 'bbc.com': 0.8, 'npr.org': 0.8, 'apnews.com': 0.8,
        'nytimes.com': 0.75, 'washingtonpost.com': 0.75, 'theguardian.com': 0.75,
        
        # Technology sources
        'stackoverflow.com': 0.7, 'github.com': 0.7, 'medium.com': 0.6,
        
        # Wikipedia and wikis
        'wikipedia.org': 0.65, 'wikimedia.org': 0.65,
        
        # Default for unknown domains
        'default': 0.5
    }
    
    # Source type weights
    SOURCE_TYPE_WEIGHTS = {
        'academic': 0.9,
        'government': 0.85,
        'news': 0.7,
        'web': 0.6,
        'vector_db': 0.75,  # Previously verified content
        'social': 0.3,
        'blog': 0.4,
        'forum': 0.3
    }
    
    def __init__(self, gemini_client: GeminiClient, credibility_threshold: float = 6.0):
        """
        Initialize the Fact Checker Agent.
        
        Args:
            gemini_client: Configured Gemini API client
            credibility_threshold: Minimum credibility score to keep sources (1.0-10.0)
        """
        self.gemini_client = gemini_client
        self.credibility_threshold = credibility_threshold
        self.prompt_config = PromptTemplates.get_agent_config(AgentType.FACT_CHECKER)
        
        logger.info("FactCheckerAgent initialized", 
                   credibility_threshold=credibility_threshold)
    
    def check_facts(self, information_sources: List[InformationSource]) -> FactCheckResult:
        """
        Main method to fact-check a list of information sources.
        
        Args:
            information_sources: List of information sources to validate
            
        Returns:
            FactCheckResult with verified facts, credibility scores, and contradictions
        """
        logger.info("Starting fact checking process", source_count=len(information_sources))
        
        try:
            # Step 1: Remove duplicates
            unique_sources = self._remove_duplicates(information_sources)
            logger.info("Duplicate removal completed", 
                       original_count=len(information_sources),
                       unique_count=len(unique_sources))
            
            # Step 2: Calculate credibility scores
            credibility_scores = self._calculate_credibility_scores(unique_sources)
            
            # Step 3: Filter by credibility threshold
            high_credibility_sources = [
                source for i, source in enumerate(unique_sources)
                if credibility_scores[i]['credibility_score'] >= self.credibility_threshold
            ]
            
            logger.info("Credibility filtering completed",
                       high_credibility_count=len(high_credibility_sources),
                       threshold=self.credibility_threshold)
            
            # Step 4: Use Gemini to analyze for contradictions and extract facts
            if high_credibility_sources:
                gemini_analysis = self._analyze_with_gemini(high_credibility_sources)
            else:
                # If no high-credibility sources, analyze all unique sources but flag low confidence
                logger.warning("No sources meet credibility threshold, analyzing all sources")
                gemini_analysis = self._analyze_with_gemini(unique_sources)
            
            # Step 5: Compile final result
            result = FactCheckResult(
                verified_facts=gemini_analysis.get('verified_facts', []),
                credibility_scores={
                    f"source_{i+1}": score['credibility_score'] 
                    for i, score in enumerate(credibility_scores)
                },
                contradictions=gemini_analysis.get('contradictions', []),
                cleaned_data=[fact['fact'] for fact in gemini_analysis.get('verified_facts', [])]
            )
            
            logger.info("Fact checking completed successfully",
                       verified_facts_count=len(result.verified_facts),
                       contradictions_count=len(result.contradictions))
            
            return result
            
        except Exception as e:
            logger.error("Fact checking failed", error=str(e))
            # Return empty result on failure
            return FactCheckResult(
                verified_facts=[],
                credibility_scores={},
                contradictions=[],
                cleaned_data=[]
            )
    
    def _remove_duplicates(self, sources: List[InformationSource]) -> List[InformationSource]:
        """
        Remove duplicate and near-duplicate content from sources.
        
        Args:
            sources: List of information sources
            
        Returns:
            List of unique sources
        """
        unique_sources = []
        seen_hashes = set()
        seen_urls = set()
        
        for source in sources:
            # Skip if we've seen this URL before
            if source.url in seen_urls:
                continue
            
            # Create content hash for duplicate detection
            content_hash = self._create_content_hash(source.content)
            
            # Check for near-duplicates using content similarity
            is_duplicate = False
            for existing_hash in seen_hashes:
                if self._calculate_similarity(content_hash, existing_hash) > 0.85:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_sources.append(source)
                seen_hashes.add(content_hash)
                seen_urls.add(source.url)
        
        return unique_sources
    
    def _create_content_hash(self, content: str) -> str:
        """
        Create a hash of content for duplicate detection.
        
        Args:
            content: Text content to hash
            
        Returns:
            Hash string
        """
        # Normalize content: lowercase, remove extra whitespace, remove punctuation
        normalized = re.sub(r'[^\w\s]', '', content.lower())
        normalized = ' '.join(normalized.split())
        
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _calculate_similarity(self, hash1: str, hash2: str) -> float:
        """
        Calculate similarity between two content hashes.
        Simple implementation - could be enhanced with more sophisticated algorithms.
        
        Args:
            hash1: First content hash
            hash2: Second content hash
            
        Returns:
            Similarity score (0.0-1.0)
        """
        # Simple character-level similarity
        if hash1 == hash2:
            return 1.0
        
        # Calculate Hamming distance for MD5 hashes
        if len(hash1) != len(hash2):
            return 0.0
        
        differences = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        similarity = 1.0 - (differences / len(hash1))
        
        return similarity
    
    def _calculate_credibility_scores(self, sources: List[InformationSource]) -> List[Dict[str, Any]]:
        """
        Calculate credibility scores for each source.
        
        Args:
            sources: List of information sources
            
        Returns:
            List of credibility analysis dictionaries
        """
        credibility_analyses = []
        
        for i, source in enumerate(sources):
            factors = self._assess_credibility_factors(source)
            
            # Calculate weighted credibility score (1.0-10.0 scale)
            score = (
                factors.domain_authority * 0.3 +
                factors.content_quality * 0.25 +
                factors.recency * 0.15 +
                factors.author_credibility * 0.15 +
                factors.source_type_weight * 0.15
            ) * 10.0
            
            # Determine credibility factors and concerns
            credibility_factors = []
            concerns = []
            
            if factors.domain_authority > 0.8:
                credibility_factors.append("authoritative domain")
            elif factors.domain_authority < 0.4:
                concerns.append("low domain authority")
            
            if factors.recency > 0.7:
                credibility_factors.append("recent publication")
            elif factors.recency < 0.3:
                concerns.append("outdated information")
            
            if factors.content_quality > 0.7:
                credibility_factors.append("high content quality")
            elif factors.content_quality < 0.4:
                concerns.append("low content quality")
            
            if source.author:
                credibility_factors.append("identified author")
            else:
                concerns.append("anonymous source")
            
            analysis = {
                "source_index": i + 1,
                "credibility_score": round(score, 1),
                "credibility_factors": credibility_factors,
                "concerns": concerns
            }
            
            credibility_analyses.append(analysis)
        
        return credibility_analyses
    
    def _assess_credibility_factors(self, source: InformationSource) -> CredibilityFactors:
        """
        Assess individual credibility factors for a source.
        
        Args:
            source: Information source to assess
            
        Returns:
            CredibilityFactors with individual factor scores
        """
        # Domain authority
        domain = self._extract_domain(source.url)
        domain_authority = self._get_domain_authority(domain)
        
        # Content quality (basic heuristics)
        content_quality = self._assess_content_quality(source.content)
        
        # Recency
        recency = self._assess_recency(source.publish_date)
        
        # Author credibility
        author_credibility = 0.7 if source.author else 0.3
        
        # Source type weight
        source_type_weight = self.SOURCE_TYPE_WEIGHTS.get(source.source_type, 0.5)
        
        return CredibilityFactors(
            domain_authority=domain_authority,
            content_quality=content_quality,
            recency=recency,
            author_credibility=author_credibility,
            source_type_weight=source_type_weight
        )
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except:
            return "unknown"
    
    def _get_domain_authority(self, domain: str) -> float:
        """Get domain authority score"""
        # Check for exact matches first
        if domain in self.DOMAIN_AUTHORITY:
            return self.DOMAIN_AUTHORITY[domain]
        
        # Check for TLD matches
        for tld, score in self.DOMAIN_AUTHORITY.items():
            if domain.endswith(tld):
                return score
        
        return self.DOMAIN_AUTHORITY['default']
    
    def _assess_content_quality(self, content: str) -> float:
        """
        Assess content quality using basic heuristics.
        
        Args:
            content: Text content to assess
            
        Returns:
            Quality score (0.0-1.0)
        """
        if not content or len(content) < 50:
            return 0.1
        
        score = 0.5  # Base score
        
        # Length factor (optimal range: 500-3000 characters)
        length = len(content)
        if 500 <= length <= 3000:
            score += 0.2
        elif length > 3000:
            score += 0.1
        
        # Sentence structure (presence of periods)
        sentences = content.count('.')
        if sentences > 3:
            score += 0.1
        
        # Capitalization (proper sentences)
        if content[0].isupper():
            score += 0.1
        
        # Avoid all caps (spam indicator)
        if content.isupper():
            score -= 0.3
        
        # Check for excessive punctuation (spam indicator)
        punct_ratio = sum(1 for c in content if c in '!?') / len(content)
        if punct_ratio > 0.05:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _assess_recency(self, publish_date: Optional[datetime]) -> float:
        """
        Assess recency of content.
        
        Args:
            publish_date: Publication date (None if unknown)
            
        Returns:
            Recency score (0.0-1.0)
        """
        if not publish_date:
            return 0.5  # Neutral score for unknown dates
        
        now = datetime.now()
        days_old = (now - publish_date).days
        
        # Scoring based on age
        if days_old <= 30:
            return 1.0
        elif days_old <= 90:
            return 0.8
        elif days_old <= 365:
            return 0.6
        elif days_old <= 1095:  # 3 years
            return 0.4
        else:
            return 0.2
    
    def _analyze_with_gemini(self, sources: List[InformationSource]) -> Dict[str, Any]:
        """
        Use Gemini API to analyze sources for contradictions and extract verified facts.
        
        Args:
            sources: List of information sources to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Prepare sources for Gemini analysis
        source_data = []
        for i, source in enumerate(sources):
            source_data.append({
                'url': source.url,
                'content': source.content[:2000],  # Limit content length
                'type': source.source_type,
                'title': source.title,
                'author': source.author
            })
        
        try:
            # Generate fact-checking prompt
            prompt = PromptTemplates.get_fact_checker_prompt(source_data)
            
            # Call Gemini API
            response = self.gemini_client.generate_json(
                prompt=prompt,
                temperature=self.prompt_config.temperature,
                system_instruction=self.prompt_config.system_instruction
            )
            
            logger.info("Gemini analysis completed successfully")
            return response
            
        except Exception as e:
            logger.error("Gemini analysis failed", error=str(e))
            # Return minimal structure on failure
            return {
                'verified_facts': [],
                'contradictions': [],
                'source_credibility': [],
                'analysis_summary': {
                    'total_sources_analyzed': len(sources),
                    'duplicates_found': 0,
                    'contradictions_found': 0,
                    'high_credibility_sources': 0,
                    'overall_reliability_score': 0.5
                }
            }
    
    def identify_contradictions(self, sources: List[InformationSource]) -> List[Dict[str, Any]]:
        """
        Identify contradictions between sources (standalone method).
        
        Args:
            sources: List of information sources
            
        Returns:
            List of contradiction dictionaries
        """
        result = self.check_facts(sources)
        return result.contradictions
    
    def score_credibility(self, source: InformationSource) -> float:
        """
        Score credibility of a single source (standalone method).
        
        Args:
            source: Information source to score
            
        Returns:
            Credibility score (1.0-10.0)
        """
        credibility_analyses = self._calculate_credibility_scores([source])
        return credibility_analyses[0]['credibility_score']
    
    def remove_duplicates(self, sources: List[InformationSource]) -> List[InformationSource]:
        """
        Remove duplicates from sources (standalone method).
        
        Args:
            sources: List of information sources
            
        Returns:
            List of unique sources
        """
        return self._remove_duplicates(sources)

# Utility functions for creating InformationSource objects from different data types

def create_information_source_from_search_result(search_result, content: str = None) -> InformationSource:
    """
    Create InformationSource from SearchResult data model.
    
    Args:
        search_result: SearchResult object
        content: Optional full content (if available)
        
    Returns:
        InformationSource object
    """
    return InformationSource(
        content=content or search_result.snippet,
        url=search_result.url,
        title=search_result.title,
        source_type="web",
        author=None,
        publish_date=None,
        extraction_method="search_api"
    )

def create_information_source_from_scraped_content(scraped_content) -> InformationSource:
    """
    Create InformationSource from ScrapedContent data model.
    
    Args:
        scraped_content: ScrapedContent object
        
    Returns:
        InformationSource object
    """
    return InformationSource(
        content=scraped_content.content,
        url=scraped_content.url,
        title=scraped_content.title,
        source_type="web",
        author=scraped_content.author,
        publish_date=scraped_content.publish_date,
        extraction_method=scraped_content.extraction_method
    )

def create_information_source_from_document(document) -> InformationSource:
    """
    Create InformationSource from Document (vector database) data model.
    
    Args:
        document: Document object from vector database
        
    Returns:
        InformationSource object
    """
    metadata = document.metadata
    return InformationSource(
        content=document.content,
        url=metadata.get('source_url', 'vector_database'),
        title=metadata.get('title', 'Vector Database Document'),
        source_type="vector_db",
        author=metadata.get('author'),
        publish_date=metadata.get('timestamp'),
        extraction_method="vector_search"
    )