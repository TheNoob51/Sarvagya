"""
Vector Search Agent for semantic search in ChromaDB.
Implements intelligent search with re-ranking and caching.
"""

import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import structlog
from utils.chroma_manager import ChromaManager
from utils.gemini_client import GeminiClient
from agents.data_models import Document

logger = structlog.get_logger(__name__)

class SearchCache:
    """Simple in-memory cache for search results"""
    
    def __init__(self, ttl_minutes: int = 60):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_minutes = ttl_minutes
    
    def _generate_key(self, query: str, top_k: int, filters: Optional[Dict] = None) -> str:
        """Generate cache key from search parameters"""
        filter_str = str(sorted(filters.items())) if filters else ""
        return f"{query}:{top_k}:{filter_str}"
    
    def get(self, query: str, top_k: int, filters: Optional[Dict] = None) -> Optional[List[Document]]:
        """Get cached results if available and not expired"""
        key = self._generate_key(query, top_k, filters)
        
        if key in self.cache:
            cached_data = self.cache[key]
            cached_time = cached_data['timestamp']
            
            # Check if cache is still valid
            if datetime.now() - cached_time < timedelta(minutes=self.ttl_minutes):
                logger.debug("Cache hit", query=query[:50], cached_results=len(cached_data['results']))
                return cached_data['results']
            else:
                # Remove expired entry
                del self.cache[key]
                logger.debug("Cache expired", query=query[:50])
        
        return None
    
    def set(self, query: str, top_k: int, results: List[Document], filters: Optional[Dict] = None):
        """Cache search results"""
        key = self._generate_key(query, top_k, filters)
        self.cache[key] = {
            'results': results,
            'timestamp': datetime.now()
        }
        logger.debug("Results cached", query=query[:50], result_count=len(results))
    
    def clear(self):
        """Clear all cached results"""
        self.cache.clear()
        logger.info("Search cache cleared")

class VectorSearchAgent:
    """
    Vector Search Agent for semantic search operations.
    
    Features:
    - Semantic similarity search using ChromaDB
    - Re-ranking by similarity, recency, and credibility
    - Threshold filtering for quality control
    - Metadata-based search filtering
    - Query result caching for performance
    - Advanced search strategies
    """
    
    def __init__(self, chroma_manager: ChromaManager, cache_ttl_minutes: int = 60):
        """
        Initialize Vector Search Agent.
        
        Args:
            chroma_manager: ChromaManager instance for database operations
            cache_ttl_minutes: Time-to-live for cached results in minutes
        """
        self.chroma_manager = chroma_manager
        self.cache = SearchCache(ttl_minutes=cache_ttl_minutes)
        
        logger.info("VectorSearchAgent initialized", cache_ttl=cache_ttl_minutes)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.6,
        metadata_filters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        rerank_strategy: str = "balanced"  # "similarity", "recency", "credibility", "balanced"
    ) -> List[Document]:
        """
        Perform semantic search with advanced filtering and re-ranking.
        
        Args:
            query: Search query text
            top_k: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            metadata_filters: Optional metadata filters (e.g., {"domain": "example.com"})
            use_cache: Whether to use cached results
            rerank_strategy: Strategy for re-ranking results
            
        Returns:
            List of Document objects ranked by the specified strategy
        """
        try:
            start_time = time.time()
            
            # Check cache first
            if use_cache:
                cached_results = self.cache.get(query, top_k, metadata_filters)
                if cached_results is not None:
                    # Apply threshold filtering to cached results
                    filtered_results = [
                        doc for doc in cached_results 
                        if doc.similarity_score >= similarity_threshold
                    ]
                    return filtered_results[:top_k]
            
            # Perform search using ChromaManager
            # Request more results than needed for better re-ranking
            search_limit = min(top_k * 3, 50)  # Get 3x results for re-ranking, max 50
            
            raw_results = self.chroma_manager.search_similar(
                query=query,
                top_k=search_limit,
                similarity_threshold=similarity_threshold,
                metadata_filter=metadata_filters
            )
            
            if not raw_results:
                logger.info("No results found", query=query[:50])
                return []
            
            # Re-rank results based on strategy
            reranked_results = self._rerank_results(raw_results, rerank_strategy)
            
            # Limit to requested number
            final_results = reranked_results[:top_k]
            
            # Cache results
            if use_cache:
                self.cache.set(query, top_k, final_results, metadata_filters)
            
            search_time = time.time() - start_time
            logger.info("Vector search completed",
                       query_length=len(query),
                       results_found=len(final_results),
                       search_time=round(search_time, 3),
                       rerank_strategy=rerank_strategy)
            
            return final_results
            
        except Exception as e:
            logger.error("Vector search failed", query=query[:100], error=str(e))
            return []
    
    def _rerank_results(self, results: List[Document], strategy: str) -> List[Document]:
        """
        Re-rank search results based on the specified strategy.
        
        Args:
            results: List of documents to re-rank
            strategy: Re-ranking strategy
            
        Returns:
            Re-ranked list of documents
        """
        if not results:
            return results
        
        try:
            if strategy == "similarity":
                # Sort by similarity score only
                return sorted(results, key=lambda x: x.similarity_score, reverse=True)
            
            elif strategy == "recency":
                # Sort by recency (timestamp in metadata)
                def recency_key(doc):
                    timestamp_str = doc.metadata.get('added_timestamp', '1970-01-01T00:00:00')
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        return timestamp
                    except:
                        return datetime.min
                
                return sorted(results, key=recency_key, reverse=True)
            
            elif strategy == "credibility":
                # Sort by credibility score
                return sorted(results, key=lambda x: x.credibility_score, reverse=True)
            
            elif strategy == "balanced":
                # Balanced scoring combining similarity, recency, and credibility
                def balanced_score(doc):
                    similarity_weight = 0.5
                    recency_weight = 0.3
                    credibility_weight = 0.2
                    
                    # Normalize similarity score (already 0-1)
                    similarity_norm = doc.similarity_score
                    
                    # Normalize credibility score (assume 1-10 scale)
                    credibility_norm = doc.credibility_score / 10.0
                    
                    # Normalize recency (days since added, max 365 days)
                    timestamp_str = doc.metadata.get('added_timestamp', '1970-01-01T00:00:00')
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        days_old = (datetime.now() - timestamp).days
                        recency_norm = max(0, 1 - (days_old / 365))  # Newer = higher score
                    except:
                        recency_norm = 0
                    
                    # Calculate weighted score
                    score = (
                        similarity_weight * similarity_norm +
                        recency_weight * recency_norm +
                        credibility_weight * credibility_norm
                    )
                    
                    return score
                
                return sorted(results, key=balanced_score, reverse=True)
            
            else:
                logger.warning("Unknown rerank strategy, using similarity", strategy=strategy)
                return sorted(results, key=lambda x: x.similarity_score, reverse=True)
                
        except Exception as e:
            logger.error("Re-ranking failed, returning original order", error=str(e))
            return results
    
    def search_by_domain(self, query: str, domain: str, top_k: int = 5) -> List[Document]:
        """
        Search for documents from a specific domain.
        
        Args:
            query: Search query text
            domain: Domain to filter by (e.g., "wikipedia.org")
            top_k: Maximum number of results
            
        Returns:
            List of documents from the specified domain
        """
        metadata_filters = {"domain": domain}
        return self.search(
            query=query,
            top_k=top_k,
            metadata_filters=metadata_filters,
            rerank_strategy="similarity"
        )
    
    def search_by_content_type(self, query: str, content_type: str, top_k: int = 5) -> List[Document]:
        """
        Search for documents of a specific content type.
        
        Args:
            query: Search query text
            content_type: Content type to filter by (e.g., "article", "research", "news")
            top_k: Maximum number of results
            
        Returns:
            List of documents of the specified content type
        """
        metadata_filters = {"content_type": content_type}
        return self.search(
            query=query,
            top_k=top_k,
            metadata_filters=metadata_filters,
            rerank_strategy="balanced"
        )
    
    def search_recent(self, query: str, days: int = 30, top_k: int = 5) -> List[Document]:
        """
        Search for recent documents within the specified time period.
        
        Args:
            query: Search query text
            days: Number of days to look back
            top_k: Maximum number of results
            
        Returns:
            List of recent documents matching the query
        """
        # Get all results first, then filter by date
        all_results = self.search(
            query=query,
            top_k=top_k * 2,  # Get more results for filtering
            use_cache=False,  # Don't cache since this is time-sensitive
            rerank_strategy="recency"
        )
        
        # Filter by recency
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_results = []
        
        for doc in all_results:
            timestamp_str = doc.metadata.get('added_timestamp', '1970-01-01T00:00:00')
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                if timestamp >= cutoff_date:
                    recent_results.append(doc)
            except:
                continue  # Skip documents with invalid timestamps
        
        return recent_results[:top_k]
    
    def get_related_documents(self, document_id: str, top_k: int = 5) -> List[Document]:
        """
        Find documents related to a specific document.
        
        Args:
            document_id: ID of the reference document
            top_k: Maximum number of related documents to return
            
        Returns:
            List of related documents
        """
        try:
            # Get the reference document content
            # Note: This would require extending ChromaManager to get document by ID
            # For now, we'll implement a simplified version
            
            logger.warning("get_related_documents not fully implemented", document_id=document_id)
            return []
            
        except Exception as e:
            logger.error("Failed to find related documents", document_id=document_id, error=str(e))
            return []
    
    def clear_cache(self):
        """Clear the search cache"""
        self.cache.clear()
    
    def get_search_stats(self) -> Dict[str, Any]:
        """
        Get statistics about search operations.
        
        Returns:
            Dictionary with search statistics
        """
        try:
            # Get ChromaDB stats
            chroma_stats = self.chroma_manager.get_collection_stats()
            
            # Add cache stats
            cache_stats = {
                "cached_queries": len(self.cache.cache),
                "cache_ttl_minutes": self.cache.ttl_minutes
            }
            
            return {
                "database_stats": chroma_stats,
                "cache_stats": cache_stats,
                "agent_status": "active"
            }
            
        except Exception as e:
            logger.error("Failed to get search stats", error=str(e))
            return {"error": str(e)}
    
    def health_check(self) -> bool:
        """
        Perform a health check on the vector search system.
        
        Returns:
            True if system is healthy, False otherwise
        """
        try:
            # Check ChromaDB health
            chroma_healthy = self.chroma_manager.health_check()
            
            if not chroma_healthy:
                return False
            
            # Test search functionality if there are documents
            stats = self.chroma_manager.get_collection_stats()
            if stats.get("total_documents", 0) > 0:
                # Try a simple search
                test_results = self.search("test", top_k=1, use_cache=False)
                search_healthy = isinstance(test_results, list)
            else:
                search_healthy = True  # No documents to search is still healthy
            
            logger.info("Vector search health check completed",
                       chroma_healthy=chroma_healthy,
                       search_healthy=search_healthy)
            
            return chroma_healthy and search_healthy
            
        except Exception as e:
            logger.error("Vector search health check failed", error=str(e))
            return False