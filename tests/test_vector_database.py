"""
Vector database tests for ChromaManager and VectorSearchAgent.
Tests document storage and retrieval operations, validates embedding generation and similarity search.
"""

import pytest
import os
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from utils.chroma_manager import ChromaManager
from agents.vector_search_agent import VectorSearchAgent, SearchCache
from agents.data_models import Document
from utils.gemini_client import GeminiClient, GeminiConfig


class TestChromaManager:
    """Test ChromaManager document storage and retrieval operations"""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_gemini_client(self):
        """Create mock Gemini client for testing"""
        mock_client = Mock(spec=GeminiClient)
        
        # Mock embedding generation - return simple vectors for testing
        def mock_generate_embeddings(texts):
            return [[0.1, 0.2, 0.3] for _ in texts]
        
        mock_client.generate_embeddings = mock_generate_embeddings
        return mock_client
    
    @pytest.fixture
    def chroma_manager(self, temp_db_path, mock_gemini_client):
        """Create ChromaManager instance for testing"""
        return ChromaManager(
            db_path=temp_db_path,
            collection_name="test_collection",
            gemini_client=mock_gemini_client
        )
    
    def test_initialization(self, temp_db_path, mock_gemini_client):
        """Test ChromaManager initialization"""
        manager = ChromaManager(
            db_path=temp_db_path,
            collection_name="test_init",
            gemini_client=mock_gemini_client
        )
        
        # Verify initialization
        assert manager.db_path == temp_db_path
        assert manager.collection_name == "test_init"
        assert manager.gemini_client == mock_gemini_client
        assert manager.collection is not None
        assert os.path.exists(temp_db_path)
    
    def test_add_documents_success(self, chroma_manager):
        """Test successful document addition"""
        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing enables computers to understand text."
        ]
        
        metadatas = [
            {"source_url": "https://example.com/ml", "domain": "example.com", "credibility_score": 8.0},
            {"source_url": "https://example.com/dl", "domain": "example.com", "credibility_score": 7.5},
            {"source_url": "https://example.com/nlp", "domain": "example.com", "credibility_score": 9.0}
        ]
        
        # Add documents
        result = chroma_manager.add_documents(texts, metadatas)
        
        # Verify success
        assert result is True
        
        # Verify documents were added
        stats = chroma_manager.get_collection_stats()
        assert stats["total_documents"] == 3
    
    def test_add_documents_with_custom_ids(self, chroma_manager):
        """Test document addition with custom IDs"""
        texts = ["Test document with custom ID"]
        metadatas = [{"source": "test"}]
        ids = ["custom_id_123"]
        
        result = chroma_manager.add_documents(texts, metadatas, ids)
        
        assert result is True
        stats = chroma_manager.get_collection_stats()
        assert stats["total_documents"] == 1
    
    def test_add_documents_empty_list(self, chroma_manager):
        """Test adding empty document list"""
        result = chroma_manager.add_documents([], [])
        assert result is False
    
    def test_add_documents_mismatched_lengths(self, chroma_manager):
        """Test error handling for mismatched text and metadata lengths"""
        texts = ["Document 1", "Document 2"]
        metadatas = [{"source": "test"}]  # Only one metadata for two texts
        
        with pytest.raises(Exception, match="Number of texts must match"):
            chroma_manager.add_documents(texts, metadatas)
    
    def test_add_documents_long_text_truncation(self, chroma_manager):
        """Test automatic truncation of long texts"""
        long_text = "A" * 6000  # Longer than 5000 char limit
        texts = [long_text]
        metadatas = [{"source": "test"}]
        
        result = chroma_manager.add_documents(texts, metadatas)
        
        assert result is True
        # Verify document was added (truncated)
        stats = chroma_manager.get_collection_stats()
        assert stats["total_documents"] == 1
    
    def test_search_similar_basic(self, chroma_manager):
        """Test basic similarity search"""
        # Add test documents
        texts = [
            "Artificial intelligence and machine learning concepts",
            "Deep neural networks for image recognition",
            "Cooking recipes and kitchen techniques"
        ]
        metadatas = [
            {"domain": "ai.com", "credibility_score": 8.0},
            {"domain": "tech.com", "credibility_score": 7.0},
            {"domain": "food.com", "credibility_score": 6.0}
        ]
        
        chroma_manager.add_documents(texts, metadatas)
        
        # Search for AI-related content
        results = chroma_manager.search_similar("machine learning artificial intelligence", top_k=2)
        
        # Verify results
        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc in results)
        
        if results:
            # First result should be most relevant (AI-related)
            assert "artificial intelligence" in results[0].content.lower() or "machine learning" in results[0].content.lower()
            assert results[0].similarity_score > 0.0
            assert results[0].credibility_score > 0.0
    
    def test_search_similar_with_threshold(self, chroma_manager):
        """Test similarity search with threshold filtering"""
        # Add test documents
        texts = ["Machine learning algorithms", "Cooking pasta recipes"]
        metadatas = [{"credibility_score": 8.0}, {"credibility_score": 6.0}]
        
        chroma_manager.add_documents(texts, metadatas)
        
        # Search with high threshold - should filter out low similarity results
        results = chroma_manager.search_similar(
            "artificial intelligence", 
            top_k=5, 
            similarity_threshold=0.9  # Very high threshold
        )
        
        # Should return fewer or no results due to high threshold
        assert len(results) <= 2
        if results:
            assert all(doc.similarity_score >= 0.9 for doc in results)
    
    def test_search_similar_with_metadata_filter(self, chroma_manager):
        """Test similarity search with metadata filtering"""
        # Add test documents with different domains
        texts = [
            "AI research paper",
            "Tech news article", 
            "Food blog post"
        ]
        metadatas = [
            {"domain": "research.edu", "content_type": "research"},
            {"domain": "tech.com", "content_type": "news"},
            {"domain": "food.blog", "content_type": "blog"}
        ]
        
        chroma_manager.add_documents(texts, metadatas)
        
        # Search with domain filter
        results = chroma_manager.search_similar(
            "artificial intelligence",
            metadata_filter={"domain": "research.edu"}
        )
        
        # Should only return results from research.edu domain
        if results:
            assert all(doc.metadata.get("domain") == "research.edu" for doc in results)
    
    def test_search_similar_empty_query(self, chroma_manager):
        """Test search with empty query"""
        results = chroma_manager.search_similar("", top_k=5)
        assert results == []
        
        results = chroma_manager.search_similar("   ", top_k=5)
        assert results == []
    
    def test_search_similar_no_documents(self, chroma_manager):
        """Test search when no documents exist"""
        results = chroma_manager.search_similar("test query", top_k=5)
        assert results == []
    
    def test_get_collection_stats_empty(self, chroma_manager):
        """Test collection statistics for empty collection"""
        stats = chroma_manager.get_collection_stats()
        
        assert stats["total_documents"] == 0
        assert stats["unique_domains"] == 0
        assert stats["content_types"] == []
        assert stats["average_word_count"] == 0
        assert stats["collection_name"] == "test_collection"
    
    def test_get_collection_stats_with_documents(self, chroma_manager):
        """Test collection statistics with documents"""
        texts = [
            "Short text",
            "This is a longer text with more words for testing purposes"
        ]
        metadatas = [
            {"domain": "example.com", "content_type": "article"},
            {"domain": "test.org", "content_type": "research"}
        ]
        
        chroma_manager.add_documents(texts, metadatas)
        
        stats = chroma_manager.get_collection_stats()
        
        assert stats["total_documents"] == 2
        assert stats["unique_domains"] == 2
        assert "article" in stats["content_types"]
        assert "research" in stats["content_types"]
        assert stats["average_word_count"] > 0
    
    def test_delete_documents(self, chroma_manager):
        """Test document deletion"""
        texts = ["Document 1", "Document 2", "Document 3"]
        metadatas = [{"source": "test"} for _ in texts]
        ids = ["doc1", "doc2", "doc3"]
        
        # Add documents
        chroma_manager.add_documents(texts, metadatas, ids)
        
        # Delete some documents
        result = chroma_manager.delete_documents(["doc1", "doc3"])
        
        assert result is True
        
        # Verify deletion
        stats = chroma_manager.get_collection_stats()
        assert stats["total_documents"] == 1
    
    def test_delete_documents_empty_list(self, chroma_manager):
        """Test deletion with empty ID list"""
        result = chroma_manager.delete_documents([])
        assert result is False
    
    def test_clear_collection(self, chroma_manager):
        """Test clearing entire collection"""
        # Add some documents
        texts = ["Doc 1", "Doc 2"]
        metadatas = [{"source": "test"}, {"source": "test"}]
        chroma_manager.add_documents(texts, metadatas)
        
        # Clear collection
        result = chroma_manager.clear_collection()
        
        assert result is True
        
        # Verify collection is empty
        stats = chroma_manager.get_collection_stats()
        assert stats["total_documents"] == 0
    
    def test_health_check_empty_collection(self, chroma_manager):
        """Test health check on empty collection"""
        health = chroma_manager.health_check()
        assert health is True
    
    def test_health_check_with_documents(self, chroma_manager):
        """Test health check with documents"""
        # Add a document
        chroma_manager.add_documents(["Test document"], [{"source": "test"}])
        
        health = chroma_manager.health_check()
        assert health is True


class TestVectorSearchAgent:
    """Test VectorSearchAgent functionality"""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_gemini_client(self):
        """Create mock Gemini client for testing"""
        mock_client = Mock(spec=GeminiClient)
        
        def mock_generate_embeddings(texts):
            # Generate different embeddings based on content for better testing
            embeddings = []
            for text in texts:
                if "machine learning" in text.lower() or "ai" in text.lower():
                    embeddings.append([0.8, 0.2, 0.1])  # AI-related embedding
                elif "cooking" in text.lower() or "food" in text.lower():
                    embeddings.append([0.1, 0.8, 0.2])  # Food-related embedding
                else:
                    embeddings.append([0.3, 0.3, 0.3])  # Neutral embedding
            return embeddings
        
        mock_client.generate_embeddings = mock_generate_embeddings
        return mock_client
    
    @pytest.fixture
    def chroma_manager(self, temp_db_path, mock_gemini_client):
        """Create ChromaManager instance for testing"""
        return ChromaManager(
            db_path=temp_db_path,
            collection_name="test_search",
            gemini_client=mock_gemini_client
        )
    
    @pytest.fixture
    def vector_search_agent(self, chroma_manager):
        """Create VectorSearchAgent instance for testing"""
        return VectorSearchAgent(chroma_manager, cache_ttl_minutes=1)
    
    @pytest.fixture
    def sample_documents(self, chroma_manager):
        """Add sample documents for testing"""
        texts = [
            "Machine learning algorithms for data analysis",
            "Deep learning neural networks and AI applications", 
            "Cooking techniques for Italian pasta dishes",
            "Recent advances in artificial intelligence research",
            "Traditional recipes from Mediterranean cuisine"
        ]
        
        metadatas = [
            {
                "domain": "ai-research.com", 
                "content_type": "research",
                "credibility_score": 9.0,
                "added_timestamp": (datetime.now() - timedelta(days=1)).isoformat()
            },
            {
                "domain": "tech-news.com", 
                "content_type": "article",
                "credibility_score": 7.5,
                "added_timestamp": (datetime.now() - timedelta(days=5)).isoformat()
            },
            {
                "domain": "food-blog.com", 
                "content_type": "blog",
                "credibility_score": 6.0,
                "added_timestamp": (datetime.now() - timedelta(days=10)).isoformat()
            },
            {
                "domain": "university.edu", 
                "content_type": "research",
                "credibility_score": 9.5,
                "added_timestamp": (datetime.now() - timedelta(hours=2)).isoformat()
            },
            {
                "domain": "cooking.org", 
                "content_type": "article",
                "credibility_score": 7.0,
                "added_timestamp": (datetime.now() - timedelta(days=30)).isoformat()
            }
        ]
        
        chroma_manager.add_documents(texts, metadatas)
        return texts, metadatas
    
    def test_basic_search(self, vector_search_agent, sample_documents):
        """Test basic vector search functionality"""
        results = vector_search_agent.search("artificial intelligence machine learning", top_k=3)
        
        # Should return AI-related documents
        assert len(results) <= 3
        assert all(isinstance(doc, Document) for doc in results)
        
        if results:
            # Check that AI-related content is prioritized
            ai_keywords = ["machine learning", "artificial intelligence", "deep learning", "ai"]
            assert any(keyword in results[0].content.lower() for keyword in ai_keywords)
    
    def test_search_with_similarity_threshold(self, vector_search_agent, sample_documents):
        """Test search with similarity threshold"""
        # High threshold should return fewer results
        results_high = vector_search_agent.search(
            "machine learning", 
            top_k=5, 
            similarity_threshold=0.8
        )
        
        # Low threshold should return more results
        results_low = vector_search_agent.search(
            "machine learning", 
            top_k=5, 
            similarity_threshold=0.1
        )
        
        assert len(results_high) <= len(results_low)
        
        # All results should meet threshold
        for doc in results_high:
            assert doc.similarity_score >= 0.8
    
    def test_search_with_metadata_filters(self, vector_search_agent, sample_documents):
        """Test search with metadata filtering"""
        # Search only research content
        research_results = vector_search_agent.search(
            "artificial intelligence",
            metadata_filters={"content_type": "research"}
        )
        
        # All results should be research type
        for doc in research_results:
            assert doc.metadata.get("content_type") == "research"
    
    def test_rerank_strategies(self, vector_search_agent, sample_documents):
        """Test different re-ranking strategies"""
        query = "artificial intelligence"
        
        # Test similarity ranking
        similarity_results = vector_search_agent.search(
            query, rerank_strategy="similarity", top_k=3
        )
        
        # Test credibility ranking
        credibility_results = vector_search_agent.search(
            query, rerank_strategy="credibility", top_k=3
        )
        
        # Test recency ranking
        recency_results = vector_search_agent.search(
            query, rerank_strategy="recency", top_k=3
        )
        
        # Test balanced ranking
        balanced_results = vector_search_agent.search(
            query, rerank_strategy="balanced", top_k=3
        )
        
        # All should return results
        assert len(similarity_results) > 0
        assert len(credibility_results) > 0
        assert len(recency_results) > 0
        assert len(balanced_results) > 0
        
        # Results may be in different orders due to different ranking strategies
        # Just verify they're all valid Document objects
        for results in [similarity_results, credibility_results, recency_results, balanced_results]:
            assert all(isinstance(doc, Document) for doc in results)
    
    def test_search_caching(self, vector_search_agent, sample_documents):
        """Test search result caching"""
        query = "machine learning"
        
        # First search - should hit database
        results1 = vector_search_agent.search(query, use_cache=True)
        
        # Second search - should use cache
        results2 = vector_search_agent.search(query, use_cache=True)
        
        # Results should be identical
        assert len(results1) == len(results2)
        if results1:
            assert results1[0].content == results2[0].content
        
        # Test cache bypass
        results3 = vector_search_agent.search(query, use_cache=False)
        assert len(results3) == len(results1)  # Should still get same results
    
    def test_search_by_domain(self, vector_search_agent, sample_documents):
        """Test domain-specific search"""
        results = vector_search_agent.search_by_domain(
            "artificial intelligence", 
            "university.edu"
        )
        
        # All results should be from university.edu
        for doc in results:
            assert doc.metadata.get("domain") == "university.edu"
    
    def test_search_by_content_type(self, vector_search_agent, sample_documents):
        """Test content type filtering"""
        results = vector_search_agent.search_by_content_type(
            "machine learning",
            "research"
        )
        
        # All results should be research type
        for doc in results:
            assert doc.metadata.get("content_type") == "research"
    
    def test_search_recent(self, vector_search_agent, sample_documents):
        """Test recent document search"""
        results = vector_search_agent.search_recent(
            "artificial intelligence",
            days=7  # Last 7 days
        )
        
        # Should return recent documents
        cutoff_date = datetime.now() - timedelta(days=7)
        for doc in results:
            timestamp_str = doc.metadata.get("added_timestamp", "")
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
                assert timestamp >= cutoff_date
    
    def test_clear_cache(self, vector_search_agent, sample_documents):
        """Test cache clearing"""
        # Perform a search to populate cache
        vector_search_agent.search("test query", use_cache=True)
        
        # Verify cache has content
        assert len(vector_search_agent.cache.cache) > 0
        
        # Clear cache
        vector_search_agent.clear_cache()
        
        # Verify cache is empty
        assert len(vector_search_agent.cache.cache) == 0
    
    def test_get_search_stats(self, vector_search_agent, sample_documents):
        """Test search statistics"""
        stats = vector_search_agent.get_search_stats()
        
        assert "database_stats" in stats
        assert "cache_stats" in stats
        assert "agent_status" in stats
        
        # Database should have documents
        assert stats["database_stats"]["total_documents"] > 0
        
        # Cache stats should be present
        assert "cached_queries" in stats["cache_stats"]
        assert "cache_ttl_minutes" in stats["cache_stats"]
    
    def test_health_check(self, vector_search_agent, sample_documents):
        """Test vector search agent health check"""
        health = vector_search_agent.health_check()
        assert health is True
    
    def test_empty_search_results(self, vector_search_agent):
        """Test search with no matching documents"""
        # Search without any documents added
        results = vector_search_agent.search("nonexistent topic")
        assert results == []
    
    def test_search_error_handling(self, vector_search_agent):
        """Test search error handling"""
        # Mock ChromaManager to raise exception
        with patch.object(vector_search_agent.chroma_manager, 'search_similar') as mock_search:
            mock_search.side_effect = Exception("Database error")
            
            # Should handle error gracefully
            results = vector_search_agent.search("test query")
            assert results == []


class TestSearchCache:
    """Test SearchCache functionality"""
    
    @pytest.fixture
    def search_cache(self):
        """Create SearchCache instance for testing"""
        return SearchCache(ttl_minutes=1)  # Short TTL for testing
    
    def test_cache_set_and_get(self, search_cache):
        """Test basic cache set and get operations"""
        query = "test query"
        results = [
            Document("Test content", {"source": "test"}, 0.8, 7.0)
        ]
        
        # Set cache
        search_cache.set(query, 5, results)
        
        # Get from cache
        cached_results = search_cache.get(query, 5)
        
        assert cached_results is not None
        assert len(cached_results) == 1
        assert cached_results[0].content == "Test content"
    
    def test_cache_expiration(self, search_cache):
        """Test cache expiration"""
        query = "expiring query"
        results = [Document("Content", {}, 0.8, 7.0)]
        
        # Set cache
        search_cache.set(query, 5, results)
        
        # Should be available immediately
        cached = search_cache.get(query, 5)
        assert cached is not None
        
        # Mock time passage to trigger expiration
        with patch('agents.vector_search_agent.datetime') as mock_datetime:
            # Set current time to 2 minutes in the future
            future_time = datetime.now() + timedelta(minutes=2)
            mock_datetime.now.return_value = future_time
            
            # Should be expired
            cached = search_cache.get(query, 5)
            assert cached is None
    
    def test_cache_key_generation(self, search_cache):
        """Test cache key generation with different parameters"""
        results = [Document("Content", {}, 0.8, 7.0)]
        
        # Same query, different parameters should be different cache entries
        search_cache.set("query", 5, results)
        search_cache.set("query", 10, results)  # Different top_k
        search_cache.set("query", 5, results, {"domain": "test.com"})  # Different filters
        
        # Should have 3 different cache entries
        assert len(search_cache.cache) == 3
    
    def test_cache_clear(self, search_cache):
        """Test cache clearing"""
        # Add some entries
        search_cache.set("query1", 5, [])
        search_cache.set("query2", 5, [])
        
        assert len(search_cache.cache) == 2
        
        # Clear cache
        search_cache.clear()
        
        assert len(search_cache.cache) == 0