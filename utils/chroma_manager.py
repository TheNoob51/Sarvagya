"""
ChromaDB manager for vector database operations.
Handles database initialization, document storage, and similarity search.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import structlog
from utils.gemini_client import GeminiClient, GeminiConfig
from agents.data_models import Document

logger = structlog.get_logger(__name__)

class ChromaManager:
    """
    Manager for ChromaDB vector database operations.
    
    Features:
    - Persistent storage with configurable path
    - Gemini embeddings integration
    - Document addition with metadata
    - Similarity search with scoring
    - Collection management
    """
    
    def __init__(self, db_path: str, collection_name: str, gemini_client: GeminiClient):
        """
        Initialize ChromaDB manager.
        
        Args:
            db_path: Path to store ChromaDB files
            collection_name: Name of the collection to use
            gemini_client: Gemini client for generating embeddings
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.gemini_client = gemini_client
        
        # Ensure database directory exists
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        logger.info("ChromaManager initialized", 
                   db_path=db_path, 
                   collection_name=collection_name)
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            logger.info("Using existing collection", name=self.collection_name)
            return collection
        except ValueError:
            # Collection doesn't exist, create it
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Research knowledge base for intelligent research assistant"}
            )
            logger.info("Created new collection", name=self.collection_name)
            return collection
    
    def add_documents(
        self, 
        texts: List[str], 
        metadatas: List[Dict[str, Any]], 
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Add documents to the vector database with Gemini embeddings.
        
        Args:
            texts: List of document texts to add
            metadatas: List of metadata dictionaries for each document
            ids: Optional list of document IDs (auto-generated if not provided)
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            Exception: If embedding generation or database operation fails
        """
        try:
            if not texts:
                logger.warning("No texts provided to add_documents")
                return False
            
            if len(texts) != len(metadatas):
                raise ValueError("Number of texts must match number of metadata entries")
            
            # Generate IDs if not provided
            if ids is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                ids = [f"doc_{timestamp}_{i}" for i in range(len(texts))]
            
            # Validate texts are not too long (ChromaDB has limits)
            max_length = 5000  # Conservative limit
            processed_texts = []
            for i, text in enumerate(texts):
                if len(text) > max_length:
                    logger.warning("Text too long, truncating", 
                                 doc_id=ids[i], 
                                 original_length=len(text),
                                 truncated_length=max_length)
                    processed_texts.append(text[:max_length])
                else:
                    processed_texts.append(text)
            
            # Generate embeddings using Gemini
            logger.info("Generating embeddings", document_count=len(processed_texts))
            embeddings = self.gemini_client.generate_embeddings(processed_texts)
            
            # Add metadata fields
            enhanced_metadatas = []
            for i, metadata in enumerate(metadatas):
                enhanced_metadata = metadata.copy()
                enhanced_metadata.update({
                    "added_timestamp": datetime.now().isoformat(),
                    "word_count": len(processed_texts[i].split()),
                    "char_count": len(processed_texts[i])
                })
                enhanced_metadatas.append(enhanced_metadata)
            
            # Add to ChromaDB
            self.collection.add(
                documents=processed_texts,
                embeddings=embeddings,
                metadatas=enhanced_metadatas,
                ids=ids
            )
            
            logger.info("Documents added successfully", 
                       count=len(processed_texts),
                       collection_size=self.collection.count())
            
            return True
            
        except Exception as e:
            logger.error("Failed to add documents", error=str(e))
            raise Exception(f"Failed to add documents to ChromaDB: {str(e)}")
    
    def search_similar(
        self, 
        query: str, 
        top_k: int = 5,
        similarity_threshold: float = 0.6,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: Search query text
            top_k: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            metadata_filter: Optional metadata filters
            
        Returns:
            List of Document objects with similarity scores
        """
        try:
            if not query.strip():
                logger.warning("Empty query provided to search_similar")
                return []
            
            # Generate embedding for the query
            logger.debug("Generating query embedding")
            query_embeddings = self.gemini_client.generate_embeddings([query])
            query_embedding = query_embeddings[0]
            
            # Perform similarity search
            search_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": top_k
            }
            
            # Add metadata filter if provided
            if metadata_filter:
                search_kwargs["where"] = metadata_filter
            
            results = self.collection.query(**search_kwargs)
            
            # Process results
            documents = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    # Calculate similarity score (ChromaDB returns distances, convert to similarity)
                    distance = results['distances'][0][i]
                    similarity_score = 1.0 - distance  # Convert distance to similarity
                    
                    # Apply similarity threshold
                    if similarity_score < similarity_threshold:
                        continue
                    
                    # Extract metadata
                    metadata = results['metadatas'][0][i] if results['metadatas'][0] else {}
                    
                    # Get credibility score from metadata or default
                    credibility_score = metadata.get('credibility_score', 5.0)
                    
                    document = Document(
                        content=results['documents'][0][i],
                        metadata=metadata,
                        similarity_score=similarity_score,
                        credibility_score=credibility_score
                    )
                    documents.append(document)
            
            logger.info("Search completed", 
                       query_length=len(query),
                       results_found=len(documents),
                       threshold_applied=similarity_threshold)
            
            return documents
            
        except Exception as e:
            logger.error("Search failed", query=query[:100], error=str(e))
            raise Exception(f"Failed to search ChromaDB: {str(e)}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample of recent documents for metadata analysis
            if count > 0:
                sample_results = self.collection.get(limit=min(100, count))
                
                # Analyze metadata
                domains = set()
                content_types = set()
                total_word_count = 0
                
                if sample_results['metadatas']:
                    for metadata in sample_results['metadatas']:
                        if 'domain' in metadata:
                            domains.add(metadata['domain'])
                        if 'content_type' in metadata:
                            content_types.add(metadata['content_type'])
                        if 'word_count' in metadata:
                            total_word_count += metadata['word_count']
                
                avg_word_count = total_word_count / len(sample_results['metadatas']) if sample_results['metadatas'] else 0
                
                stats = {
                    "total_documents": count,
                    "unique_domains": len(domains),
                    "content_types": list(content_types),
                    "average_word_count": round(avg_word_count, 2),
                    "collection_name": self.collection_name,
                    "database_path": self.db_path
                }
            else:
                stats = {
                    "total_documents": 0,
                    "unique_domains": 0,
                    "content_types": [],
                    "average_word_count": 0,
                    "collection_name": self.collection_name,
                    "database_path": self.db_path
                }
            
            logger.info("Collection stats retrieved", **stats)
            return stats
            
        except Exception as e:
            logger.error("Failed to get collection stats", error=str(e))
            return {"error": str(e)}
    
    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents by their IDs.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not ids:
                logger.warning("No IDs provided to delete_documents")
                return False
            
            self.collection.delete(ids=ids)
            
            logger.info("Documents deleted successfully", 
                       deleted_count=len(ids),
                       remaining_count=self.collection.count())
            
            return True
            
        except Exception as e:
            logger.error("Failed to delete documents", ids=ids, error=str(e))
            return False
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self._get_or_create_collection()
            
            logger.info("Collection cleared successfully")
            return True
            
        except Exception as e:
            logger.error("Failed to clear collection", error=str(e))
            return False
    
    def health_check(self) -> bool:
        """
        Perform a health check on the ChromaDB connection.
        
        Returns:
            True if database is accessible, False otherwise
        """
        try:
            # Try to get collection count
            count = self.collection.count()
            
            # Try a simple query if there are documents
            if count > 0:
                test_results = self.collection.get(limit=1)
                is_healthy = len(test_results['ids']) > 0
            else:
                is_healthy = True  # Empty collection is still healthy
            
            logger.info("ChromaDB health check completed", 
                       is_healthy=is_healthy, 
                       document_count=count)
            
            return is_healthy
            
        except Exception as e:
            logger.error("ChromaDB health check failed", error=str(e))
            return False