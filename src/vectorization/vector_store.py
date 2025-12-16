"""
Vector Database Module

This module provides functionality to store and retrieve vector embeddings
using ChromaDB as the vector store.
"""
from typing import Optional, Union
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings, VectorDBConfig, ensure_directories


class VectorStore:
    """
    Manages vector storage using ChromaDB.
    
    Provides methods for adding, updating, and querying vector embeddings.
    """
    
    def __init__(self, config: Optional[VectorDBConfig] = None):
        """
        Initialize the VectorStore.
        
        Args:
            config: Vector database configuration.
        """
        self.config = config or settings.vectordb
        self._client = None
        self._collection = None
        ensure_directories()
    
    @property
    def client(self):
        """Lazy loading of ChromaDB client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client
    
    @property
    def collection(self):
        """Get or create the collection."""
        if self._collection is None:
            self._collection = self._get_or_create_collection()
        return self._collection
    
    def _create_client(self):
        """Create ChromaDB client with persistence."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )
        
        # Ensure persist directory exists
        persist_dir = self.config.persist_directory
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        print(f"ChromaDB client created. Persist directory: {persist_dir}")
        return client
    
    def _get_or_create_collection(self):
        """Get or create the vector collection."""
        # Map distance metric
        distance_functions = {
            "cosine": "cosine",
            "l2": "l2",
            "ip": "ip"  # inner product
        }
        
        return self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": distance_functions.get(
                self.config.distance_metric, "cosine"
            )}
        )
    
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: Optional[list[str]] = None,
        metadatas: Optional[list[dict]] = None
    ):
        """
        Add vectors to the store.
        
        Args:
            ids: Unique identifiers for each vector.
            embeddings: List of embedding vectors.
            documents: Optional list of original documents.
            metadatas: Optional list of metadata dicts.
        """
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Added {len(ids)} vectors to collection '{self.config.collection_name}'")

    def update(
        self,
        ids: list[str],
        embeddings: Optional[list[list[float]]] = None,
        documents: Optional[list[str]] = None,
        metadatas: Optional[list[dict]] = None
    ):
        """
        Update vectors in the store.
        """
        self.collection.update(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Updated {len(ids)} vectors in collection '{self.config.collection_name}'")
    
    def add_chunks(self, vectorized_chunks: list[dict]):
        """
        Add vectorized chunks to the store.
        
        Args:
            vectorized_chunks: List of dicts from TextVectorizer.vectorize_chunks()
        """
        if not vectorized_chunks:
            return
        
        ids = [chunk["chunk_id"] for chunk in vectorized_chunks]
        embeddings = [chunk["embedding"] for chunk in vectorized_chunks]
        documents = [chunk["content"] for chunk in vectorized_chunks]
        metadatas = [
            {
                "source_file": chunk.get("source_file", ""),
                **chunk.get("metadata", {})
            }
            for chunk in vectorized_chunks
        ]
        
        # Ensure metadata values are JSON serializable
        metadatas = [
            {k: str(v) if not isinstance(v, (str, int, float, bool)) else v
             for k, v in m.items()}
            for m in metadatas
        ]
        
        self.add(ids, embeddings, documents, metadatas)
    
    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 5,
        where: Optional[dict] = None,
        include: Optional[list[str]] = None
    ) -> dict:
        """
        Query the vector store.
        
        Args:
            query_embeddings: Query embedding vectors.
            n_results: Number of results to return.
            where: Optional filter conditions.
            include: What to include in results (documents, metadatas, distances).
            
        Returns:
            Query results dict.
        """
        include = include or ["documents", "metadatas", "distances"]
        
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            include=include
        )
        
        return results
    
    def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> list[dict]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: The query embedding vector.
            top_k: Number of top results to return.
            threshold: Optional similarity threshold.
            
        Returns:
            List of search results with documents and scores.
        """
        results = self.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        formatted = []
        if results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0] if results.get("documents") else [None] * len(ids)
            metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
            distances = results["distances"][0] if results.get("distances") else [0] * len(ids)
            
            for i, (doc_id, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances)):
                # Convert distance to similarity score (for cosine: 1 - distance)
                similarity = 1 - dist if self.config.distance_metric == "cosine" else -dist
                
                if threshold is None or similarity >= threshold:
                    formatted.append({
                        "id": doc_id,
                        "document": doc,
                        "metadata": meta,
                        "distance": dist,
                        "similarity": similarity,
                        "rank": i + 1
                    })
        
        return formatted
    
    def delete(self, ids: list[str]):
        """Delete vectors by ID."""
        self.collection.delete(ids=ids)
        print(f"Deleted {len(ids)} vectors")
    
    def clear(self):
        """Clear all vectors from the collection."""
        self.client.delete_collection(self.config.collection_name)
        self._collection = None
        print(f"Cleared collection '{self.config.collection_name}'")
    
    def count(self) -> int:
        """Get the number of vectors in the collection."""
        return self.collection.count()
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        return {
            "collection_name": self.config.collection_name,
            "count": self.count(),
            "persist_directory": str(self.config.persist_directory),
            "distance_metric": self.config.distance_metric
        }


# Global vector store instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
