"""
FAISS Vector Store Module

This module provides a FAISS-based vector store for fast similarity search.
FAISS (Facebook AI Similarity Search) is optimized for efficient similarity search
and clustering of dense vectors.
"""
from typing import Optional
from pathlib import Path
import numpy as np
import pickle
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings, VectorDBConfig, ensure_directories


class FaissVectorStore:
    """
    FAISS-based vector store for fast similarity search.
    
    Supports multiple index types:
    - Flat: Exact search, best for smaller datasets
    - IVF: Approximate search, better for large datasets
    - HNSW: Graph-based approximate search
    """
    
    def __init__(self, config: Optional[VectorDBConfig] = None):
        """
        Initialize the FaissVectorStore.
        
        Args:
            config: Vector database configuration.
        """
        self.config = config or settings.vectordb
        self._index = None
        self._metadata = {}  # Store metadata separately
        self._documents = {}  # Store document content
        self._id_to_idx = {}  # Map string IDs to FAISS indices
        self._idx_to_id = {}  # Map FAISS indices to string IDs
        self._dimension = None
        ensure_directories()
        
        # Load existing index if available
        self._load_if_exists()
    
    @property
    def index_path(self) -> Path:
        """Path to the FAISS index file."""
        return self.config.faiss_index_dir / f"{self.config.collection_name}.index"
    
    @property
    def metadata_path(self) -> Path:
        """Path to the metadata file."""
        return self.config.faiss_index_dir / f"{self.config.collection_name}_metadata.pkl"
    
    def _load_if_exists(self):
        """Load existing index and metadata if available."""
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                import faiss
                self._index = faiss.read_index(str(self.index_path))
                
                with open(self.metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self._metadata = data.get('metadata', {})
                    self._documents = data.get('documents', {})
                    self._id_to_idx = data.get('id_to_idx', {})
                    self._idx_to_id = data.get('idx_to_id', {})
                    self._dimension = data.get('dimension')
                
                print(f"Loaded FAISS index with {self._index.ntotal} vectors")
            except Exception as e:
                print(f"Failed to load existing index: {e}")
    
    def _create_index(self, dimension: int):
        """Create a new FAISS index."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required. Install with: pip install faiss-cpu"
            )
        
        self._dimension = dimension
        index_type = self.config.faiss_index_type
        
        if index_type == "Flat":
            # Exact search - L2 distance
            if self.config.distance_metric == "cosine":
                # For cosine similarity, we use inner product on normalized vectors
                self._index = faiss.IndexFlatIP(dimension)
            else:
                self._index = faiss.IndexFlatL2(dimension)
        elif index_type == "IVF":
            # Inverted file index for large datasets
            quantizer = faiss.IndexFlatL2(dimension)
            nlist = min(100, max(1, len(self._id_to_idx) // 10))
            self._index = faiss.IndexIVFFlat(quantizer, dimension, max(nlist, 1))
        elif index_type == "HNSW":
            # Hierarchical Navigable Small World graph
            self._index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            # Default to Flat
            self._index = faiss.IndexFlatIP(dimension)
        
        print(f"Created FAISS index: {index_type}, dimension: {dimension}")
    
    def _save(self):
        """Save index and metadata to disk."""
        import faiss
        
        self.config.faiss_index_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self._index, str(self.index_path))
        
        # Save metadata
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({
                'metadata': self._metadata,
                'documents': self._documents,
                'id_to_idx': self._id_to_idx,
                'idx_to_id': self._idx_to_id,
                'dimension': self._dimension,
            }, f)
    
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
        if not embeddings:
            return
        
        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        # Normalize for cosine similarity
        if self.config.distance_metric == "cosine":
            import faiss
            faiss.normalize_L2(embeddings_np)
        
        # Create index if not exists
        if self._index is None:
            self._create_index(embeddings_np.shape[1])
        
        # Add vectors to index
        start_idx = self._index.ntotal
        self._index.add(embeddings_np)
        
        # Store metadata and documents
        for i, doc_id in enumerate(ids):
            idx = start_idx + i
            self._id_to_idx[doc_id] = idx
            self._idx_to_id[idx] = doc_id
            
            if documents:
                self._documents[doc_id] = documents[i]
            if metadatas:
                self._metadata[doc_id] = metadatas[i]
        
        # Save to disk
        self._save()
        print(f"Added {len(ids)} vectors to FAISS index")
    
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
    
    def update(
        self,
        ids: list[str],
        embeddings: Optional[list[list[float]]] = None,
        documents: Optional[list[str]] = None,
        metadatas: Optional[list[dict]] = None
    ):
        """
        Update vectors in the store.
        
        Strategy: Append new vectors to FAISS index and update mappings.
        Old vectors remain in the index but are unreachable via ID mapping.
        """
        if not embeddings:
            # If only updating metadata/content, we can just update the dicts
            # But usually we update everything.
            # For now, require embeddings for simplicity or handle metadata-only update
            pass

        if embeddings:
            embeddings_np = np.array(embeddings, dtype=np.float32)
            
            # Normalize
            if self.config.distance_metric == "cosine":
                import faiss
                faiss.normalize_L2(embeddings_np)
            
            # Add to index
            start_idx = self._index.ntotal
            self._index.add(embeddings_np)
            
            for i, doc_id in enumerate(ids):
                new_idx = start_idx + i
                
                # Handle old mapping
                old_idx = self._id_to_idx.get(doc_id)
                if old_idx is not None:
                    # Remove old index mapping so it won't be found in search
                    self._idx_to_id.pop(old_idx, None)
                
                # Update mappings
                self._id_to_idx[doc_id] = new_idx
                self._idx_to_id[new_idx] = doc_id
                
                # Update content/metadata
                if documents:
                    self._documents[doc_id] = documents[i]
                if metadatas:
                    self._metadata[doc_id] = metadatas[i]
            
            self._save()
            print(f"Updated {len(ids)} vectors in FAISS index")
        
        elif documents or metadatas:
            # Metadata/Content only update
            for i, doc_id in enumerate(ids):
                if documents:
                    self._documents[doc_id] = documents[i]
                if metadatas:
                    self._metadata[doc_id] = metadatas[i]
            self._save()
            print(f"Updated metadata/content for {len(ids)} items")
    
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
        if self._index is None or self._index.ntotal == 0:
            return []
        
        import faiss
        
        query_np = np.array([query_embedding], dtype=np.float32)
        
        # Normalize for cosine similarity
        if self.config.distance_metric == "cosine":
            faiss.normalize_L2(query_np)
        
        # Search
        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(query_np, k)
        
        # Format results
        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # Invalid result
                continue
            
            doc_id = self._idx_to_id.get(idx)
            if not doc_id:
                continue
            
            # Convert distance to similarity score
            if self.config.distance_metric == "cosine":
                # For inner product on normalized vectors, score is directly the similarity
                similarity = float(dist)
            else:
                # For L2 distance, convert to similarity
                similarity = 1 / (1 + float(dist))
            
            if threshold is not None and similarity < threshold:
                continue
            
            results.append({
                "id": doc_id,
                "document": self._documents.get(doc_id),
                "metadata": self._metadata.get(doc_id, {}),
                "distance": float(dist),
                "similarity": similarity,
                "rank": rank + 1
            })
        
        return results
    
    def delete(self, ids: list[str]):
        """
        Delete vectors by ID.
        
        Note: FAISS doesn't support direct deletion. We rebuild the index
        without the deleted vectors.
        """
        if not ids or self._index is None:
            return
        
        # Remove from metadata
        for doc_id in ids:
            self._documents.pop(doc_id, None)
            self._metadata.pop(doc_id, None)
            idx = self._id_to_idx.pop(doc_id, None)
            if idx is not None:
                self._idx_to_id.pop(idx, None)
        
        # Rebuild index with remaining vectors
        # This is expensive but FAISS doesn't support direct deletion
        self._rebuild_index()
        print(f"Deleted {len(ids)} vectors")
    
    def _rebuild_index(self):
        """Rebuild the index from remaining data."""
        if not self._documents:
            self._index = None
            self._id_to_idx = {}
            self._idx_to_id = {}
            self._save()
            return
        
        # This would require re-computing embeddings, so we skip for now
        # In a real implementation, you'd store embeddings alongside documents
        print("Warning: Index rebuild would require re-computing embeddings")
    
    def clear(self):
        """Clear all vectors from the index."""
        self._index = None
        self._metadata = {}
        self._documents = {}
        self._id_to_idx = {}
        self._idx_to_id = {}
        
        # Remove files
        if self.index_path.exists():
            self.index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()
        
        print(f"Cleared FAISS index '{self.config.collection_name}'")
    
    def count(self) -> int:
        """Get the number of vectors in the index."""
        return self._index.ntotal if self._index else 0
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        return {
            "backend": "faiss",
            "collection_name": self.config.collection_name,
            "count": self.count(),
            "index_type": self.config.faiss_index_type,
            "dimension": self._dimension,
            "index_directory": str(self.config.faiss_index_dir),
            "distance_metric": self.config.distance_metric
        }


# Global FAISS store instance
_faiss_store: Optional[FaissVectorStore] = None


def get_faiss_store() -> FaissVectorStore:
    """Get or create the global FAISS store instance."""
    global _faiss_store
    if _faiss_store is None:
        _faiss_store = FaissVectorStore()
    return _faiss_store
