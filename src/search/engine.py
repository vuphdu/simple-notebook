"""
Search Module

This module provides functionality for semantic search through vectorized documents.
"""
from typing import Optional, Union
from pathlib import Path
from datetime import datetime
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    settings, SearchConfig, 
    PROCESS_INPUT_DIR, PROCESS_OUTPUT_DIR,
    ensure_directories
)
from src.vectorization import get_vectorizer, get_vector_store


class SearchResult:
    """Represents a single search result."""
    
    def __init__(
        self,
        rank: int,
        document: str,
        similarity: float,
        metadata: dict,
        chunk_id: str
    ):
        self.rank = rank
        self.document = document
        self.similarity = similarity
        self.metadata = metadata
        self.chunk_id = chunk_id
    
    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "document": self.document,
            "similarity": round(self.similarity, 4),
            "metadata": self.metadata,
            "chunk_id": self.chunk_id
        }
    
    def __repr__(self):
        return f"SearchResult(rank={self.rank}, similarity={self.similarity:.4f})"


class SearchEngine:
    """
    Semantic search engine for vectorized documents.
    
    Handles query processing, search execution, and result formatting.
    """
    
    def __init__(self, config: Optional[SearchConfig] = None):
        """
        Initialize the SearchEngine.
        
        Args:
            config: Search configuration.
        """
        self.config = config or settings.search
        self._vectorizer = None
        self._vector_store = None
        ensure_directories()
    
    @property
    def vectorizer(self):
        """Get the vectorizer instance."""
        if self._vectorizer is None:
            self._vectorizer = get_vectorizer()
        return self._vectorizer
    
    @property
    def vector_store(self):
        """Get the vector store instance."""
        if self._vector_store is None:
            self._vector_store = get_vector_store()
        return self._vector_store
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        filter_metadata: Optional[dict] = None
    ) -> list[SearchResult]:
        """
        Perform semantic search.
        
        Args:
            query: The search query text.
            top_k: Number of top results to return.
            threshold: Minimum similarity threshold.
            filter_metadata: Optional metadata filters.
            
        Returns:
            List of SearchResult objects ordered by relevance.
        """
        top_k = top_k or self.config.top_k
        threshold = threshold if threshold is not None else self.config.score_threshold
        
        # Vectorize the query
        query_embedding = self.vectorizer.vectorize(query)[0].tolist()
        
        # Search in vector store
        raw_results = self.vector_store.search_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=threshold
        )
        
        # Convert to SearchResult objects
        results = []
        for r in raw_results:
            result = SearchResult(
                rank=r["rank"],
                document=r["document"],
                similarity=r["similarity"],
                metadata=r["metadata"],
                chunk_id=r["id"]
            )
            results.append(result)
        
        return results
    
    def search_and_save(
        self,
        query: str,
        query_id: Optional[str] = None,
        **kwargs
    ) -> list[SearchResult]:
        """
        Perform search and save input/output to process directories.
        
        Args:
            query: The search query text.
            query_id: Optional identifier for the query.
            **kwargs: Additional arguments passed to search().
            
        Returns:
            List of SearchResult objects.
        """
        timestamp = datetime.now().isoformat()
        query_id = query_id or f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save input
        input_data = {
            "query_id": query_id,
            "query": query,
            "timestamp": timestamp,
            "parameters": {
                "top_k": kwargs.get("top_k", self.config.top_k),
                "threshold": kwargs.get("threshold", self.config.score_threshold)
            }
        }
        
        input_file = PROCESS_INPUT_DIR / f"{query_id}.json"
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f, ensure_ascii=False, indent=2)
        
        # Perform search
        results = self.search(query, **kwargs)
        
        # Save output
        output_data = {
            "query_id": query_id,
            "query": query,
            "timestamp": timestamp,
            "result_count": len(results),
            "results": [r.to_dict() for r in results]
        }
        
        output_file = PROCESS_OUTPUT_DIR / f"{query_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"Search completed. Input: {input_file}, Output: {output_file}")
        
        return results
    
    def batch_search(
        self,
        queries: list[str],
        **kwargs
    ) -> dict[str, list[SearchResult]]:
        """
        Perform batch search for multiple queries.
        
        Args:
            queries: List of query texts.
            **kwargs: Additional arguments passed to search().
            
        Returns:
            Dict mapping query to list of results.
        """
        results = {}
        for query in queries:
            results[query] = self.search(query, **kwargs)
        return results
    
    def format_results(
        self,
        results: list[SearchResult],
        format_type: str = "text"
    ) -> str:
        """
        Format search results for display.
        
        Args:
            results: List of SearchResult objects.
            format_type: Output format ('text', 'json', 'markdown').
            
        Returns:
            Formatted string.
        """
        if not results:
            return "No results found."
        
        if format_type == "json":
            return json.dumps(
                [r.to_dict() for r in results],
                ensure_ascii=False,
                indent=2
            )
        
        elif format_type == "markdown":
            lines = ["# Search Results\n"]
            for r in results:
                lines.append(f"## Result {r.rank} (Similarity: {r.similarity:.4f})")
                lines.append(f"\n**Source:** {r.metadata.get('source_file', 'Unknown')}\n")
                lines.append(f"```\n{r.document}\n```\n")
            return "\n".join(lines)
        
        else:  # text
            lines = [f"Found {len(results)} results:\n"]
            for r in results:
                lines.append(f"[{r.rank}] Score: {r.similarity:.4f}")
                lines.append(f"    Source: {r.metadata.get('source_file', 'Unknown')}")
                lines.append(f"    Content: {r.document[:200]}...")
                lines.append("")
            return "\n".join(lines)


# Global search engine instance
_search_engine: Optional[SearchEngine] = None


def get_search_engine() -> SearchEngine:
    """Get or create the global search engine instance."""
    global _search_engine
    if _search_engine is None:
        _search_engine = SearchEngine()
    return _search_engine


def search(query: str, **kwargs) -> list[SearchResult]:
    """
    Convenience function for semantic search.
    
    Args:
        query: The search query text.
        **kwargs: Additional search parameters.
        
    Returns:
        List of SearchResult objects.
    """
    engine = get_search_engine()
    return engine.search(query, **kwargs)
