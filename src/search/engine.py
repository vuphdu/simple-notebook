"""
Search Module

This module provides functionality for semantic search through vectorized documents.
"""
from typing import Optional
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
        filter_metadata: Optional[dict] = None,
        use_hybrid: Optional[bool] = None
    ) -> list[SearchResult]:
        """
        Perform semantic search with optional hybrid mode.
        
        Hybrid search combines:
        - Vector search (semantic similarity)
        - BM25 search (keyword matching)
        
        Final score = alpha * vector_score + (1 - alpha) * normalized_bm25_score
        
        Args:
            query: The search query text.
            top_k: Number of top results to return.
            threshold: Minimum similarity threshold.
            filter_metadata: Optional metadata filters.
            use_hybrid: Override config's use_hybrid setting.
            
        Returns:
            List of SearchResult objects ordered by relevance.
        """
        top_k = top_k or self.config.top_k
        threshold = threshold if threshold is not None else self.config.score_threshold
        use_hybrid = use_hybrid if use_hybrid is not None else self.config.use_hybrid
        
        # Vectorize the query
        query_embedding = self.vectorizer.vectorize(query)[0].tolist()
        
        # Get more candidates for re-ranking if using hybrid
        fetch_k = top_k * 3 if use_hybrid else top_k
        
        # Search in vector store
        raw_results = self.vector_store.search_similar(
            query_embedding=query_embedding,
            top_k=fetch_k,
            threshold=0.0  # Get all candidates, filter later
        )
        
        if not raw_results:
            return []
        
        # If hybrid mode, combine with BM25
        if use_hybrid and len(raw_results) > 0:
            results = self._hybrid_rerank(query, raw_results, top_k, threshold)
        else:
            # Pure vector search - apply threshold and limit
            results = []
            for r in raw_results[:top_k]:
                if r["similarity"] < threshold:
                    continue
                result = SearchResult(
                    rank=len(results) + 1,
                    document=r["document"],
                    similarity=r["similarity"],
                    metadata=r["metadata"],
                    chunk_id=r["id"]
                )
                results.append(result)
        
        return results
    
    def _hybrid_rerank(
        self,
        query: str,
        vector_results: list[dict],
        top_k: int,
        threshold: float
    ) -> list[SearchResult]:
        """
        Re-rank results using hybrid scoring (vector + BM25).
        
        Args:
            query: Original query text.
            vector_results: Results from vector search.
            top_k: Number of results to return.
            threshold: Minimum score threshold.
            
        Returns:
            Re-ranked SearchResult list.
        """
        from .bm25_search import BM25Index
        
        alpha = self.config.hybrid_alpha
        
        # Build temporary BM25 index from candidate documents
        bm25 = BM25Index(k1=self.config.bm25_k1, b=self.config.bm25_b)
        
        # Prepare documents for BM25
        docs_for_bm25 = []
        for r in vector_results:
            docs_for_bm25.append({
                "chunk_id": r["id"],
                "content": r["document"],
                "metadata": r["metadata"]
            })
        
        bm25.add_documents(docs_for_bm25)
        
        # Get BM25 scores
        bm25_results = bm25.search(query, top_k=len(vector_results))
        
        # Create lookup for BM25 scores
        bm25_scores = {r["id"]: r["score"] for r in bm25_results}
        
        # Normalize BM25 scores to [0, 1]
        if bm25_scores:
            max_bm25 = max(bm25_scores.values()) if bm25_scores.values() else 1.0
            if max_bm25 > 0:
                bm25_scores = {k: v / max_bm25 for k, v in bm25_scores.items()}
        
        # Combine scores
        combined_results = []
        for r in vector_results:
            doc_id = r["id"]
            vector_score = r["similarity"]
            bm25_score = bm25_scores.get(doc_id, 0.0)
            
            # Hybrid score: weighted combination
            hybrid_score = alpha * vector_score + (1 - alpha) * bm25_score
            
            if hybrid_score >= threshold:
                combined_results.append({
                    "id": doc_id,
                    "document": r["document"],
                    "metadata": r["metadata"],
                    "vector_score": vector_score,
                    "bm25_score": bm25_score,
                    "hybrid_score": hybrid_score
                })
        
        # Sort by hybrid score
        combined_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        # Convert to SearchResult
        results = []
        for i, r in enumerate(combined_results[:top_k]):
            result = SearchResult(
                rank=i + 1,
                document=r["document"],
                similarity=r["hybrid_score"],  # Use hybrid score as similarity
                metadata={
                    **r["metadata"],
                    "_vector_score": round(r["vector_score"], 4),
                    "_bm25_score": round(r["bm25_score"], 4),
                    "_hybrid_mode": True
                },
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
            format_type: Output format ('text', 'json', 'markdown', 'compact').
            
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
        
        elif format_type == "compact":
            # Compact format for AI assistants - minimal, easy to parse
            lines = []
            for r in results:
                source = Path(r.metadata.get('source_file', 'unknown')).name
                page = r.metadata.get('page_number', r.metadata.get('chunk_index', '?'))
                score = f"{r.similarity:.2f}"
                
                # Header line
                lines.append(f"--- [{r.rank}] {source} (p.{page}) score:{score} ---")
                
                # Content - clean and trimmed
                content = r.document.strip()
                if len(content) > 1500:
                    content = content[:1500] + "..."
                lines.append(content)
                lines.append("")
            return "\n".join(lines)
        
        elif format_type == "markdown":
            lines = ["# Search Results\n"]
            for r in results:
                lines.append(f"## Result {r.rank} (Similarity: {r.similarity:.4f})")
                lines.append(f"\n**Source:** {r.metadata.get('source_file', 'Unknown')}\n")
                lines.append(f"```\n{r.document}\n```\n")
            return "\n".join(lines)
        
        else:  # text - IMPROVED FORMAT
            lines = [f"Found {len(results)} results:\n"]
            for r in results:
                # Determine result type
                result_type = r.metadata.get('type', 'text_chunk')
                is_image = result_type == 'extracted_image'
                
                # Header with type indicator
                type_label = "[IMAGE]" if is_image else "[TEXT]"
                lines.append(f"[{r.rank}] Score: {r.similarity:.4f} {type_label}")
                
                # Source file
                source = r.metadata.get('source_file', 'Unknown')
                lines.append(f"    Source: {source}")
                
                # Type-specific metadata
                if is_image:
                    # Image-specific info
                    image_path = r.metadata.get('image_path', 'N/A')
                    page_num = r.metadata.get('page_number', 'N/A')
                    region_type = r.metadata.get('region_type', 'N/A')
                    
                    lines.append(f"    📷 Image: {image_path}")
                    lines.append(f"    📄 Page: {page_num}")
                    lines.append(f"    🎯 Type: {region_type}")
                    
                    # Show full context for images
                    lines.append(f"    Content: {r.document}")
                else:
                    # Text chunk info
                    page_num = r.metadata.get('page_number', r.metadata.get('chunk_index', 'N/A'))
                    lines.append(f"    📄 Page/Chunk: {page_num}")
                    
                    # Show more content (no hard truncation)
                    if len(r.document) > 300:
                        lines.append(f"    Content:")
                        # Indent for readability
                        content_lines = r.document.split('\n')
                        for content_line in content_lines[:20]:
                            lines.append(f"        {content_line}")
                        if len(content_lines) > 20:
                            lines.append(f"        ... ({len(content_lines) - 20} more lines)")
                    else:
                        lines.append(f"    Content: {r.document}")
                
                lines.append("")  # Blank line between results
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
