"""
BM25 Keyword Search Module

This module provides BM25-based keyword search for hybrid RAG.
BM25 excels at exact keyword matching, complementing semantic vector search.
"""
from typing import Optional
from pathlib import Path
import math
import re
from collections import Counter

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class BM25Index:
    """
    BM25 (Best Matching 25) index for keyword-based retrieval.
    
    BM25 is a ranking function that scores documents based on:
    - Term frequency (TF): How often a term appears in a document
    - Inverse document frequency (IDF): How rare a term is across all documents
    - Document length normalization: Penalizes very long documents
    
    Formula: score = sum(IDF(term) * TF(term, doc) * (k1 + 1) / (TF + k1 * (1 - b + b * |doc|/avgdl)))
    
    Parameters:
        k1: Term frequency saturation parameter (default: 1.5)
        b: Document length normalization (default: 0.75)
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 index.
        
        Args:
            k1: Controls term frequency saturation. Higher = more weight to repeated terms.
            b: Controls document length normalization. 0 = no normalization, 1 = full normalization.
        """
        self.k1 = k1
        self.b = b
        
        # Index data
        self.documents: list[dict] = []  # [{id, content, tokens, metadata}]
        self.doc_lengths: list[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_count: int = 0
        
        # Inverted index: term -> list of (doc_idx, term_freq)
        self.inverted_index: dict[str, list[tuple[int, int]]] = {}
        
        # IDF cache
        self.idf_cache: dict[str, float] = {}
        
        # Stopwords (common words to filter out)
        self.stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'and', 'but', 'if', 'or', 'because', 'until', 'while', 'this',
            'that', 'these', 'those', 'it', 'its',
            # Vietnamese stopwords
            'và', 'là', 'của', 'có', 'được', 'trong', 'cho', 'với', 'các',
            'một', 'để', 'này', 'đó', 'từ', 'về', 'theo', 'khi', 'đã',
            'như', 'hay', 'hoặc', 'nhưng', 'tuy', 'nếu', 'thì', 'cũng',
            'bởi', 'vì', 'nên', 'mà', 'sẽ', 'đang', 'còn', 'tại', 'trên',
        }
    
    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into terms.
        
        - Lowercase
        - Split on non-alphanumeric (preserving unicode for Vietnamese)
        - Filter stopwords
        - Filter very short tokens
        """
        # Lowercase
        text = text.lower()
        
        # Split on non-word characters (keep Vietnamese diacritics)
        tokens = re.findall(r'[\w]+', text, re.UNICODE)
        
        # Filter
        tokens = [
            t for t in tokens 
            if len(t) > 1 and t not in self.stopwords and not t.isdigit()
        ]
        
        return tokens
    
    def add_documents(
        self, 
        documents: list[dict],
        id_field: str = "chunk_id",
        content_field: str = "content"
    ):
        """
        Add documents to the BM25 index.
        
        Args:
            documents: List of document dicts with id and content.
            id_field: Field name for document ID.
            content_field: Field name for document content.
        """
        for doc in documents:
            doc_id = doc.get(id_field, str(len(self.documents)))
            content = doc.get(content_field, "")
            metadata = doc.get("metadata", {})
            
            # Tokenize
            tokens = self.tokenize(content)
            
            if not tokens:
                continue
            
            # Store document
            doc_idx = len(self.documents)
            self.documents.append({
                "id": doc_id,
                "content": content,
                "tokens": tokens,
                "metadata": metadata
            })
            
            self.doc_lengths.append(len(tokens))
            
            # Update inverted index
            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((doc_idx, count))
        
        # Update stats
        self.doc_count = len(self.documents)
        if self.doc_count > 0:
            self.avg_doc_length = sum(self.doc_lengths) / self.doc_count
        
        # Clear IDF cache (needs recalculation)
        self.idf_cache.clear()
    
    def _idf(self, term: str) -> float:
        """Calculate IDF for a term."""
        if term in self.idf_cache:
            return self.idf_cache[term]
        
        # Number of documents containing the term
        doc_freq = len(self.inverted_index.get(term, []))
        
        if doc_freq == 0:
            idf = 0.0
        else:
            # BM25 IDF formula (with smoothing)
            idf = math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        
        self.idf_cache[term] = idf
        return idf
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> list[dict]:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query string.
            top_k: Number of top results to return.
            threshold: Minimum score threshold.
            
        Returns:
            List of results with id, content, score, metadata.
        """
        if self.doc_count == 0:
            return []
        
        # Tokenize query
        query_tokens = self.tokenize(query)
        
        if not query_tokens:
            return []
        
        # Calculate scores for each document
        scores: dict[int, float] = {}
        
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            
            idf = self._idf(term)
            
            for doc_idx, term_freq in self.inverted_index[term]:
                doc_len = self.doc_lengths[doc_idx]
                
                # BM25 score component for this term
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                score = idf * (numerator / denominator)
                
                if doc_idx not in scores:
                    scores[doc_idx] = 0.0
                scores[doc_idx] += score
        
        # Rank by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Apply threshold and top_k
        results = []
        for doc_idx, score in ranked[:top_k]:
            if score < threshold:
                continue
            
            doc = self.documents[doc_idx]
            results.append({
                "id": doc["id"],
                "content": doc["content"],
                "score": score,
                "metadata": doc["metadata"],
                "rank": len(results) + 1
            })
        
        return results
    
    def clear(self):
        """Clear the index."""
        self.documents.clear()
        self.doc_lengths.clear()
        self.inverted_index.clear()
        self.idf_cache.clear()
        self.doc_count = 0
        self.avg_doc_length = 0.0
    
    def count(self) -> int:
        """Return number of indexed documents."""
        return self.doc_count
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "doc_count": self.doc_count,
            "avg_doc_length": round(self.avg_doc_length, 2),
            "vocabulary_size": len(self.inverted_index),
            "k1": self.k1,
            "b": self.b
        }


# Global BM25 index instance
_bm25_index: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    """Get or create the global BM25 index instance."""
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
    return _bm25_index
