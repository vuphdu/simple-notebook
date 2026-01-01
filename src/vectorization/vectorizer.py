"""
Text Vectorization Module

This module provides functionality to convert text chunks into vector embeddings
using the Alibaba-NLP/gte-multilingual-base model.
"""
from typing import Optional, Union
from pathlib import Path
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings, ModelConfig, ensure_directories


class TextVectorizer:
    """
    Handles text vectorization using sentence-transformers.
    
    Uses Alibaba-NLP/gte-multilingual-base model for generating embeddings.
    Model is downloaded on first use and cached for subsequent runs.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the TextVectorizer.
        
        Args:
            config: Model configuration. Uses default settings if not provided.
        """
        self.config = config or settings.model
        self._model = None
        ensure_directories()
    
    @property
    def model(self):
        """Lazy loading of the embedding model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model
    
    def _load_model(self):
        """
        Load the embedding model.
        
        Downloads the model on first run, uses cache on subsequent runs.
        
        Returns:
            SentenceTransformer model instance.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
        
        print(f"Loading model: {self.config.model_name}")
        print(f"Cache directory: {self.config.model_cache_dir}")
        
        # Ensure cache directory exists
        self.config.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model with cache
        # trust_remote_code=True is required for Alibaba-NLP/gte-multilingual-base
        model = SentenceTransformer(
            self.config.model_name,
            cache_folder=str(self.config.model_cache_dir),
            device=self.config.device,
            trust_remote_code=True
        )
        
        # Set max sequence length
        model.max_seq_length = self.config.max_seq_length
        
        print(f"Model loaded successfully. Embedding dimension: {model.get_sentence_embedding_dimension()}")
        
        return model
    
    def vectorize(
        self,
        texts: Union[str, list[str]],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Convert text(s) to vector embeddings.
        
        Args:
            texts: Single text or list of texts to vectorize.
            batch_size: Batch size for processing.
            show_progress: Whether to show progress bar.
            
        Returns:
            NumPy array of embeddings. Shape: (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def vectorize_chunks(
        self,
        chunks: list,
        batch_size: int = 32
    ) -> list[dict]:
        """
        Vectorize document chunks with OlmOCR-style metadata.
        
        Args:
            chunks: List of DocumentChunk objects.
            batch_size: Batch size for processing.
            
        Returns:
            List of dicts with chunk info, embeddings, and classification.
        """
        if not chunks:
            return []
        
        # Extract text content from chunks
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        print(f"Vectorizing {len(texts)} chunks...")
        embeddings = self.vectorize(texts, batch_size=batch_size)
        
        # Combine chunks with embeddings and OlmOCR metadata
        results = []
        for chunk, embedding in zip(chunks, embeddings):
            # Get OlmOCR classification fields if available
            chunk_type = getattr(chunk, 'chunk_type', 'text')
            has_equations = getattr(chunk, 'has_equations', False)
            has_code_blocks = getattr(chunk, 'has_code_blocks', False)
            
            # Merge classification into metadata
            enhanced_metadata = {
                **chunk.metadata,
                "chunk_type": chunk_type,
                "has_equations": has_equations,
                "has_code_blocks": has_code_blocks,
            }
            
            results.append({
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": enhanced_metadata,
                "source_file": chunk.source_file,
                "embedding": embedding.tolist(),
                # Top-level classification for easy access
                "chunk_type": chunk_type,
                "has_equations": has_equations,
                "has_code_blocks": has_code_blocks,
            })
        
        return results
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model."""
        return self.model.get_sentence_embedding_dimension()
    
    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.
            
        Returns:
            Cosine similarity score (0-1 for normalized embeddings).
        """
        if self.config.normalize_embeddings:
            # For normalized vectors, cosine similarity is just dot product
            return float(np.dot(embedding1, embedding2))
        else:
            # Calculate cosine similarity
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


# Global vectorizer instance (lazy loaded)
_vectorizer: Optional[TextVectorizer] = None


def get_vectorizer() -> TextVectorizer:
    """Get or create the global vectorizer instance."""
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = TextVectorizer()
    return _vectorizer


def vectorize_text(
    texts: Union[str, list[str]],
    config: Optional[ModelConfig] = None
) -> np.ndarray:
    """
    Convenience function to vectorize text.
    
    Args:
        texts: Text or list of texts to vectorize.
        config: Optional model configuration.
        
    Returns:
        NumPy array of embeddings.
    """
    vectorizer = TextVectorizer(config) if config else get_vectorizer()
    return vectorizer.vectorize(texts)
