"""Vectorization module."""
from .vectorizer import TextVectorizer, get_vectorizer, vectorize_text
from .vector_store import VectorStore, get_vector_store as get_chroma_store
from .faiss_store import FaissVectorStore, get_faiss_store

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings


def get_vector_store():
    """
    Get the vector store based on configured backend.
    
    Returns:
        VectorStore or FaissVectorStore based on settings.vectordb.backend
    """
    backend = settings.vectordb.backend.lower()
    
    if backend == "faiss":
        return get_faiss_store()
    elif backend == "chromadb":
        return get_chroma_store()
    else:
        print(f"Unknown backend '{backend}', defaulting to FAISS")
        return get_faiss_store()


__all__ = [
    "TextVectorizer",
    "get_vectorizer",
    "vectorize_text",
    "VectorStore",
    "FaissVectorStore",
    "get_vector_store",
    "get_chroma_store",
    "get_faiss_store",
]
