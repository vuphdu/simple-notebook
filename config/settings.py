"""
Configuration settings for the RAG system.
"""
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional


# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
INPUT_DOCS_DIR = DOCUMENTS_DIR / "input-doc"  # Input documents location
EXTRACTED_IMAGES_DIR = DOCUMENTS_DIR / "extracted_images"  # Extracted images output
VECTORDB_DIR = DATA_DIR / "vectordb"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
MODELS_DIR = BASE_DIR / "models"
PROCESS_DIR = BASE_DIR / "process"
PROCESS_INPUT_DIR = PROCESS_DIR / "input"
PROCESS_OUTPUT_DIR = PROCESS_DIR / "output"


class ModelConfig(BaseModel):
    """Configuration for the embedding model."""
    model_name: str = "Alibaba-NLP/gte-multilingual-base"
    model_cache_dir: Path = MODELS_DIR
    max_seq_length: int = 8192
    device: str = "cpu"  # or "cuda" if GPU available
    normalize_embeddings: bool = True


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    separators: list[str] = Field(default=["\n\n", "\n", ". ", " ", ""])
    length_function: str = "len"  # or "tiktoken" for token-based


class VectorDBConfig(BaseModel):
    """Configuration for vector database."""
    backend: str = "faiss"  # "chromadb" or "faiss"
    collection_name: str = "documents"
    persist_directory: Path = VECTORDB_DIR
    faiss_index_dir: Path = FAISS_INDEX_DIR
    distance_metric: str = "cosine"  # cosine, l2, ip
    # FAISS specific options
    faiss_index_type: str = "Flat"  # Flat, IVF, HNSW


class SearchConfig(BaseModel):
    """Configuration for search."""
    top_k: int = 5
    score_threshold: float = 0.5
    include_metadata: bool = True


class SequenceChartConfig(BaseModel):
    """Configuration for sequence chart processing."""
    export_format: str = "png"  # png, svg, pdf
    export_quality: int = 150  # DPI
    include_description: bool = True


class Settings(BaseModel):
    """Main settings class combining all configurations."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    vectordb: VectorDBConfig = Field(default_factory=VectorDBConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    sequence_chart: SequenceChartConfig = Field(default_factory=SequenceChartConfig)


# Global settings instance
settings = Settings()


def ensure_directories():
    """Create all required directories if they don't exist."""
    directories = [
        DOCUMENTS_DIR,
        INPUT_DOCS_DIR,
        EXTRACTED_IMAGES_DIR,
        VECTORDB_DIR,
        FAISS_INDEX_DIR,
        MODELS_DIR,
        PROCESS_INPUT_DIR,
        PROCESS_OUTPUT_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
