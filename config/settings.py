"""
Configuration settings for the RAG system.
"""
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional


def detect_device(force_cpu: bool = False) -> str:
    """
    Auto-detect the best available device for model inference.
    
    Args:
        force_cpu: If True, always return "cpu" regardless of GPU availability.
    
    Returns:
        "cuda" if GPU is available and not forced to CPU, otherwise "cpu".
    """
    if force_cpu:
        print("💻 Device: CPU (forced)")
        return "cpu"
    
    try:
        import torch
        if torch.cuda.is_available():
            # Log GPU info for debugging
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🚀 Device: GPU - {gpu_name} ({gpu_memory:.1f} GB)")
            return "cuda"
        else:
            print("💻 Device: CPU (no CUDA GPU detected)")
    except ImportError:
        print("💻 Device: CPU (torch not available for detection)")
    
    return "cpu"


# Auto-detect device at module load
DEFAULT_DEVICE = detect_device()


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
    device: str = DEFAULT_DEVICE  # Auto-detect: "cuda" if GPU available, else "cpu"
    normalize_embeddings: bool = True
    force_cpu: bool = False  # Set to True to force CPU even if GPU is available


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""
    chunk_size: int = 1024
    chunk_overlap: int = 100
    separators: list[str] = Field(default=["\n\n## ", "\n\n### ", "\n\n#### ", "\n\n", "\n", ". ", " ", ""])
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
