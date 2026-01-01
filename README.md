# Simple RAG System

A lean and efficient Retrieval-Augmented Generation (RAG) system with smart image extraction, chunking, vectorization, and semantic search capabilities.

## Features

- **Document Chunking**: Split documents into manageable chunks for processing
- **Vectorization**: Convert text chunks to vectors using Alibaba-NLP/gte-multilingual-base model
- **Hybrid Search**: Combine semantic vector search with BM25 keyword matching for better results
- **Sequence Chart Processing**: Extract and process sequence diagrams with image export and description vectorization
- **Semantic Search**: Search through vectorized documents with similarity matching
- **Dual Backend Support**: Choose between FAISS (fast, default) or ChromaDB (full-featured)
- **Smart Image Extraction**: Extract images and vector drawings from PDFs using PyMuPDF intelligent clustering
- **Context Updates**: Manually add context/notes to specific documents or images to improve searchability

## Directory Structure

```
simple-notebook/
├── data/
│   ├── documents/
│   │   ├── input-doc/          # Source documents for processing
│   │   ├── extracted_images/   # Extracted image regions
│   ├── vectordb/               # ChromaDB storage
│   └── faiss_index/            # FAISS index storage
├── models/                     # Downloaded ML models cache
├── process/
│   ├── input/                  # Search queries input
│   └── output/                 # Search results output
├── src/
│   ├── chunking/               # Document chunking module
│   ├── vectorization/          # Text vectorization module
│   ├── search/                 # Search and retrieval module
│   ├── sequence_chart/         # Sequence chart processing
│   └── image_extraction/       # Smart image extraction with PyMuPDF
├── config/                     # Configuration files
└── tests/                      # Unit tests
```

## Installation

```bash
# Install requirements
pip install -r requirements.txt
```

## Quick Start

### 1. Initialize folders

```bash
python -m src.main init
```

### 2. Process documents

Process documents with smart image extraction (automatically enabled):

```bash
python -m src.main process
# OR specify input directory
python -m src.main process --input data/documents
```

This will:

- Extract text from PDFs, DOCX, TXT, and MD files
- Smart crop images and diagrams from PDFs
- Process sequence charts
- Vectorize all content
- Store in vector database

### 3. Search

```bash
# Default: Hybrid search (vector + BM25 keyword matching)
python -m src.main search "authentication flow"

# Pure vector search (semantic only, disable BM25)
python -m src.main search "authentication flow" --no-hybrid

# Explicitly enable hybrid search
python -m src.main search "WiFi timeout error" --hybrid
```

**Hybrid Search** combines semantic vector similarity with BM25 keyword matching for better results, especially when exact terms matter.

Input/output will be saved in `/process/input/` and `/process/output/`.

### 4. Update Context

Add manual notes or context to a specific document or image to make it easier to find.

```bash
# Search for a document and add context
python -m src.main update "page 100" "This diagram shows the login process"
```

### 5. Extract images only

```bash
# Extract images and vectorize descriptions
python -m src.main extract-images --input data/documents

# Extract images only (no vectorization)
python -m src.main extract-images --input data/documents/file.pdf --no-vectorize
```

Extracted images will be saved in `data/documents/extracted_images/`.

### 6. View statistics

```bash
python -m src.main stats
```

### 7. Clean extracted data

```bash
# Clean vectordb and process logs (with confirmation)
python -m src.main clean

# Clean without confirmation
python -m src.main clean -y

# Clean everything including downloaded models
python -m src.main clean --all -y
```

### 8. Index source code (for AI assistants)

Index large codebases for RAG retrieval by AI coding assistants:

```bash
# Index a code directory with auto-detected tag
python -m src.main index-code /path/to/wpa_supplicant/

# Index with custom project tag
python -m src.main index-code /path/to/code --tag wifi_driver

# Index specific file types only
python -m src.main index-code /path/to/code --ext .c .h
```

Supported file types: C/C++, Python, Java, JavaScript/TypeScript, Go, Rust, Shell, and config files.

### 9. Quick query (for AI assistants)

Optimized search command for AI coding assistants with compact output:

```bash
# Quick search with compact output (no logging)
python -m src.main query "wifi connection handling" -k 10

# Output as JSON
python -m src.main query "authentication flow" --format json
```

## Model

This system uses **Alibaba-NLP/gte-multilingual-base** for text embedding:

- First run downloads the model to `/models/`
- Subsequent runs use the cached model
- Embedding dimension: 768
- Supports 70+ languages

## Smart Image Extraction

The system uses **PyMuPDF** for intelligent image extraction:

### Features

- **Fast & Lightweight**: Direct PDF parsing with no external dependencies
- **Smart Cropping**: Automatically detects and extracts figures, diagrams, and vector drawings
- **Vector Drawing Clustering**: Groups nearby vector paths into coherent diagrams
- **Context Extraction**: Captures surrounding text for better searchability
- **Embedded Images**: Extracts all bitmap images from PDFs

### How It Works

1. **Detects embedded images**: Finds all bitmap images in the PDF
2. **Clusters vector drawings**: Groups nearby vector paths (lines, shapes) into diagrams
3. **Merges nearby regions**: Combines overlapping or adjacent elements
4. **Extracts context**: Captures surrounding text as metadata
5. **Saves as PNG**: All extracted regions saved in `extracted_images/`

## Vector Store Backends

The system supports two vector store backends:

### FAISS (Default - Recommended for Speed)

```python
# In config/settings.py
backend: str = "faiss"  # Fast similarity search
faiss_index_type: str = "Flat"  # Options: Flat, IVF, HNSW
```

**Benefits:**

- Extremely fast similarity search
- Low memory footprint
- Good for local/single-machine deployments

### ChromaDB (Alternative - Full-featured)

```python
# In config/settings.py
backend: str = "faiss"  # Persistent, with metadata filtering
```

**Benefits:**

- Rich metadata filtering
- Built-in persistence
- Easy to inspect and debug

## Hybrid Search

The system uses **Hybrid Search** by default, combining two approaches for better results:

### How It Works

```
Query: "WiFi authentication timeout"
                    │
    ┌───────────────┴───────────────┐
    ▼                               ▼
┌─────────────┐               ┌─────────────┐
│ Vector      │               │ BM25        │
│ Search      │               │ Search      │
│ (Semantic)  │               │ (Keywords)  │
└─────┬───────┘               └─────┬───────┘
      │                             │
      └─────────────┬───────────────┘
                    ▼
            ┌───────────────┐
            │ Hybrid Score  │
            │ = α × vector  │
            │ + (1-α) × bm25│
            └───────────────┘
```

- **Vector Search**: Finds semantically similar content (e.g., "wireless auth" matches "WiFi authentication")
- **BM25 Search**: Finds exact keyword matches (e.g., "timeout" → documents containing "timeout")
- **Combined**: Results ranked by weighted combination of both scores

### Configuration

```python
# In config/settings.py
class SearchConfig(BaseModel):
    use_hybrid: bool = True       # Enable hybrid search
    hybrid_alpha: float = 0.7     # 70% vector + 30% BM25
    bm25_k1: float = 1.5          # BM25 term frequency saturation
    bm25_b: float = 0.75          # BM25 length normalization
```

### Tuning `hybrid_alpha`

| Value | Behavior                     | Best For                           |
| ----- | ---------------------------- | ---------------------------------- |
| `1.0` | Pure semantic search         | Conceptual queries                 |
| `0.7` | **Default** - semantic-heavy | General use                        |
| `0.5` | Equal weight                 | Balanced                           |
| `0.3` | Keyword-heavy                | Technical docs with specific terms |

### CLI Usage

```bash
# Hybrid search (default)
python -m src.main search "WiFi timeout error"

# Disable hybrid (pure vector)
python -m src.main search "authentication flow" --no-hybrid

# Force hybrid on
python -m src.main search "error code 0x8007" --hybrid
```

## CLI Commands

```bash
# Process documents
python -m src.main process [--input DIR] [--no-charts]

# Search (hybrid by default)
python -m src.main search "query" [--top-k K] [--no-save] [--hybrid|--no-hybrid]

# Quick query for AI (compact output)
python -m src.main query "query" [--top-k K] [--hybrid|--no-hybrid]

# Extract images
python -m src.main extract-images [--input PATH] [--no-vectorize]

# Update context
python -m src.main update "query" "new context" [--yes]

# View stats
python -m src.main stats

# Clean data
python -m src.main clean [--all] [--yes]

# Initialize
python -m src.main init
```

## Dependencies

### Core (17 packages)

- **Deep Learning**: torch, transformers, sentence-transformers
- **Vector DB**: chromadb, faiss-cpu
- **Document Processing**: langchain-text-splitters, python-docx, PyPDF2, pymupdf
- **Image Processing**: Pillow
- **Utilities**: pydantic, tqdm, numpy
- **Testing**: pytest, pytest-asyncio

All lean and actively used!

## License

MIT License
