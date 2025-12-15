# Simple RAG System

A simple Retrieval-Augmented Generation (RAG) system with chunking, vectorization, and search capabilities.

## Features

- **Document Chunking**: Split documents into manageable chunks for processing
- **Vectorization**: Convert text chunks to vectors using Alibaba-NLP/gte-multilingual-base model
- **Sequence Chart Processing**: Extract and process sequence diagrams with image export and description vectorization
- **Semantic Search**: Search through vectorized documents with similarity matching
- **Dual Backend Support**: Choose between FAISS (fast) or ChromaDB (full-featured)
- **Image Extraction**: Extract images/figures from documents using PaddleOCR PP-Structure

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
│   └── image_extraction/       # Image extraction with PP-Structure
├── config/                     # Configuration files
└── tests/                      # Unit tests
```

## Installation

```bash
# Install base requirements
pip install -r requirements.txt

# For PaddleOCR image extraction, also install:
pip install paddlepaddle  # CPU version
# OR for GPU:
# pip install paddlepaddle-gpu
```

## Quick Start

### 1. Initialize folders

```bash
python -m src.main init
```

### 2. Process documents

```bash
python -m src.main process --input data/documents
```

### 3. Search

```bash
python -m src.main search "authentication flow"
```

Input/output sẽ được lưu trong `/process/input/` và `/process/output/`

### 4. Extract images from documents

```bash
# Extract images and vectorize descriptions
python -m src.main extract-images --input data/documents

# Extract images only (no vectorization)
python -m src.main extract-images --input data/documents/file.pdf --no-vectorize
```

Extracted images will be saved in `data/documents/extracted_images/`

### 5. View statistics

```bash
python -m src.main stats
```

### 6. Clean extracted data

```bash
# Clean vectordb and process logs (with confirmation)
python -m src.main clean

# Clean without confirmation
python -m src.main clean -y

# Clean everything including downloaded models
python -m src.main clean --all -y
```

## Model

This system uses **Alibaba-NLP/gte-multilingual-base** for text embedding:

- First run downloads the model to `/models/`
- Subsequent runs use the cached model

## Image Extraction with PP-Structure

The system uses PaddleOCR's PP-Structure for:

- **Layout Analysis**: Detect image/figure regions in documents
- **Region Cropping**: Extract and save image regions
- **Context Extraction**: Capture surrounding text as descriptions
- **Vectorization**: Store image descriptions for semantic search

Supported document types:

- PDF files (using PyMuPDF or pdf2image)
- Image files (PNG, JPG, JPEG, BMP, TIFF)

## Vector Store Backends

The system supports two vector store backends:

### FAISS (Default - Recommended for Speed)

```python
# In config/settings.py
backend: str = "faiss"  # Fast similarity search
faiss_index_type: str = "Flat"  # Options: Flat, IVF, HNSW
```

### ChromaDB (Alternative - Full-featured)

```python
# In config/settings.py
backend: str = "chromadb"  # Persistent, with metadata filtering
```

**FAISS Index Types:**
| Type | Use Case |
|------|----------|
| `Flat` | Exact search, best for <100k vectors |
| `IVF` | Approximate search, good for large datasets |
| `HNSW` | Graph-based, balanced speed/accuracy |

## Main Features

🔧 Tính năng chính

| Module           | Mô tả                                                         |
| ---------------- | ------------------------------------------------------------- |
| Chunking         | Chia tài liệu (TXT, MD, PDF, DOCX) thành chunks nhỏ           |
| Vectorization    | Chuyển text → vector với Alibaba-NLP/gte-multilingual-base    |
| Vector Store     | Lưu trữ với **FAISS** (default) hoặc ChromaDB                 |
| Search           | Tìm kiếm semantic với cosine similarity                       |
| Sequence Chart   | Parse Mermaid diagrams → export image + vectorize description |
| Image Extraction | Trích xuất hình ảnh từ tài liệu với PP-Structure              |

## License

MIT License
