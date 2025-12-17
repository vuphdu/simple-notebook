# Simple RAG System

A simple Retrieval-Augmented Generation (RAG) system with chunking, vectorization, and search capabilities.

## Features

- **Document Chunking**: Split documents into manageable chunks for processing
- **Vectorization**: Convert text chunks to vectors using Alibaba-NLP/gte-multilingual-base model
- **Sequence Chart Processing**: Extract and process sequence diagrams with image export and description vectorization
- **Semantic Search**: Search through vectorized documents with similarity matching
- **Dual Backend Support**: Choose between FAISS (fast) or ChromaDB (full-featured)
- **Image Extraction**: Extract images and vector drawings from documents using PyMuPDF (fitz) or PaddleOCR
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
│   └── image_extraction/       # Image extraction with PyMuPDF & PaddleOCR
├── config/                     # Configuration files
└── tests/                      # Unit tests
```

## Installation

```bash
# Install base requirements
pip install -r requirements.txt
```

## Quick Start

### 1. Initialize folders

```bash
python -m src.main init
```

### 2. Process documents

You can process documents using the default PyMuPDF extractor or the advanced PaddleOCR extractor.

**Default (PyMuPDF - Fast, Smart Cropping):**

```bash
python -m src.main process --input data/documents
```

**Advanced (PaddleOCR - AI Layout Analysis):**
_Requires `paddlepaddle` and `paddleocr` installed._

```bash
python -m src.main process --input data/documents --mode paddle
```

### 3. Search

```bash
python -m src.main search "authentication flow"
```

Input/output will be saved in `/process/input/` and `/process/output/`.

### 4. Update Context (New)

Add manual notes or context to a specific document or image to make it easier to find.

```bash
# Search for a document and add context
python -m src.main update "page 100" "This diagram shows the login process"
```

### 5. Extract images only

```bash
# Extract images and vectorize descriptions (PyMuPDF only)
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

## Model

This system uses **Alibaba-NLP/gte-multilingual-base** for text embedding:

- First run downloads the model to `/models/`
- Subsequent runs use the cached model

## Image Extraction

The system supports two modes for image extraction:

### 1. PyMuPDF (Default)

- **Fast & Lightweight**: Uses direct PDF parsing.
- **Smart Cropping**: Automatically detects and crops figures and vector drawings.
- **Best for**: Vector graphics, embedded images, and clean PDFs.

### 2. PaddleOCR (Advanced)

- **AI-Powered**: Uses PP-Structure for layout analysis.
- **Subprocess Execution**: Runs in a separate process to avoid DLL conflicts with PyTorch.
- **Best for**: Complex layouts, scanned documents, and identifying regions like tables and figures.
- **Usage**: Add `--mode paddle` to the process command.

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

## License

MIT License
