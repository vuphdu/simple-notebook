# Simple RAG System

A simple Retrieval-Augmented Generation (RAG) system with chunking, vectorization, and search capabilities.

## Features

- **Document Chunking**: Split documents into manageable chunks for processing
- **Vectorization**: Convert text chunks to vectors using Alibaba-NLP/gte-multilingual-base model
- **Sequence Chart Processing**: Extract and process sequence diagrams with image export and description vectorization
- **Semantic Search**: Search through vectorized documents with similarity matching

## Directory Structure

```
simple-notebook/
├── data/
│   ├── documents/          # Source documents for processing
│   └── vectordb/           # Vector database storage
├── models/                 # Downloaded ML models cache
├── process/
│   ├── input/              # Search queries input
│   └── output/             # Search results output
├── src/
│   ├── chunking/           # Document chunking module
│   ├── vectorization/      # Text vectorization module
│   ├── search/             # Search and retrieval module
│   └── sequence_chart/     # Sequence chart processing
├── config/                 # Configuration files
└── tests/                  # Unit tests
```

## Installation

```bash
pip install -r requirements.txt
```

## Init folders
```bash
python -m src.main init
```

## Xử lý và vector hóa tài liệu:
```bash
python -m src.main process --input data/documents
```

## Tìm kiếm:
```bash
python -m src.main search "authentication flow"
```Input/output sẽ được lưu trong /process/input/ và /process/output/

## Usage

### 1. Process Documents
```bash
python -m src.main process --input data/documents --output data/vectordb
```

### 2. Search
```bash
python -m src.main search --query "your search query"
```

### 3. Process Sequence Charts
```bash
python -m src.main chart --input path/to/chart.md --export-image
```

## Model

This system uses **Alibaba-NLP/gte-multilingual-base** for text embedding:
- First run downloads the model to `/models/`
- Subsequent runs use the cached model

## Main function
🔧 Tính năng chính
Module	Mô tả
Chunking	Chia tài liệu (TXT, MD, PDF, DOCX) thành chunks nhỏ
Vectorization	Chuyển text → vector với Alibaba-NLP/gte-multilingual-base
Vector Store	Lưu trữ với ChromaDB (persistent)
Search	Tìm kiếm semantic với cosine similarity
Sequence Chart	Parse Mermaid diagrams → export image + vectorize description


## License

MIT License
