# Project Workflows

This document outlines the core workflows and data pipelines in the Simple RAG System.

## 1. Data Ingestion Pipeline

This pipeline processes raw documents into searchable vector embeddings.

```mermaid
graph TD
    A[Raw Documents] -->|data/documents/input-doc| B(Document Processor)
    B --> C{File Type?}
    C -->|PDF/TXT/MD| D[Chunking Module]
    C -->|Mermaid| E[Sequence Chart Processor]

    subgraph Text Processing
    D -->|Split Text| F[Text Chunks]
    F --> G[Vectorizer]
    G -->|gte-multilingual-base| H[Vector Embeddings]
    end

    subgraph Chart Processing
    E -->|Render| I[PNG Image]
    E -->|Extract Description| J[Chart Text]
    J --> G
    end

    H --> K[(Vector Store)]
    K -->|Option A| L[FAISS Index]
    K -->|Option B| M[ChromaDB]
```

## 2. Image Extraction Pipeline

This pipeline extracts figures and diagrams from PDF documents.

```mermaid
graph TD
    A[PDF Document] --> B(Image Extractor)

    subgraph Extraction Strategy
    B --> C[Bitmap Extraction]
    B --> D[Vector Drawing Extraction]
    end

    C -->|Raw Images| E{Validation Filter}
    D -->|Rendered Regions| E

    E -->|Too Dark/Uniform| F[Discard]
    E -->|Valid Content| G[Save to Disk]

    G -->|data/documents/extracted_images| H[PNG Files]

    subgraph Context Indexing
    B -->|Extract Surrounding Text| I[Image Context]
    I --> J[Vectorizer]
    J --> K[(Vector Store)]
    end
```

## 3. Search Workflow

How the system retrieves information based on user queries.

```mermaid
graph LR
    A[User Query] --> B(Search Engine)
    B --> C[Query Vectorizer]
    C -->|Generate Embedding| D[Query Vector]

    D --> E[(Vector Store)]
    E -->|Cosine Similarity| F[Nearest Neighbors]

    F --> G[Result Aggregation]
    G --> H[Final Output]

    style H fill:#f9f,stroke:#333,stroke-width:4px
```

## 4. Operational Commands

### Initialization

```bash
python -m src.main init
# Creates necessary directory structure
```

### Processing

```bash
python -m src.main process --input data/documents
# Runs Ingestion Pipeline
```

### Image Extraction

```bash
python -m src.main extract-images
# Runs Image Extraction Pipeline
```

### Search

```bash
python -m src.main search "your query here"
# Runs Search Workflow
```
