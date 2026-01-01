https://github.com/allenai/olmocr# Simple RAG System - Cline Rules

## Overview

This is a Retrieval-Augmented Generation (RAG) system that provides document processing, vectorization, and semantic search capabilities. The CLI is the primary interface for interacting with the system.

## CLI Entry Point

All commands are executed via Python module:

```bash
python -m src.main <command> [options]
```

## Available Commands

### 1. Query (Recommended for AI Assistants)

**Purpose**: Quick search with compact output, optimized for AI coding assistants.

```bash
# Basic query
python -m src.main query "your search query" -k 5

# With more results
python -m src.main query "WiFi authentication" -k 10

# JSON output format
python -m src.main query "error handling" --format json

# Disable hybrid search (pure vector search)
python -m src.main query "authentication flow" --no-hybrid
```

**When to use**:

- When you need to find relevant code, documentation, or diagrams
- When you need context about a specific topic from the indexed documents
- When searching for technical specifications or implementations

**Parameters**:

- `-k, --top-k`: Number of results (default: 5, max recommended: 20)
- `--format`: Output format: "text" or "json"
- `--hybrid/--no-hybrid`: Enable/disable BM25 keyword matching

### 2. Search (Full search with logging)

**Purpose**: Detailed search with results saved to process/output/.

```bash
python -m src.main search "query text" --top-k 5
python -m src.main search "WiFi timeout" --hybrid
python -m src.main search "authentication" --no-hybrid --no-save
```

**When to use**:

- When you need to save search results for later reference
- When you need detailed output formatting

### 3. Process Documents

**Purpose**: Index new documents into the RAG system.

```bash
# Process all documents in data/documents/
python -m src.main process

# Process specific directory
python -m src.main process --input /path/to/docs

# Skip sequence chart processing
python -m src.main process --no-charts
```

**When to use**:

- When new documents are added to the system
- When re-indexing is needed

### 4. Index Source Code

**Purpose**: Index codebases for AI-assisted retrieval.

```bash
# Auto-detect project tag from folder name
python -m src.main index-code /path/to/codebase

# With custom project tag
python -m src.main index-code /path/to/code --tag wifi_driver

# Specific file types
python -m src.main index-code /path/to/code --ext .c .h
```

**When to use**:

- When you need to index a large codebase for RAG search
- Supported: C/C++, Python, Java, JavaScript/TypeScript, Go, Rust, Shell

### 5. Extract Images

**Purpose**: Extract images and diagrams from PDFs.

```bash
python -m src.main extract-images --input /path/to/file.pdf
python -m src.main extract-images --input /path/to/docs --no-vectorize
```

### 6. Update Context

**Purpose**: Add notes/tags to specific documents or images.

```bash
python -m src.main update "search query" "new context to add" --yes
```

### 7. Statistics

**Purpose**: View database statistics.

```bash
python -m src.main stats
```

### 8. Clean

**Purpose**: Reset/clean the vector database.

```bash
python -m src.main clean -y
python -m src.main clean --all -y  # Also removes models
```

## Best Practices for Cline

### When to Query the RAG System

1. **Finding relevant documentation**: Before implementing a feature, query for existing patterns
2. **Understanding architecture**: Query for "architecture", "flow", or "diagram"
3. **Looking for examples**: Query for specific function names or concepts
4. **Debugging**: Query for error codes or error messages

### Query Tips

1. **Be specific**: Use exact terms when known (e.g., "wpa_supplicant authentication")
2. **Use hybrid search**: Default enabled, good for exact keyword matching
3. **Use pure vector**: Add `--no-hybrid` for conceptual/semantic queries
4. **Limit results**: Use `-k 5` or `-k 10` to avoid overwhelming output

### Example Workflow

```bash
# 1. First, check what's indexed
python -m src.main stats

# 2. Query for relevant context
python -m src.main query "WiFi connection state machine" -k 5

# 3. If you need more detail, increase k
python -m src.main query "WiFi connection state machine" -k 10

# 4. For exact term matching
python -m src.main query "WPA_COMPLETED" --hybrid -k 5
```

## Output Interpretation

Query results include:

- **Source**: File path and page number (for PDFs)
- **Type**: "text", "image", "code", or "chart"
- **Content**: The matched content snippet
- **Metadata**: Additional context like tags or surrounding text

## Directory Structure

```
data/
├── documents/
│   ├── input-doc/          # Place source documents here
│   └── extracted_images/   # Extracted images
├── vectordb/               # Vector database storage
└── faiss_index/            # FAISS index storage

process/
├── input/                  # Search query logs
└── output/                 # Search result logs
```

## Notes for AI Assistants

- **Always use `query` command** for RAG retrieval (not `search`)
- **Parse JSON output** when `--format json` is used for structured data
- **Check stats first** if unsure about indexed content
- **The system supports Vietnamese and English** queries (multilingual model)
