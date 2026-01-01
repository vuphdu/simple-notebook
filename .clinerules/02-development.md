# Development & Coding Rules

## Project Structure

```
simple-notebook/
├── src/                    # Source code
│   ├── main.py             # CLI entry point
│   ├── chunking/           # Document chunking
│   ├── vectorization/      # Embeddings & vector store
│   ├── search/             # Search engine (vector + BM25)
│   ├── image_extraction/   # PDF image extraction
│   └── sequence_chart/     # Diagram processing
├── config/                 # Configuration (settings.py)
├── data/
│   ├── documents/          # Source documents
│   │   ├── input-doc/      # Place PDFs here
│   │   └── extracted_images/
│   ├── vectordb/           # ChromaDB storage
│   └── faiss_index/        # FAISS index
├── process/
│   ├── input/              # Query logs
│   └── output/             # Result logs
└── tests/                  # Unit tests
```

## Tech Stack

- **Python**: 3.10+
- **Embeddings**: Alibaba-NLP/gte-multilingual-base (768 dim)
- **Vector Store**: FAISS (default) or ChromaDB
- **Search**: Hybrid semantic + BM25

---

## Coding Standards

### Type Hints Required

```python
def search_documents(
    query: str,
    top_k: int = 5,
    use_hybrid: bool = True
) -> list[SearchResult]:
    """Search through documents."""
```

### Import Order

1. Standard library
2. Third-party packages
3. Local imports (`from src.xxx import ...`)

### Configuration

- Settings in `config/settings.py`
- Use Pydantic BaseModel
- Never hardcode paths/secrets

---

## Adding New Features

1. Create module: `src/<module>/`
2. Add `__init__.py` with exports
3. Update `src/main.py` for CLI
4. Add tests: `tests/test_<module>.py`

### Module Structure

```
src/<module>/
├── __init__.py         # from .<impl> import ...
├── <implementation>.py
└── <utils>.py
```

---

## Running & Testing

### Development

```bash
# Activate venv
.venv\Scripts\activate  # Windows

# Run CLI
python -m src.main <command>
```

### Testing

```bash
pytest tests/              # All tests
pytest tests/test_x.py -v  # Specific
pytest tests/ --cov=src    # Coverage
```

### Dependencies

```bash
pip install -r requirements.txt
```

---

## Version Control

- Format: `<type>: <description>`
- Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`
- Don't commit: `data/`, `models/`, `.venv/`
