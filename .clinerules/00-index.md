# Cline Rules Index

This folder contains rules and guidelines for the Simple RAG System.

## Quick Start for AI Assistants

**Primary command to retrieve context:**

```bash
python -m src.main query "your search query" -k 5
```

**Check database status:**

```bash
python -m src.main stats
```

## File Organization

| File                 | Purpose                          | Read When            |
| -------------------- | -------------------------------- | -------------------- |
| `00-index.md`        | This index file                  | Always (entry point) |
| `01-rag-commands.md` | CLI commands & RAG workflow      | Using CLI            |
| `02-development.md`  | Coding rules & project structure | Writing code         |
| `03-prompting.md`    | Image/PDF understanding prompts  | Processing images    |

## Critical Rules

1. **Always use `query` command** (not `search`) for AI-assisted retrieval
2. **Query RAG before implementing** new features or debugging
3. **Use `--format json`** when you need to parse results
4. **Follow Python 3.10+ type hints** when writing code

## Project Summary

- **Type**: RAG (Retrieval-Augmented Generation) system
- **CLI Entry**: `python -m src.main <command>`
- **Languages**: Python 3.10+
- **Search**: Hybrid (semantic + BM25 keyword matching)
