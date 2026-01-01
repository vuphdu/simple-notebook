# RAG Commands & Workflow

## CLI Entry Point

```bash
python -m src.main <command> [options]
```

## Command Quick Reference

| Command          | Purpose                     | Example                                              |
| ---------------- | --------------------------- | ---------------------------------------------------- |
| `query`          | **AI search (recommended)** | `python -m src.main query "topic" -k 5`              |
| `search`         | Full search with logging    | `python -m src.main search "topic" --top-k 10`       |
| `stats`          | Show DB statistics          | `python -m src.main stats`                           |
| `process`        | Index documents             | `python -m src.main process`                         |
| `index-code`     | Index source code           | `python -m src.main index-code /path --tag proj`     |
| `extract-images` | Extract PDF images          | `python -m src.main extract-images --input file.pdf` |
| `update`         | Add context/tags            | `python -m src.main update "query" "note" -y`        |
| `clean`          | Reset database              | `python -m src.main clean -y`                        |
| `init`           | Initialize folders          | `python -m src.main init`                            |

---

## Query Command (Primary for AI)

```bash
python -m src.main query "search terms" [options]

Options:
  -k, --top-k N        Number of results (default: 5)
  --format FORMAT      Output format: text|json
  --hybrid             Enable keyword matching (default)
  --no-hybrid          Pure semantic search
```

### When to Query

✅ **DO query:**

- Before implementing features (find patterns)
- When debugging (find error handling)
- For unfamiliar code (find docs)

❌ **DON'T query:**

- Trivial changes (typos, formatting)
- File already visible
- External docs (use web search)

### Query Examples

```bash
# Basic search
python -m src.main query "WiFi authentication" -k 5

# More results
python -m src.main query "connection state" -k 20

# Exact term matching
python -m src.main query "WPA_COMPLETED" --hybrid -k 5

# Conceptual search
python -m src.main query "how does handshake work" --no-hybrid -k 10

# JSON output for parsing
python -m src.main query "error handling" --format json -k 5
```

---

## Other Commands

### process - Index Documents

```bash
python -m src.main process [--input DIR] [--no-charts]
```

### index-code - Index Source Code

```bash
python -m src.main index-code PATH [--tag TAG] [--ext .c .h]

# Supported: .c .h .cpp .py .java .js .ts .go .rs .sh
```

### extract-images - Extract from PDFs

```bash
python -m src.main extract-images [--input PATH] [--no-vectorize]
```

### update - Add Context

```bash
python -m src.main update "query" "context" [-y]
```

### stats - Database Info

```bash
python -m src.main stats
```

### clean - Reset Database

```bash
python -m src.main clean [-y] [--all]
```

---

## Workflow: First Time Setup

```bash
python -m src.main init
# Place documents in data/documents/input-doc/
python -m src.main process
python -m src.main stats
```

## Workflow: Search for Context

```bash
# 1. Check what's indexed
python -m src.main stats

# 2. Query for context
python -m src.main query "topic" -k 5

# 3. Increase results if needed
python -m src.main query "topic" -k 20
```

---

## Output Format

Results include:

- **Source**: File path, page number
- **Type**: text, image, code, chart
- **Content**: Matched snippet
- **Score**: Relevance (higher = better)
