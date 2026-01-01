# RAG System - Recent Improvements & Configuration

## Overview

This document describes the recent improvements made to the RAG system (Phase 1 + Phase 2) and the current configuration optimized for technical documentation with diagrams.

---

## Phase 1: Enhanced Search Output ✅

### Improvements Made

**1. Type Indicators**

- Added `[TEXT]` and `[IMAGE]` labels to distinguish result types
- Visual icons: 📷 (image path), 📄 (page number), 🎯 (type)

**2. Full Content Display**

- Removed 200-character truncation
- Shows complete content for better context understanding
- Multi-line formatting for readability

**3. Image Path Display**

- Full path to extracted PNG files
- Page number and bounding box info
- Region type (smart_crop)

**4. Expanded Context Window**

- Vertical: ±150px down (was 50px), ±80px up (was 20px)
- Horizontal: ±20px left/right
- Captures diagram captions, titles, and descriptions better

### Example Output

```
Found 3 results:

[1] Score: 0.6265 [TEXT]
    Source: HFP_v1.8.pdf
    📄 Page/Chunk: 45
    Content:
        Figure 1.2: Conventions used in signaling diagrams

        A B
        Mandatory procedure initiated by B
        Optional state/condition
        [FULL CONTENT - NO TRUNCATION]

[2] Score: 0.6012 [IMAGE]
    Source: HFP_v1.8.pdf
    📷 Image: E:\...\extracted_images\HFP_v1.8_p14_crop_1.png
    📄 Page: 14
    🎯 Type: smart_crop
    Content: Description: Smart crop from HFP_v1.8.pdf, page 14
    Context: [Full surrounding text]
```

### Files Modified

- `src/search/engine.py` - Enhanced `format_results()` method
- `src/image_extraction/extractor.py` - Expanded context window

---

## Phase 2: Optimized Chunking ✅

### Configuration Changes

**File:** `config/settings.py`

```python
class ChunkingConfig(BaseModel):
    """Configuration for document chunking - optimized for technical docs"""
    chunk_size: int = 1024  # Increased from 512 (Phase 2)
    chunk_overlap: int = 100  # Increased from 50 (Phase 2)

    # Structure-aware separators for technical documentation
    separators: list[str] = Field(default=[
        "\n\n## ",      # Section headers (H2)
        "\n\n### ",    # Subsection headers (H3)
        "\n\n#### ",   # Sub-subsection headers (H4)
        "\n\n",        # Paragraphs
        "\n",          # Lines
        ". ",          # Sentences
        " ",           # Words
        ""             # Characters
    ])
    length_function: str = "len"
```

### Impact

**Before (512 chunk_size):**

- 622 text chunks from HFP document
- Context often incomplete
- Diagrams + descriptions split across chunks

**After (1024 chunk_size):**

- 332 text chunks (-47% reduction!)
- Complete context preserved
- Diagrams kept with descriptions
- Better search relevance

---

## Clean Architecture: Image Extraction

### Single Method Approach

The system now uses **ONLY** PyMuPDF smart cropping:

- ✅ Removed PaddleOCR (unstable, DLL conflicts)
- ✅ Removed full-page extraction (too generic)
- ✅ Single, reliable extraction method

### Smart Cropping Algorithm

**File:** `src/image_extraction/extractor.py`

```python
def _extract_smart_crops(pdf_path):
    """
    1. Detect embedded images (get_image_info)
    2. Cluster vector drawings (get_drawings + clustering)
    3. Merge nearby regions (bounding box merge)
    4. Render & save PNG files
    5. Extract ±150px context text
    """
```

### Extraction Results

**Example: HFP_v1.8.pdf (139 pages)**

- 85 smart crops extracted
- ~61% extraction rate (meaningful diagrams)
- File naming: `HFP_v1.8_p{page}_crop_{number}.png`

---

## Current System Stats

### Database Contents

```
Total: 417 searchable items
├── Text chunks: 332 (optimized chunking)
└── Images: 85 (smart crops with context)
```

### Processing Pipeline

```
PDF Document
    ↓
[1] Text Extraction → Chunking (1024 chars, 100 overlap)
    ↓
[2] Image Extraction → Smart Cropping (±150px context)
    ↓
[3] Vectorization → Alibaba-NLP/gte-multilingual-base (768-dim)
    ↓
[4] Storage → FAISS Index (fast similarity search)
```

### Performance

- **Chunking**: ~2-3 seconds for 139-page PDF
- **Image Extraction**: ~30-40 seconds for 139-page PDF
- **Vectorization**: ~45-50 seconds for 417 items
- **Search**: <1 second for top-5 results

---

## Configuration Reference

### Chunking

```python
# config/settings.py
chunk_size: int = 1024      # Optimal for technical docs
chunk_overlap: int = 100    # Good context preservation
```

### Image Extraction

```python
# src/image_extraction/extractor.py
min_width: int = 50         # Minimum image width
min_height: int = 50        # Minimum image height
cluster_merge_distance: int = 50    # Merge nearby elements
min_diagram_size: int = 100 # Minimum diagram size

# Context window (Phase 1)
context_rect.y1 += 150  # Below (caption)
context_rect.y0 -= 80   # Above (title)
context_rect.x0 -= 20   # Left
context_rect.x1 += 20   # Right
```

### Search

```python
# config/settings.py
top_k: int = 5              # Default results count
score_threshold: float = 0.5 # Similarity threshold
```

---

## Usage Examples

### Process Documents

```bash
# Clean previous data
python -m src.main clean -y

# Process with new config
python -m src.main process

# Output:
# Chunked: HFP_v1.8.pdf -> 332 chunks
# Extracted 85 smart crops
# Total: 417 items in database
```

### Search

```bash
python -m src.main search "Audio Gateway side" --top-k 3

# Output shows:
# - [TEXT] or [IMAGE] type indicators
# - Full content (no truncation)
# - Image paths for visual results
# - Page numbers and metadata
```

### View Stats

```bash
python -m src.main stats

# Output:
# Total items in database: 417
# Backend: FAISS
# Index type: Flat
```

---

## Recommended Workflow

### Initial Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Initialize folders
python -m src.main init

# 3. Add PDFs to data/documents/input-doc/

# 4. Process
python -m src.main process
```

### Daily Usage

```bash
# Search for content
python -m src.main search "your query"

# Add new documents
# 1. Copy to input-doc/
# 2. python -m src.main process

# Update database
python -m src.main clean -y
python -m src.main process
```

---

## Phase 3: Hybrid Search ✅

### Overview

Implemented **Hybrid Search** combining Vector Search (semantic) with BM25 (keyword matching) for significantly improved retrieval quality.

### How It Works

```
Query: "WiFi authentication timeout"
                    │
    ┌───────────────┴───────────────┐
    ▼                               ▼
┌─────────────┐               ┌─────────────┐
│ Vector      │               │ BM25        │
│ (Semantic)  │               │ (Keywords)  │
│ Score: 0.72 │               │ Score: 0.85 │
└─────┬───────┘               └─────┬───────┘
      │                             │
      └─────────────┬───────────────┘
                    ▼
            ┌───────────────┐
            │ Hybrid Score  │
            │ = 0.7 × 0.72  │
            │ + 0.3 × 0.85  │
            │ = 0.759       │
            └───────────────┘
```

### Benefits

| Feature                | Vector Only | Hybrid |
| ---------------------- | ----------- | ------ |
| Semantic understanding | ✅          | ✅     |
| Exact keyword match    | ❌          | ✅     |
| Technical terms        | Weak        | Strong |
| Error codes, IDs       | Miss        | Match  |

### New Files

- `src/search/bm25_search.py` - BM25 index with Vietnamese stopwords
- Updated `src/search/engine.py` - `_hybrid_rerank()` method
- Updated `config/settings.py` - Hybrid search config

### Configuration

```python
# config/settings.py
class SearchConfig(BaseModel):
    use_hybrid: bool = True       # Enable by default
    hybrid_alpha: float = 0.7     # 70% vector + 30% BM25
    bm25_k1: float = 1.5          # Term frequency saturation
    bm25_b: float = 0.75          # Length normalization
```

### CLI Usage

```bash
# Hybrid search (default)
python -m src.main search "WiFi timeout error"

# Pure vector search
python -m src.main search "authentication flow" --no-hybrid

# Force hybrid
python -m src.main search "error 0x8007" --hybrid
```

---

## Phase 4: OlmOCR Integration ✅

### Overview

Integrated advanced prompting techniques from [OlmOCR](https://github.com/allenai/olmocr) project for enhanced Vision Language Model (VLM) understanding and better semantic search.

### Key Features

**1. Anchor Text Generation**

Generate OlmOCR-style positional context for extracted images:

```
Page dimensions: 612x792
[Image 100x200 to 300x400]
[72x720]Title of the document
[72x680]Author name here
[72x300]Main content text...
```

**2. Content Type Detection**

Automatically classify images as:

- `DIAGRAM` - Vector drawings, flowcharts, architecture diagrams
- `TABLE` - Tabular data with grid patterns
- `IMAGE` - Generic images/photos

**3. Chunk Classification**

Automatically classify text chunks:

| Type      | Detection                                 |
| --------- | ----------------------------------------- |
| `text`    | Default prose content                     |
| `code`    | Contains `def `, `function `, code blocks |
| `table`   | Markdown tables with `\|` patterns        |
| `heading` | Starts with `#`                           |
| `list`    | Multiple list items                       |

**4. Enhanced Metadata**

New fields for chunks and images:

- `has_equations` - Contains LaTeX (`\(`, `\[`, `$$`)
- `has_code_blocks` - Contains code blocks or code keywords
- `anchor_text` - Positional context for images
- `is_diagram`, `is_table` - Content classification

### Files Modified

- `src/image_extraction/extractor.py` - Anchor text + content detection
- `src/chunking/chunker.py` - Chunk type classification
- `src/vectorization/vectorizer.py` - Enhanced metadata output

### Benefits

| Feature              | Before        | After                |
| -------------------- | ------------- | -------------------- |
| Image search         | Basic context | Rich anchor text     |
| Content filtering    | None          | Type prefixes        |
| Chunk classification | Generic       | Auto-detected types  |
| VLM understanding    | Limited       | Positional grounding |

### Reference

- OlmOCR: [github.com/allenai/olmocr](https://github.com/allenai/olmocr)
- Paper: "olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models"

---

## Future Enhancements (Phase 5)

**Planned features:**

- Result grouping (group related text + images)
- Deduplication (remove similar results)
- Cross-references (show "See also: Figure X")
- Section detection (link images to sections)
- Cross-encoder re-ranking (for higher precision)
- Search type filtering (filter by DIAGRAM, TABLE, CODE)

---

## Troubleshooting

### Common Issues

**1. No images extracted**

```bash
# Check PyMuPDF installed
pip install pymupdf

# Verify PDF has diagrams
# Some PDFs have text-only content
```

**2. Search returns too few results**

```bash
# Lower threshold
python -m src.main search "query" --threshold 0.3

# Increase top-k
python -m src.main search "query" --top-k 10
```

**3. Chunks too large/small**

```python
# Edit config/settings.py
chunk_size: int = 1024  # Adjust as needed
chunk_overlap: int = 100
```

---

## Version History

- **v0.1.0** - Initial release
- **v0.2.0 (Phase 1)** - Enhanced search output, expanded context
- **v0.3.0 (Phase 2)** - Optimized chunking, clean architecture
- **v0.4.0 (Phase 3)** - Hybrid Search (Vector + BM25)
- **v0.5.0 (Phase 4)** - OlmOCR Integration (anchor text, content classification)
- **v0.6.0 (Planned)** - Phase 5 grouping & cross-references

---

**Last Updated:** 2026-01-01  
**System Status:** ✅ Production Ready (Phase 1 + 2 + 3 + 4 Complete)
