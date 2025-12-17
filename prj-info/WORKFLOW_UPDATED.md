# RAG System Workflow - Phase 1 + 2 Improvements

## 🎯 Latest Updates (2025-12-18)

### Phase 1: Enhanced Search Output ✅

- **[TEXT]/[IMAGE] type indicators** - Phân biệt rõ ràng loại kết quả
- **Full content display** - Không còn cắt ngắn ở 200 ký tự
- **Expanded context window** - ±150px dọc, ±20px ngang (từ 50px/20px)
- **Better formatting** - Icons 📷📄🎯, multi-line display

### Phase 2: Optimized Chunking ✅

- **Chunk size: 512 → 1024** - Giảm 47% số chunks (622 → 332)
- **Overlap: 50 → 100** - Context tốt hơn
- **Structure-aware separators** - Không cắt giữa sections (##, ###, ####)

### Clean Architecture ✅

- **Single extraction method** - Chỉ PyMuPDF smart cropping
- **No PaddleOCR** - Removed (unstable, DLL conflicts)
- **No full-page** - Removed (too generic)

### Current Stats

```
Total: 417 items
├── Text chunks: 332 (optimized)
└── Images: 85 (smart crops)
```

---

## 📊 Processing Flow

```
Input PDF (e.g., HFP_v1.8.pdf - 139 pages)
    ↓
┌─────────────────────────────────────────────────────┐
│ STEP 1: Text Extraction & Chunking                  │
├─────────────────────────────────────────────────────┤
│ • Extract text from PDF                             │
│ • Split into chunks:                                │
│   - Size: 1024 characters (Phase 2)                 │
│   - Overlap: 100 characters (Phase 2)               │
│   - Separators: ##, ###, ####, \n\n, \n, ...       │
│ • Output: 332 text chunks                           │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ STEP 2: Smart Image Extraction (Phase 1)            │
├─────────────────────────────────────────────────────┤
│ • Detect bitmap images (get_image_info)             │
│ • Cluster vector drawings (get_drawings)            │
│ • Merge nearby regions (±50px)                      │
│ • Render crops with 150 DPI                         │
│ • Extract context (±150px down, ±80px up)           │
│ • Save as: filename_p{page}_crop_{number}.png       │
│ • Output: 85 smart crops                            │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ STEP 3: Vectorization                               │
├─────────────────────────────────────────────────────┤
│ • Model: Alibaba-NLP/gte-multilingual-base          │
│ • Dimension: 768                                    │
│ • Process: Text → Embedding vector                  │
│ • Total: 417 vectors (332 + 85)                     │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ STEP 4: Storage (FAISS Index)                       │
├─────────────────────────────────────────────────────┤
│ • Index type: Flat (exact search)                   │
│ • Distance metric: Cosine similarity                │
│ • Location: data/faiss_index/                       │
│ • Size: ~1.3 MB                                     │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ STEP 5: Search (Phase 1 improvements)               │
├─────────────────────────────────────────────────────┤
│ Query: "Optional state/condition"                   │
│     ↓                                               │
│ [1] Score: 0.6265 [TEXT]                           │
│     📄 Page: 45                                     │
│     Content: Figure 1.2: Conventions...            │
│                                                      │
│ [2] Score: 0.6012 [IMAGE]                          │
│     📷 Image: HFP_v1.8_p14_crop_1.png              │
│     📄 Page: 14                                     │
│     🎯 Type: smart_crop                             │
│     Content: [Full context shown]                   │
└─────────────────────────────────────────────────────┘
```

---

## 🔧 Configuration Details

### Chunking (Phase 2)

**File:** `config/settings.py`

```python
class ChunkingConfig(BaseModel):
    chunk_size: int = 1024  # ← Phase 2: Increased from 512
    chunk_overlap: int = 100  # ← Phase 2: Increased from 50

    # Phase 2: Structure-aware separators
    separators: list[str] = Field(default=[
        "\n\n## ",      # H2 headers (don't split sections)
        "\n\n### ",    # H3 headers (don't split subsections)
        "\n\n#### ",   # H4 headers
        "\n\n",        # Paragraphs
        "\n",          # Lines
        ". ",          # Sentences
        " ",           # Words
        ""             # Characters
    ])
```

**Impact:**

- Fewer chunks (332 vs 622)
- Better context preservation
- Diagrams + descriptions stay together

### Image Extraction (Phase 1)

**File:** `src/image_extraction/extractor.py`

```python
# Context window expansion (Phase 1)
context_rect = fitz.Rect(rect)
context_rect.y1 += 150  # ← Was 50px, now 150px down
context_rect.y0 -= 80   # ← Was 20px, now 80px up
context_rect.x0 -= 20   # ← New: horizontal expansion
context_rect.x1 += 20   # ← New: horizontal expansion
text = page.get_text(clip=context_rect)
```

**Impact:**

- Captures diagram titles (above)
- Captures captions (below)
- Full figure descriptions
- Better search relevance

### Search Output (Phase 1)

**File:** `src/search/engine.py`

```python
def format_results(results, format_type="text"):
    # Phase 1: Type indicators + full content
    type_label = "[IMAGE]" if is_image else "[TEXT]"

    if is_image:
        lines.append(f"📷 Image: {image_path}")
        lines.append(f"📄 Page: {page_num}")
        lines.append(f"🎯 Type: {region_type}")
        lines.append(f"Content: {full_document}")  # No truncation!
    else:
        # Full content for text too
        if len(document) > 300:
            # Show with indentation
        else:
            lines.append(f"Content: {full_document}")
```

---

## 📈 Performance Metrics

### HFP v1.8 Document (139 pages)

| Metric      | Before | After (Phase 2) | Improvement    |
| ----------- | ------ | --------------- | -------------- |
| Text Chunks | 622    | 332             | -47%           |
| Images      | 0\*    | 85              | +∞             |
| Total Items | 622    | 417             | Better quality |
| Chunk Size  | 512    | 1024            | 2x larger      |
| Overlap     | 50     | 100             | 2x context     |

\*Images were extracted before but with issues

### Processing Time

- Chunking: ~2-3 seconds
- Image extraction: ~30-40 seconds
- Vectorization: ~45-50 seconds
- **Total: ~80-90 seconds**

### Search Performance

- Query vectorization: <100ms
- FAISS search: <50ms
- Result formatting: <10ms
- **Total: <200ms per query**

---

## 🎨 Search Output Examples

### Before Phase 1

```
[1] Score: 0.7130
    Source: HFP_v1.8.pdf
    Content: HF AG Synchronous Connection Setup...  ← TRUNCATED
```

❌ No type, truncated, no metadata

### After Phase 1 + 2

```
[1] Score: 0.6265 [TEXT]  ← Type indicator
    Source: HFP_v1.8.pdf
    📄 Page/Chunk: 45  ← Clear metadata
    Content:  ← Full content below
        Hands-Free Profile / Profile Specification
        Bluetooth SIG Proprietary Page 14 of 139

        Figure 1.2: Conventions used in signaling diagrams

        A B
        Mandatory procedure initiated by B
        Mandatory signal sent by A
        Optional signal sent by B
        ...
        Current state/condition
        Optional state/condition  ← COMPLETE!

[2] Score: 0.6012 [IMAGE]  ← Image result!
    Source: HFP_v1.8.pdf
    📷 Image: extracted_images\HFP_v1.8_p14_crop_1.png  ← Full path
    📄 Page: 14
    🎯 Type: smart_crop
    Content: Description: Smart crop from HFP_v1.8.pdf, page 14
    Context:  ← Full surrounding text
    Hands-F
    e 1.2: Conventions used in signaling diagrams
    ...
```

✅ Type clear, full content, image path, complete metadata

---

## 🚀 Usage Guide

### Daily Workflow

1. **Add documents**

   ```bash
   # Copy PDFs to input folder
   cp *.pdf data/documents/input-doc/
   ```

2. **Process**

   ```bash
   # Clean old data
   python -m src.main clean -y

   # Process with Phase 2 config
   python -m src.main process
   # → 332 chunks + 85 images = 417 items
   ```

3. **Search**
   ```bash
   # Use Phase 1 improved output
   python -m src.main search "Audio Gateway"
   # → Shows [TEXT]/[IMAGE] with full content
   ```

### Troubleshooting

**Q: Too many/few chunks?**

```python
# Edit config/settings.py
chunk_size: int = 1024  # Increase/decrease
```

**Q: Images not extracted?**

```bash
# Check PyMuPDF
pip install pymupdf

# Verify PDF has diagrams
# Some PDFs are text-only
```

**Q: Search results not relevant?**

```bash
# Lower threshold
python -m src.main search "query" --threshold 0.3

# Increase results
python -m src.main search "query" --top-k 10
```

---

## 📚 Additional Documentation

- **IMPROVEMENTS.md** - Detailed Phase 1 + 2 changes
- **README.md** - Quick start guide
- **Implementation plans** - Phase 3 roadmap

---

**Last Updated:** 2025-12-18  
**Version:** 0.3.0 (Phase 1 + 2 Complete)  
**Status:** ✅ Production Ready
