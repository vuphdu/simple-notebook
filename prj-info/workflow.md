# Simple RAG System - Chi Tiết Kiến Trúc & Workflow

## 📋 Tổng Quan Dự Án

**Simple RAG System** là một hệ thống Retrieval-Augmented Generation (RAG) đơn giản nhưng mạnh mẽ, cho phép:

- Xử lý và chunking các tài liệu (PDF, TXT, MD, DOCX)
- Trích xuất hình ảnh từ PDF bằng PyMuPDF smart cropping
- Vector hóa nội dung bằng mô hình embedding đa ngôn ngữ
- Lưu trữ và tìm kiếm ngữ nghĩa (semantic search)

---

## 🗂️ Cấu Trúc Thư Mục

```
simple-notebook/
├── config/                    # Cấu hình hệ thống
│   ├── __init__.py           # Export các config classes
│   └── settings.py           # Định nghĩa các class cấu hình
│
├── src/                       # Source code chính
│   ├── __init__.py
│   ├── main.py               # Entry point CLI
│   ├── chunking/             # Module chia nhỏ tài liệu
│   │   ├── __init__.py
│   │   └── chunker.py
│   ├── vectorization/        # Module vector hóa
│   │   ├── __init__.py
│   │   ├── vectorizer.py     # Text → Vector embedding
│   │   ├── vector_store.py   # ChromaDB backend
│   │   └── faiss_store.py    # FAISS backend
│   ├── search/               # Module tìm kiếm
│   │   ├── __init__.py
│   │   └── engine.py
│   ├── image_extraction/     # Module trích xuất hình ảnh
│   │   ├── __init__.py
│   │   └── extractor.py      # PyMuPDF smart cropping
│   └── sequence_chart/       # Module xử lý sequence diagram
│       ├── __init__.py
│       └── processor.py
│
├── data/                      # Dữ liệu
│   ├── documents/
│   │   ├── input-doc/        # Tài liệu đầu vào
│   │   └── extracted_images/ # Hình ảnh đã trích xuất
│   ├── vectordb/             # ChromaDB storage
│   └── faiss_index/          # FAISS index files
│
├── models/                    # Cached embedding models
├── process/                   # Input/Output của search queries
│   ├── input/
│   └── output/
│
├── requirements.txt
├── pyproject.toml
└── tests/
```

---

## 📦 Thư Viện Sử Dụng (Tối Ưu - 17 Packages)

### Core Dependencies

| Thư viện                | Phiên bản | Mục đích                                          |
| ----------------------- | --------- | ------------------------------------------------- |
| `torch`                 | ≥2.0.0    | Deep learning framework, backend cho transformers |
| `transformers`          | ≥4.35.0   | Hugging Face transformers cho NLP models          |
| `sentence-transformers` | ≥2.2.0    | Embedding text thành vectors                      |
| `pydantic`              | ≥2.0.0    | Data validation và settings management            |
| `numpy`                 | ≥1.24.0   | Numerical operations                              |
| `tqdm`                  | ≥4.66.0   | Progress bars                                     |

### Vector Database

| Thư viện | Phiên bản | Mục đích |
| `pytest` | ≥7.4.0 | Testing framework |
| `pytest-asyncio` | ≥0.21.0 | Async testing support |

---

## ⚙️ Cấu Hình Hệ Thống

### `config/settings.py`

```
┌─────────────────────────────────────────────────────────────────┐
│                         Settings                                 │
│  (Pydantic BaseModel - Global Configuration)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ ModelConfig                                              │    │
│  │ - model_name: "Alibaba-NLP/gte-multilingual-base"       │    │
│  │ - model_cache_dir: Path(MODELS_DIR)                     │    │
│  │ - max_seq_length: 8192                                  │    │
│  │ - device: "cpu" | "cuda"                                │    │
│  │ - normalize_embeddings: True                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ ChunkingConfig                                           │    │
│  │ - chunk_size: 512                                       │    │
│  │ - chunk_overlap: 50                                     │    │
│  │ - separators: ["\\n\\n", "\\n", ". ", " ", ""]            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ VectorDBConfig                                           │    │
│  │ - backend: "faiss" | "chromadb"                         │    │
│  │ - collection_name: "documents"                          │    │
│  │ - persist_directory: Path(VECTORDB_DIR)                 │    │
│  │ - faiss_index_dir: Path(FAISS_INDEX_DIR)               │    │
│  │ - distance_metric: "cosine" | "l2" | "ip"              │    │
│  │ - faiss_index_type: "Flat" | "IVF" | "HNSW"            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ SearchConfig                                             │    │
│  │ - top_k: 5                                              │    │
│  │ - score_threshold: 0.5                                  │    │
│  │ - include_metadata: True                                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ ImageExtractionConfig                                    │    │
│  │ - output_dir: EXTRACTED_IMAGES_DIR                      │    │
│  │ - min_width: 50                                         │    │
│  │ - min_height: 50                                        │    │
│  │ - cluster_merge_distance: 50                            │    │
│  │ - min_drawing_size: 20                                  │    │
│  │ - min_diagram_size: 100                                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Workflow Chính

### 1. Luồng Xử Lý Tài Liệu (`process`) order

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DOCUMENT PROCESSING FLOW                           │
└─────────────────────────────────────────────────────────────────────────────┘

                              INPUT
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
            ┌─────────────┐         ┌─────────────┐
            │   PDF Files │         │ TXT/MD/DOCX │
            └──────┬──────┘         └──────┬──────┘
                   │                       │
          ┌────────┴────────┐              │
          ▼                 ▼              │
    ┌───────────┐    ┌───────────┐         │
    │ PyMuPDF   │    │ PyPDF2    │         │
    │ Smart Crop│    │ (Text)    │         │
    └─────┬─────┘    └─────┬─────┘         │
          │                │               │
          │                └───────┬───────┘
          │                        ▼
          │              ┌─────────────────┐
          │              │ DocumentChunker │
          │              │ (custom impl)   │
          │              └────────┬────────┘
          │                       │
          │                       ▼
          │              ┌─────────────────┐
          │              │ Text Chunks     │
          │              │ (DocumentChunk) │
          │              └────────┬────────┘
          │                       │
          ▼                       ▼
    ┌───────────┐        ┌─────────────────┐
    │ Extracted │        │ TextVectorizer  │
    │ Images    │        │ (sentence-      │
    │           │        │  transformers)  │
    └─────┬─────┘        └────────┬────────┘
          │                       │
          │      ┌────────────────┘
          │      │
          ▼      ▼
    ┌─────────────────┐
    │ Vector Embeddings│
    │ (768-dim vectors)│
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ VectorStore     │
    │ (FAISS/ChromaDB)│
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Persisted Index │
    │ + Metadata      │
    └─────────────────┘
```

### 2. Luồng Smart Image Extraction

```
┌─────────────────────────────────────────────────────────────────┐
│              PYMUPDF SMART CROPPING WORKFLOW                     │
└─────────────────────────────────────────────────────────────────┘

    PDF File
        │
        ▼
    ┌─────────────────┐
    │ PyMuPDF Open    │
    │ fitz.open()     │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ For each page   │
    └────────┬────────┘
             │
    ┌────────┴────────┬────────────────┐
    ▼                 ▼                ▼
┌──────────┐  ┌──────────────┐  ┌──────────────┐
│ Detect   │  │ Cluster      │  │ Detect       │
│ Bitmap   │  │ Vector       │  │ Embedded     │
│ Images   │  │ Drawings     │  │ Images       │
└─────┬────┘  └──────┬───────┘  └──────┬───────┘
      │              │                 │
      └──────┬───────┴─────────────────┘
             │
             ▼
    ┌─────────────────┐
    │ Merge Nearby    │
    │ Regions         │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Render & Crop   │
    │ (150 DPI PNG)   │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Extract Context │
    │ (surrounding    │
    │  text)          │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Save to         │
    │ extracted_      │
    │ images/         │
    └─────────────────┘
```

---

## 🚀 Các Lệnh CLI

### Process Documents

```bash
# Xử lý tất cả documents
python -m src.main process

# Chỉ định thư mục thay vì
python -m src.main process --input data/documents

# Bỏ qua sequence charts
python -m src.main process --no-charts
```

### Search

```bash
# Tìm kiếm
python -m src.main search "authentication flow"

# Chỉ định số kết quả
python -m src.main search "login" --top-k 10

# Không lưu kết quả
python -m src.main search "query" --no-save
```

### Extract Images

```bash
# Trích xuất và vector hóa
python -m src.main extract-images

# Chỉ trích xuất, không vector hóa
python -m src.main extract-images --no-vectorize

# Chỉ định file cụ thể
python -m src.main extract-images --input data/documents/file.pdf
```

### Update Context

```bash
# Thêm context cho document/image
python -m src.main update "page 100" "Login flow diagram"

# Skip confirmation
python -m src.main update "diagram" "Architecture" --yes
```

### Stats

```bash
# Xem thống kê
python -m src.main stats
```

### Clean

```bash
# Dọn dẹp vectordb
python -m src.main clean

# Không hỏi
python -m src.main clean -y

# Dọn cả models
python -m src.main clean --all -y
```

---

## 💡 Best Practices

### 1. Document Organization

- Đặt tất cả tài liệu trong `data/documents/input-doc/`
- Sử dụng tên file rõ ràng, mô tả nội dung
- Tổ chức theo thư mục con nếu cần

### 2. Search Optimization

- Sử dụng natural language queries
- Thử different phrasings nếu kết quả không tốt
- Adjust `top_k` tùy use case

### 3. Image Extraction

- Smart cropping hoạt động tốt nhất với vector graphics
- PDF quality càng cao, kết quả càng tốt
- Kiểm tra `extracted_images/` để verify

### 4. Vector Store Backend

- **FAISS**: Dùng cho speed (default, recommended)
- **ChromaDB**: Dùng khi cần advanced filtering

---

## 🎯 So Với Trước Đây

| Feature              | Trước                              | Sau                         |
| -------------------- | ---------------------------------- | --------------------------- |
| **Dependencies**     | 19 packages (~1.5GB)               | 16 packages (~1GB)          |
| **Image Extraction** | 3 modes (PyMuPDF/Paddle/Full-page) | 1 mode (Smart Cropping)     |
| **Code Lines**       | ~1400 lines                        | ~570 lines                  |
| **Complexity**       | High (multiple modes)              | Low (single focused method) |
| **Stability**        | Medium (DLL conflicts)             | High (no conflicts)         |
| **CLI**              | Confusing (--mode)                 | Simple (no options)         |

---

## 📄 License

MIT License
