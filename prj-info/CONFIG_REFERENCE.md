# 📋 Báo Cáo Chi Tiết Cấu Hình Dự Án Simple RAG System

> **Ngày tạo:** 2025-12-22  
> **Mục đích:** Tài liệu tham khảo nhanh các cấu hình quan trọng trong dự án

---

## 🏗️ Cấu Trúc Thư Mục

```
simple-notebook/
├── data/
│   ├── documents/
│   │   ├── input-doc/          # Thư mục chứa tài liệu nguồn để xử lý
│   │   └── extracted_images/   # Ảnh được trích xuất từ PDF
│   ├── vectordb/               # ChromaDB storage
│   └── faiss_index/            # FAISS index storage
├── models/                     # Cache model ML (tự download lần đầu)
├── process/
│   ├── input/                  # Log query tìm kiếm
│   └── output/                 # Log kết quả tìm kiếm
├── src/                        # Source code các module
├── config/                     # File cấu hình
└── tests/                      # Unit tests
```

---

## ⚙️ Cấu Hình Chi Tiết

> **File cấu hình chính:** `config/settings.py`

### 1. ModelConfig - Cấu hình Model Embedding

| Tham số                | Giá trị mặc định                    | Mô tả                                               |
| ---------------------- | ----------------------------------- | --------------------------------------------------- |
| `model_name`           | `Alibaba-NLP/gte-multilingual-base` | Model embedding đa ngôn ngữ                         |
| `model_cache_dir`      | `/models/`                          | Thư mục cache model                                 |
| `max_seq_length`       | `8192`                              | Độ dài sequence tối đa                              |
| `device`               | **Auto-detect**                     | Tự động chọn `"cuda"` nếu có GPU, ngược lại `"cpu"` |
| `normalize_embeddings` | `True`                              | Chuẩn hóa embedding                                 |
| `force_cpu`            | `False`                             | Đặt `True` để buộc dùng CPU dù có GPU               |

> **🚀 Auto-detect GPU:** Hệ thống tự động phát hiện GPU khi khởi động. Nếu có CUDA GPU, sẽ in thông báo: `🚀 GPU detected: [GPU Name] (X.X GB)`

### 2. ChunkingConfig - Cấu hình Chunking văn bản

| Tham số           | Giá trị mặc định                   | Mô tả                             |
| ----------------- | ---------------------------------- | --------------------------------- |
| `chunk_size`      | `1024`                             | Kích thước chunk (ký tự)          |
| `chunk_overlap`   | `100`                              | Độ chồng lấp giữa các chunk       |
| `separators`      | `["\\n\\n## ", "\\n\\n### ", ...]` | Các dấu phân cách ưu tiên         |
| `length_function` | `"len"`                            | Hàm đo độ dài (hoặc `"tiktoken"`) |

### 3. VectorDBConfig - Cấu hình Vector Database

| Tham số             | Giá trị mặc định    | Mô tả                                         |
| ------------------- | ------------------- | --------------------------------------------- |
| `backend`           | **`"faiss"`**       | Backend sử dụng (`"faiss"` hoặc `"chromadb"`) |
| `collection_name`   | `"documents"`       | Tên collection                                |
| `persist_directory` | `data/vectordb/`    | Thư mục lưu ChromaDB                          |
| `faiss_index_dir`   | `data/faiss_index/` | Thư mục lưu FAISS index                       |
| `distance_metric`   | `"cosine"`          | Metric khoảng cách (`cosine`, `l2`, `ip`)     |
| `faiss_index_type`  | `"Flat"`            | Loại FAISS index (`Flat`, `IVF`, `HNSW`)      |

#### Các loại FAISS Index:

| Type       | Mô tả                              | Khi nào dùng                |
| ---------- | ---------------------------------- | --------------------------- |
| **`Flat`** | Brute-force, chính xác 100%        | Dataset nhỏ (<100k vectors) |
| **`IVF`**  | Inverted File Index, nhanh hơn     | Dataset trung bình          |
| **`HNSW`** | Hierarchical Navigable Small World | Dataset lớn, cần tốc độ cao |

### 4. SearchConfig - Cấu hình Tìm kiếm

| Tham số            | Giá trị mặc định | Mô tả                            |
| ------------------ | ---------------- | -------------------------------- |
| `top_k`            | `5`              | Số kết quả trả về                |
| `score_threshold`  | `0.5`            | Ngưỡng điểm similarity tối thiểu |
| `include_metadata` | `True`           | Bao gồm metadata trong kết quả   |

### 5. SequenceChartConfig - Cấu hình Sequence Chart

| Tham số               | Giá trị mặc định | Mô tả                                  |
| --------------------- | ---------------- | -------------------------------------- |
| `export_format`       | `"png"`          | Định dạng export (`png`, `svg`, `pdf`) |
| `export_quality`      | `150`            | DPI chất lượng ảnh                     |
| `include_description` | `True`           | Bao gồm mô tả                          |

---

## 📦 Dependencies (`requirements.txt`)

| Nhóm                    | Packages                                                               |
| ----------------------- | ---------------------------------------------------------------------- |
| **Deep Learning**       | `torch>=2.0.0`, `transformers>=4.35.0`, `sentence-transformers>=2.2.0` |
| **Vector Database**     | `chromadb>=0.4.0`, `faiss-cpu>=1.7.4`                                  |
| **Document Processing** | `langchain-text-splitters`, `python-docx>=0.8.11`, `PyPDF2>=3.0.0`     |
| **Image Extraction**    | `pymupdf>=1.24.0`, `Pillow>=10.0.0`                                    |
| **Utilities**           | `pydantic>=2.0.0`, `tqdm>=4.66.0`, `numpy>=1.24.0`                     |
| **Testing**             | `pytest>=7.4.0`, `pytest-asyncio>=0.21.0`                              |

**Python version yêu cầu:** `>=3.10`

---

## 🚀 Các Lệnh CLI Thường Dùng

```bash
# Khởi tạo thư mục
python -m src.main init

# Xử lý tài liệu (extract + chunking + vectorize)
python -m src.main process --input data/documents

# Tìm kiếm
python -m src.main search "từ khóa" --top-k 10

# Trích xuất ảnh từ PDF
python -m src.main extract-images --input data/documents

# Trích xuất ảnh không vectorize
python -m src.main extract-images --input data/documents/file.pdf --no-vectorize

# Thêm context/tag cho document/image
python -m src.main update "query" "context mới"

# Xem thống kê
python -m src.main stats

# Dọn dẹp dữ liệu (có xác nhận)
python -m src.main clean

# Dọn dẹp không cần xác nhận
python -m src.main clean -y

# Dọn dẹp tất cả kể cả models
python -m src.main clean --all -y
```

---

## 📝 Tóm Tắt Nhanh

| Mục                  | Giá trị                                                            |
| -------------------- | ------------------------------------------------------------------ |
| **Model Embedding**  | `Alibaba-NLP/gte-multilingual-base` (768 dimensions, 70+ ngôn ngữ) |
| **Device**           | **Auto-detect** - Tự động chọn GPU nếu có                          |
| **Vector Backend**   | **FAISS** (mặc định, nhanh) hoặc ChromaDB                          |
| **Image Extraction** | **PyMuPDF** - smart cropping, clustering vector drawings           |
| **Chunk Size**       | 1024 ký tự, overlap 100                                            |
| **Top-K Search**     | 5 kết quả, threshold 0.5                                           |

---

## 🔧 Cách Thay Đổi Cấu Hình

Mở file `config/settings.py` và sửa giá trị trong các class config tương ứng.

**Ví dụ 1:** Chuyển sang ChromaDB:

```python
class VectorDBConfig(BaseModel):
    backend: str = "chromadb"  # Đổi từ "faiss"
```

**Ví dụ 2:** Đổi loại FAISS index:

```python
class VectorDBConfig(BaseModel):
    faiss_index_type: str = "HNSW"  # Đổi từ "Flat"
```

**Ví dụ 3:** Tăng chunk size:

```python
class ChunkingConfig(BaseModel):
    chunk_size: int = 2048  # Đổi từ 1024
```

**Ví dụ 4:** Buộc dùng CPU (dù có GPU):

```python
class ModelConfig(BaseModel):
    force_cpu: bool = True  # Buộc dùng CPU
```

---

_Tài liệu này được tạo tự động để tham khảo nhanh._
