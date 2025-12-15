---
description: Extract images from PDF and documents using PP-Structure
---

# Extract Images Workflow

This workflow extracts images, figures, tables, and charts from documents using PaddleOCR PP-Structure.

## Prerequisites

```bash
# Install PaddleOCR dependencies
pip install paddlepaddle paddleocr PyMuPDF
```

## Commands

### Extract Images (Quick extraction using PyMuPDF)

```bash
# Extract from all documents
python -m src.main extract-images

# Extract from specific file
python -m src.main extract-images -i data/documents/input-doc/your_file.pdf

# Extract without vectorization
python -m src.main extract-images --no-vectorize
```

**Output:** `data/documents/extracted_images/`

## Search for Images

After extraction, you can search for images by their descriptions:

```bash
python -m src.main search "diagram flow chart"
python -m src.main search "table specification"
```

## Configuration

### Image Extraction Config

```python
# src/image_extraction/extractor.py
class ImageExtractionConfig:
    min_width: int = 50       # Minimum image width
    min_height: int = 50      # Minimum image height
```

## Cleanup

```bash
python -m src.main clean -y
```
