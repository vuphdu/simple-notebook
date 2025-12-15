"""Image Extraction module using PaddleOCR PP-Structure."""
from .extractor import (
    ImageExtractor,
    ExtractedImage,
    ImageExtractionConfig,
    get_image_extractor,
)

__all__ = [
    # Image extraction
    "ImageExtractor",
    "ExtractedImage",
    "ImageExtractionConfig",
    "get_image_extractor",
]

