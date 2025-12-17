"""Image Extraction module using PyMuPDF or PaddleOCR PP-Structure."""
from .extractor import (
    ImageExtractor,
    ExtractedImage,
    ImageExtractionConfig,
    get_image_extractor,
    check_paddle_installation,
    print_paddle_status,
)

__all__ = [
    # Image extraction
    "ImageExtractor",
    "ExtractedImage",
    "ImageExtractionConfig",
    "get_image_extractor",
    # Paddle utilities
    "check_paddle_installation",
    "print_paddle_status",
]

