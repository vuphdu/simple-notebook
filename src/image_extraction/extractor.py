"""
Image Extraction Module using PaddleOCR PP-Structure

This module provides functionality to extract image regions from documents
using PaddleOCR's PP-Structure for layout analysis.
"""
from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field
import hashlib
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings, ensure_directories, EXTRACTED_IMAGES_DIR
from PIL import Image, ImageOps
import io
import numpy as np


@dataclass
class ExtractedImage:
    """Represents an extracted image region from a document."""
    image_id: str
    image_path: str
    source_file: str
    page_number: int
    bbox: tuple  # (x1, y1, x2, y2)
    region_type: str  # "figure", "table", "chart", etc.
    surrounding_text: str = ""
    caption: str = ""
    description: str = ""
    
    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "bbox": self.bbox,
            "region_type": self.region_type,
            "surrounding_text": self.surrounding_text,
            "caption": self.caption,
            "description": self.description,
        }
    
    def get_vectorization_text(self) -> str:
        """Generate text for vectorization."""
        parts = []
        if self.caption:
            parts.append(f"Caption: {self.caption}")
        if self.description:
            parts.append(f"Description: {self.description}")
        if self.surrounding_text:
            parts.append(f"Context: {self.surrounding_text}")
        parts.append(f"Type: {self.region_type}")
        parts.append(f"Source: {Path(self.source_file).name}, Page {self.page_number}")
        return "\n".join(parts)


@dataclass  
class ImageExtractionConfig:
    """Configuration for image extraction."""
    output_dir: Path = field(default_factory=lambda: EXTRACTED_IMAGES_DIR)
    image_format: str = "png"
    min_width: int = 50  # Minimum width to consider as valid image
    min_height: int = 50  # Minimum height
    min_brightness: float = 20.0  # Minimum mean brightness (0-255) to avoid black images
    min_std_dev: float = 10.0  # Minimum standard deviation to avoid solid color images
    extract_tables: bool = True
    extract_figures: bool = True
    context_chars: int = 500  # Number of characters to extract around image
    use_gpu: bool = False
    
    # Clustering config
    cluster_merge_distance: int = 50  # Pixels to merge nearby drawings
    min_drawing_size: int = 20  # Minimum size for a drawing path
    min_diagram_size: int = 100  # Minimum size for a clustered diagram


class ImageExtractor:
    """
    Extracts image regions from documents using PaddleOCR PP-Structure.
    
    PP-Structure performs layout analysis to identify:
    - Figures/Images
    - Tables
    - Charts
    - Text blocks
    """
    
    def __init__(self, config: Optional[ImageExtractionConfig] = None):
        """
        Initialize the ImageExtractor.
        
        Args:
            config: Image extraction configuration.
        """
        self.config = config or ImageExtractionConfig()
        self._structure_engine = None
        ensure_directories()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def structure_engine(self):
        """Lazy loading of PP-Structure engine."""
        if self._structure_engine is None:
            self._structure_engine = self._init_structure_engine()
        return self._structure_engine
    
    def _init_structure_engine(self):
        """Initialize the PP-Structure engine."""
        try:
            from paddleocr import PPStructureV3
            
            print("Initializing PP-StructureV3 engine...")
            engine = PPStructureV3(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
            print("PP-StructureV3 engine initialized successfully")
            return engine
        except ImportError:
            raise ImportError(
                "PaddleOCR is required for image extraction. "
                "Install with: pip install paddleocr[all]"
            )
    
    def extract_from_pdf(self, pdf_path: Path) -> list[ExtractedImage]:
        """
        Extract images from a PDF file.
        
        Uses PyMuPDF (fitz) as primary method, falls back to pdf2image if needed.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            List of ExtractedImage objects.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        print(f"Extracting images from: {pdf_path.name}")
        
        # 1. Extract embedded images (bitmaps)
        extracted_images = self._extract_pdf_with_pymupdf(pdf_path)
        
        # 2. Extract vector drawings (diagrams)
        if self.config.extract_figures:
            drawings = self._extract_drawings_with_pymupdf(pdf_path)
            extracted_images.extend(drawings)
        
        print(f"  Extracted {len(extracted_images)} images/drawings from {pdf_path.name}")
        return extracted_images
    
    def _extract_pdf_with_pymupdf(self, pdf_path: Path) -> list[ExtractedImage]:
        """Extract images using PyMuPDF (fitz) - primary method."""
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(str(pdf_path))
            extracted_images = []
            
            print(f"  Using PyMuPDF, {len(doc)} pages")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get images directly from PDF
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        
                        # Use Pixmap to handle color spaces and masks correctly
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Check image size
                        if pix.width < self.config.min_width or pix.height < self.config.min_height:
                            pix = None
                            continue
                            
                        # Handle alpha channel / color space
                        # If CMYK or grayscale with alpha, convert to RGB
                        if pix.n - pix.alpha < 3: 
                            new_pix = fitz.Pixmap(fitz.csRGB, pix)
                            pix = new_pix
                        
                        # Store dimensions before clearing pix
                        width = pix.width
                        height = pix.height
                            
                        # Generate image ID
                        # Validate image content (check for black/empty images)
                        # Pass the pixmap directly to avoid PIL overhead if possible
                        is_valid, _, _ = self._is_valid_pixmap(pix)
                        
                        if not is_valid:
                            pix = None
                            continue
                            
                        image_data = pix.tobytes("png")
                        image_id = hashlib.md5(image_data).hexdigest()[:8]
                        image_filename = f"{pdf_path.stem}_p{page_num + 1}_img{img_index}_{image_id}.png"
                        image_path = self.config.output_dir / image_filename
                        
                        # Save image
                        pix.save(str(image_path))
                        pix = None  # Free memory
                        
                        # Get surrounding text
                        text = page.get_text()
                        
                        extracted_images.append(ExtractedImage(
                            image_id=image_id,
                            image_path=str(image_path),
                            source_file=str(pdf_path),
                            page_number=page_num + 1,
                            bbox=(0, 0, width, height),
                            region_type="figure",
                            surrounding_text=text[:self.config.context_chars],
                            caption="",
                            description=f"Image from {pdf_path.name}, page {page_num + 1}"
                        ))
                        
                        print(f"    Found image on page {page_num + 1}: {width}x{height}")
                    except Exception as e:
                        print(f"    Error extracting image {img_index} on page {page_num + 1}: {e}")
            
            doc.close()
            return extracted_images
            
        except ImportError:
            print("  PyMuPDF not available")
            return []
        except Exception as e:
            print(f"  Error with PyMuPDF: {e}")
            return []

    def _extract_drawings_with_pymupdf(self, pdf_path: Path) -> list[ExtractedImage]:
        """Extract vector drawings (diagrams) by rendering drawing paths."""
        extracted_drawings = []
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            
            print(f"  Scanning for vector drawings...")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                drawings = page.get_drawings()
                
                if not drawings:
                    continue
                    
                # Filter small drawings and group them
                valid_rects = []
                for d in drawings:
                    rect = d["rect"]
                    # Filter out very small paths (e.g. bullets, table lines)
                    if rect.width > 20 or rect.height > 20:
                        valid_rects.append(rect)
                
                if not valid_rects:
                    continue
                
                # Cluster rects to find diagram regions
                # Optimized approach: Sort by position and single-pass merge
                
                # 1. Sort by Y then X
                valid_rects.sort(key=lambda r: (r.y0, r.x0))
                
                clusters = []
                if valid_rects:
                    current_cluster = fitz.Rect(valid_rects[0])
                    
                    for rect in valid_rects[1:]:
                        # Check if rect is close to current cluster
                        # Expand current cluster by merge distance for check
                        expanded = fitz.Rect(current_cluster)
                        expanded.x0 -= self.config.cluster_merge_distance
                        expanded.y0 -= self.config.cluster_merge_distance
                        expanded.x1 += self.config.cluster_merge_distance
                        expanded.y1 += self.config.cluster_merge_distance
                        
                        if rect.intersects(expanded):
                            current_cluster.include_rect(rect)
                        else:
                            # Finish current cluster and start new one
                            clusters.append(current_cluster)
                            current_cluster = fitz.Rect(rect)
                    
                    # Append the last cluster
                    clusters.append(current_cluster)
                
                # Process clusters
                for i, rect in enumerate(clusters):
                    # Check if cluster is big enough to be a diagram
                    if rect.width < self.config.min_diagram_size or rect.height < self.config.min_diagram_size:
                        continue
                        
                    # Render the region
                    # Use higher DPI for better quality
                    try:
                        pix = page.get_pixmap(clip=rect, dpi=150)
                        
                        # Check validity using optimized check
                        is_valid, _, _ = self._is_valid_pixmap(pix)
                        
                        if not is_valid:
                            pix = None
                            continue
                        
                        image_data = pix.tobytes("png")
                        # Save
                        image_id = hashlib.md5(image_data).hexdigest()[:8]
                        image_filename = f"{pdf_path.stem}_p{page_num + 1}_drawing{i}_{image_id}.png"
                        image_path = self.config.output_dir / image_filename
                        
                        pix.save(str(image_path))
                        
                        # Get text
                        text = page.get_text(clip=rect)
                        
                        extracted_drawings.append(ExtractedImage(
                            image_id=image_id,
                            image_path=str(image_path),
                            source_file=str(pdf_path),
                            page_number=page_num + 1,
                            bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                            region_type="drawing",
                            surrounding_text=text[:self.config.context_chars],
                            caption="",
                            description=f"Drawing from {pdf_path.name}, page {page_num + 1}"
                        ))
                        pix = None
                        
                    except Exception as e:
                        print(f"    Error rendering drawing on page {page_num + 1}: {e}")

            doc.close()
        except Exception as e:
            print(f"Error extracting drawings: {e}")
            
        return extracted_drawings
            

    
    def _extract_pdf_with_pdf2image(self, pdf_path: Path) -> list[ExtractedImage]:
        """Extract images using pdf2image and PP-Structure."""
        try:
            from pdf2image import convert_from_path
            
            extracted_images = []
            pages = convert_from_path(str(pdf_path), dpi=150)
            
            for page_num, page_image in enumerate(pages, start=1):
                print(f"  Processing page {page_num}/{len(pages)}...")
                
                # Save page as temporary image
                temp_path = self.config.output_dir / f"_temp_page_{page_num}.png"
                page_image.save(str(temp_path))
                
                # Extract images from this page
                page_images = self._extract_from_image(
                    temp_path,
                    source_file=str(pdf_path),
                    page_number=page_num
                )
                extracted_images.extend(page_images)
                
                # Clean up temp file
                temp_path.unlink(missing_ok=True)
            
            return extracted_images
                
        except ImportError:
            print("  pdf2image not available. Install poppler and pdf2image.")
            return []
        except Exception as e:
            print(f"  pdf2image error: {e}")
            return []
    
    def _extract_from_image(
        self,
        image_path: Path,
        source_file: str,
        page_number: int = 1
    ) -> list[ExtractedImage]:
        """
        Extract image regions from a single image using PP-Structure.
        
        Args:
            image_path: Path to the image.
            source_file: Original source file path.
            page_number: Page number in the source document.
            
        Returns:
            List of ExtractedImage objects.
        """
        extracted_images = []
        
        try:
            from PIL import Image
            import numpy as np
            
            # Run PP-Structure analysis
            results = self.structure_engine.predict(input=str(image_path))
            
            # Load original image for cropping
            original_image = Image.open(image_path)
            
            # Process results
            for result in results:
                # Get layout parsing result
                if hasattr(result, 'layout_parsing_result'):
                    layout_result = result.layout_parsing_result
                elif isinstance(result, dict) and 'layout_parsing_result' in result:
                    layout_result = result['layout_parsing_result']
                else:
                    continue
                
                # Extract text content for context
                all_text = self._extract_text_from_result(result)
                
                # Process each detected region
                if isinstance(layout_result, dict):
                    boxes = layout_result.get('boxes', [])
                    for box_idx, box in enumerate(boxes):
                        region_type = box.get('type', 'unknown')
                        bbox = box.get('bbox', box.get('coordinate', []))
                        
                        # Filter by region type
                        if region_type in ['figure', 'image', 'chart', 'picture']:
                            if not self.config.extract_figures:
                                continue
                        elif region_type == 'table':
                            if not self.config.extract_tables:
                                continue
                        else:
                            continue  # Skip text regions
                        
                        if len(bbox) >= 4:
                            # Crop and save the image region
                            cropped = self._crop_region(original_image, bbox)
                            
                            if cropped and self._is_valid_size(cropped):
                                image_id = hashlib.md5(
                                    f"{source_file}_{page_number}_{box_idx}".encode()
                                ).hexdigest()[:8]
                                
                                image_filename = f"{Path(source_file).stem}_p{page_number}_{region_type}_{image_id}.{self.config.image_format}"
                                save_path = self.config.output_dir / image_filename
                                cropped.save(str(save_path))
                                
                                # Get nearby text as context
                                nearby_text = self._get_nearby_text(box, layout_result)
                                
                                extracted_images.append(ExtractedImage(
                                    image_id=image_id,
                                    image_path=str(save_path),
                                    source_file=source_file,
                                    page_number=page_number,
                                    bbox=tuple(bbox[:4]),
                                    region_type=region_type,
                                    surrounding_text=nearby_text or all_text[:self.config.context_chars],
                                    caption=box.get('caption', ''),
                                    description=f"{region_type} from page {page_number}"
                                ))
            
        except Exception as e:
            print(f"Error extracting from image: {e}")
        
        return extracted_images
    
    def _crop_region(self, image, bbox) -> Optional["Image.Image"]:
        """Crop a region from an image."""
        try:
            from PIL import Image
            
            # Handle different bbox formats
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
            elif len(bbox) == 8:  # Polygon format
                x1 = min(bbox[0], bbox[2], bbox[4], bbox[6])
                y1 = min(bbox[1], bbox[3], bbox[5], bbox[7])
                x2 = max(bbox[0], bbox[2], bbox[4], bbox[6])
                y2 = max(bbox[1], bbox[3], bbox[5], bbox[7])
            else:
                return None
            
            # Ensure valid coordinates
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(image.width, int(x2)), min(image.height, int(y2))
            
            if x2 > x1 and y2 > y1:
                return image.crop((x1, y1, x2, y2))
            return None
            
        except Exception:
            return None
    
    def _is_valid_size(self, image) -> bool:
        """Check if image meets minimum size requirements."""
        return image.width >= self.config.min_width and image.height >= self.config.min_height
    
    def _extract_text_from_result(self, result) -> str:
        """Extract all text content from PP-Structure result."""
        texts = []
        
        try:
            if hasattr(result, 'layout_parsing_result'):
                layout = result.layout_parsing_result
            elif isinstance(result, dict):
                layout = result.get('layout_parsing_result', {})
            else:
                return ""
            
            if isinstance(layout, dict):
                boxes = layout.get('boxes', [])
                for box in boxes:
                    if box.get('type') in ['text', 'title', 'paragraph']:
                        texts.append(box.get('text', ''))
        except Exception:
            pass
        
        return " ".join(texts)
    
    def _get_nearby_text(self, target_box: dict, layout_result: dict) -> str:
        """Get text from regions near the target box."""
        if not isinstance(layout_result, dict):
            return ""
        
        target_bbox = target_box.get('bbox', target_box.get('coordinate', []))
        if len(target_bbox) < 4:
            return ""
        
        target_center_y = (target_bbox[1] + target_bbox[3]) / 2
        
        nearby_texts = []
        boxes = layout_result.get('boxes', [])
        
        for box in boxes:
            if box.get('type') in ['text', 'title', 'paragraph', 'caption']:
                box_bbox = box.get('bbox', box.get('coordinate', []))
                if len(box_bbox) >= 4:
                    box_center_y = (box_bbox[1] + box_bbox[3]) / 2
                    
                    # Check if text is near (within 100 pixels vertically)
                    if abs(box_center_y - target_center_y) < 200:
                        text = box.get('text', '')
                        if text:
                            nearby_texts.append(text)
        
        return " ".join(nearby_texts[:3])  # Limit to 3 nearby text blocks
    
    def extract_from_file(self, file_path: Path) -> list[ExtractedImage]:
        """
        Extract images from a file (PDF or image).
        
        Args:
            file_path: Path to the file.
            
        Returns:
            List of ExtractedImage objects.
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self.extract_from_pdf(file_path)
        elif suffix in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            return self._extract_from_image(
                file_path,
                source_file=str(file_path),
                page_number=1
            )
        else:
            print(f"Unsupported file type: {suffix}")
            return []
    
    def extract_from_directory(
        self,
        directory: Path,
        recursive: bool = True
    ) -> list[ExtractedImage]:
        """
        Extract images from all documents in a directory.
        
        Args:
            directory: Path to the directory.
            recursive: Whether to process subdirectories.
            
        Returns:
            List of all ExtractedImage objects.
        """
        directory = Path(directory)
        all_images = []
        
        pattern = '**/*' if recursive else '*'
        extensions = ['.pdf', '.png', '.jpg', '.jpeg']
        
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    images = self.extract_from_file(file_path)
                    all_images.extend(images)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        return all_images
    
    def _is_valid_pixmap(self, pix) -> tuple[bool, str, dict]:
        """
        Check if pixmap is valid using direct buffer access (faster than PIL).
        """
        stats = {"brightness": 0, "std_dev": 0}
        try:
            # Get raw samples
            # pix.samples is a bytes object
            # pix.n is number of channels (e.g. 3 for RGB, 4 for RGBA)
            
            # Convert to numpy array
            # Note: This is a view, no copy if possible, but safe to copy for stats
            samples = np.frombuffer(pix.samples, dtype=np.uint8)
            
            if pix.n >= 3:
                # Reshape to (H, W, C)
                # We only care about brightness, so we can just average all channels or take a stride
                # For speed, let's just use the buffer directly.
                # If RGB, mean of all bytes is roughly mean brightness (approximation)
                # But to be accurate to PIL's convert('L'), we should do weighted sum.
                # However, for "is it black?", simple mean is enough.
                pass
            
            # Simple statistical check on the whole buffer
            # This treats R, G, B, A equally, which is a rough approximation but fast
            mean_val = np.mean(samples)
            std_val = np.std(samples)
            
            stats["brightness"] = mean_val
            stats["std_dev"] = std_val
            
            # Thresholds might need slight adjustment compared to PIL 'L' mode
            # but usually close enough for "black vs content"
            
            if mean_val < self.config.min_brightness:
                return False, f"Too dark (mean={mean_val:.2f})", stats
                
            if std_val < self.config.min_std_dev:
                return False, f"Low variance (std={std_val:.2f})", stats
                
            return True, "OK", stats
            
        except Exception as e:
            # Fallback to PIL method if something fails
            print(f"Error in fast validation: {e}")
            return self._is_valid_image(pix.tobytes("png"))

    def _is_valid_image(self, image_data: bytes) -> tuple[bool, str, dict]:
        """
        Check if image is valid (not too dark, not solid color).
        
        Args:
            image_data: Image data in bytes (PNG format).
            
        Returns:
            Tuple (is_valid, reason, stats_dict)
        """
        stats = {"brightness": 0, "std_dev": 0}
        try:
            with Image.open(io.BytesIO(image_data)) as img:
                # Convert to grayscale for analysis
                gray = img.convert('L')
                np_img = np.array(gray)
                
                # Calculate statistics
                mean_brightness = np.mean(np_img)
                std_dev = np.std(np_img)
                
                stats["brightness"] = mean_brightness
                stats["std_dev"] = std_dev
                
                # Check brightness (avoid black images)
                if mean_brightness < self.config.min_brightness:
                    return False, f"Too dark (brightness={mean_brightness:.2f} < {self.config.min_brightness})", stats
                
                # Check variance (avoid solid color images)
                if std_dev < self.config.min_std_dev:
                    return False, f"Low variance (std_dev={std_dev:.2f} < {self.config.min_std_dev})", stats
                    
                return True, "OK", stats
        except Exception as e:
            return False, f"Error: {e}", stats

    def get_vectorization_data(self, images: list[ExtractedImage]) -> list[dict]:
        """
        Prepare extracted images for vectorization.
        
        Args:
            images: List of ExtractedImage objects.
            
        Returns:
            List of dicts ready for vectorization.
        """
        data = []
        for img in images:
            data.append({
                "chunk_id": f"img_{img.image_id}",
                "content": img.get_vectorization_text(),
                "metadata": {
                    "type": "extracted_image",
                    "region_type": img.region_type,
                    "image_path": img.image_path,
                    "source_file": img.source_file,
                    "page_number": img.page_number,
                    "bbox": str(img.bbox),
                },
                "source_file": img.source_file
            })
        return data


# Global extractor instance
_extractor: Optional[ImageExtractor] = None


def get_image_extractor() -> ImageExtractor:
    """Get or create the global image extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = ImageExtractor()
    return _extractor
