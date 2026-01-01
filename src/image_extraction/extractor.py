"""
Image Extraction Module

This module provides smart image extraction from PDF documents using PyMuPDF.
Extracts figures, diagrams, and embedded images using intelligent clustering.
"""
from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings, ensure_directories, EXTRACTED_IMAGES_DIR

try:
    import fitz
except ImportError:
    fitz = None


@dataclass
class ExtractedImage:
    """Represents an extracted image region from a document."""
    image_id: str
    image_path: str
    source_file: str
    page_number: int
    bbox: tuple  # (x1, y1, x2, y2)
    region_type: str  # "smart_crop"
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
    """Configuration for PyMuPDF smart image extraction."""
    output_dir: Path = field(default_factory=lambda: EXTRACTED_IMAGES_DIR)
    min_width: int = 50  # Minimum width to consider as valid image
    min_height: int = 50  # Minimum height
    
    # Clustering config for smart cropping
    cluster_merge_distance: int = 50  # Pixels to merge nearby drawings
    min_drawing_size: int = 20  # Minimum size for a drawing path
    min_diagram_size: int = 100  # Minimum size for a clustered diagram


class ImageExtractor:
    """
    Extracts image regions from PDF documents using PyMuPDF smart cropping.
    
    Features:
    - Detects embedded bitmap images
    - Clusters vector drawings into diagrams
    - Merges nearby elements
    - Captures surrounding text as context
    """
    
    def __init__(self, config: Optional[ImageExtractionConfig] = None):
        """
        Initialize the ImageExtractor with PyMuPDF smart cropping.
        
        Args:
            config: Image extraction configuration.
        """
        self.config = config or ImageExtractionConfig()
        ensure_directories()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        if fitz is None:
            print("Warning: PyMuPDF (fitz) is not installed. Image extraction will not work.")
            print("Install with: pip install pymupdf")

    def extract_from_pdf(self, pdf_path: Path) -> list[ExtractedImage]:
        """
        Extract images from a PDF file using smart cropping.
        
        Detects and extracts:
        - Embedded bitmap images
        - Vector drawings and diagrams
        - Figures and charts
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            List of ExtractedImage objects.
        """
        if fitz is None:
            print("  PyMuPDF not available. Install with: pip install pymupdf")
            return []

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
        print(f"Extracting images from: {pdf_path.name}")
        return self._extract_smart_crops(pdf_path)

    def _extract_smart_crops(self, pdf_path: Path) -> list[ExtractedImage]:
        """
        Smartly detect and crop figures/diagrams from PDF pages using PyMuPDF.
        """
        try:
            doc = fitz.open(str(pdf_path))
            extracted_images = []
            
            print(f"  Smart scanning {len(doc)} pages...")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_images = []
                
                # A. Get Bitmaps (Images)
                img_infos = page.get_image_info(xrefs=True)
                for img in img_infos:
                    bbox = fitz.Rect(img['bbox'])
                    if bbox.width >= self.config.min_width and bbox.height >= self.config.min_height:
                        page_images.append(bbox)

                # B. Get Vector Drawings (Clustered)
                drawing_clusters = self._cluster_drawings(page)
                page_images.extend(drawing_clusters)
                
                # C. Merge Overlapping/Nearby Regions
                merged_regions = self._merge_nearby_blocks(page_images)
                
                # D. Render and Save
                for i, rect in enumerate(merged_regions):
                    try:
                        # Render region
                        pix = page.get_pixmap(clip=rect, dpi=150)
                        
                        # Validate
                        if pix.width < self.config.min_width or pix.height < self.config.min_height:
                            continue
                            
                        image_data = pix.tobytes("png")
                        # Format: filename_p{page}_crop_{number}.png
                        image_filename = f"{pdf_path.stem}_p{page_num + 1}_crop_{i + 1}.png"
                        # Use filename (without extension) as ID for clarity
                        image_id = Path(image_filename).stem  # e.g., "HFP_v1.8_p14_crop_1"
                        image_path = self.config.output_dir / image_filename
                        
                        pix.save(str(image_path))
                        
                        # Get context text (caption/surrounding) - EXPANDED WINDOW (Phase 1)
                        context_rect = fitz.Rect(rect)
                        context_rect.y1 += 150  # Look 150px below for caption (was 50)
                        context_rect.y0 -= 80   # Look 80px above for title (was 20)
                        context_rect.x0 -= 20   # Expand horizontally for full text
                        context_rect.x1 += 20
                        text = page.get_text(clip=context_rect)
                        
                        extracted_images.append(ExtractedImage(
                            image_id=image_id,
                            image_path=str(image_path),
                            source_file=str(pdf_path),
                            page_number=page_num + 1,
                            bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                            region_type="smart_crop",
                            surrounding_text=text,
                            caption="",
                            description=f"Smart crop from {pdf_path.name}, page {page_num + 1}"
                        ))
                    except Exception as e:
                        print(f"    Error saving crop {i} on page {page_num + 1}: {e}")
            
            doc.close()
            print(f"  Extracted {len(extracted_images)} smart crops from {pdf_path.name}")
            return extracted_images
            
        except Exception as e:
            print(f"  Error in smart cropping: {e}")
            return []

    def _cluster_drawings(self, page) -> list[fitz.Rect]:
        """Cluster vector drawings into diagram regions."""
        drawings = page.get_drawings()
        if not drawings:
            return []
            
        # Filter valid drawing paths
        valid_rects = []
        for d in drawings:
            rect = d["rect"]
            if rect.width > 5 or rect.height > 5:
                valid_rects.append(rect)
        
        if not valid_rects:
            return []
            
        # Sort for clustering
        valid_rects.sort(key=lambda r: (r.y0, r.x0))
        
        clusters = []
        if valid_rects:
            current_cluster = fitz.Rect(valid_rects[0])
            
            for rect in valid_rects[1:]:
                # Expand cluster by merge distance
                expanded = fitz.Rect(current_cluster)
                dist = self.config.cluster_merge_distance
                expanded.x0 -= dist
                expanded.y0 -= dist
                expanded.x1 += dist
                expanded.y1 += dist
                
                if rect.intersects(expanded):
                    current_cluster.include_rect(rect)
                else:
                    clusters.append(current_cluster)
                    current_cluster = fitz.Rect(rect)
            clusters.append(current_cluster)
            
        # Filter small clusters
        final_clusters = []
        for c in clusters:
            if c.width >= self.config.min_diagram_size and c.height >= self.config.min_diagram_size:
                final_clusters.append(c)
                
        return final_clusters

    def _merge_nearby_blocks(self, rects: list[fitz.Rect]) -> list[fitz.Rect]:
        """Merge overlapping or nearby bounding boxes."""
        if not rects:
            return []
        
        merged = []
        sorted_rects = sorted(rects, key=lambda r: (r.y0, r.x0))
        
        current = fitz.Rect(sorted_rects[0])
        
        for rect in sorted_rects[1:]:
            # Check if close enough to merge
            expanded = fitz.Rect(current)
            dist = self.config.cluster_merge_distance
            expanded.x0 -= dist
            expanded.y0 -= dist
            expanded.x1 += dist
            expanded.y1 += dist
            
            if rect.intersects(expanded):
                current.include_rect(rect)
            else:
                merged.append(current)
                current = fitz.Rect(rect)
        
        merged.append(current)
        return merged

    def extract_from_file(self, file_path: Path) -> list[ExtractedImage]:
        """
        Extract images from a file (wrapper for PDF extraction).
        
        Args:
            file_path: Path to the file.
            
        Returns:
            List of ExtractedImage objects.
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.pdf':
            return self.extract_from_pdf(file_path)
        else:
            print(f"  Unsupported file type: {file_path.suffix}")
            return []

    def extract_from_directory(
        self,
        directory: Path,
        recursive: bool = True
    ) -> list[ExtractedImage]:
        """
        Extract images from all PDFs in a directory.
        
        Args:
            directory: Directory containing PDF files.
            recursive: Whether to search recursively.
            
        Returns:
            List of all extracted images.
        """
        directory = Path(directory)
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(directory.glob(pattern))
        
        all_images = []
        for pdf_file in pdf_files:
            images = self.extract_from_pdf(pdf_file)
            all_images.extend(images)
        
        return all_images

    def get_vectorization_data(self, images: list[ExtractedImage]) -> list[dict]:
        """
        Prepare extracted images for vectorization.
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
