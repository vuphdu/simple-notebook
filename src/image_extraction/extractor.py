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
    bbox: tuple  # (x0, y0, x1, y1)
    region_type: str  # "smart_crop"
    surrounding_text: str = ""
    caption: str = ""
    description: str = ""
    # OlmOCR-style fields
    anchor_text: str = ""           # Positional text context with coordinates
    is_table: bool = False          # Majority content is tabular
    is_diagram: bool = False        # Majority content is diagram/figure
    page_dimensions: tuple = (0, 0) # (width, height) of the page
    
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
            "anchor_text": self.anchor_text,
            "is_table": self.is_table,
            "is_diagram": self.is_diagram,
            "page_dimensions": self.page_dimensions,
        }
    
    def get_vectorization_text(self) -> str:
        """
        Generate rich text for vectorization with OlmOCR-style context.
        
        Format optimized for semantic search with:
        - Type prefix for filtering
        - Position information
        - Classification metadata
        - Rich contextual text
        """
        parts = []
        
        # Type prefix for easy identification and filtering
        content_type = "DIAGRAM" if self.is_diagram else "TABLE" if self.is_table else "IMAGE"
        parts.append(f"[{content_type}] {Path(self.source_file).name} - Page {self.page_number}")
        
        # Position information (OlmOCR style)
        if self.page_dimensions != (0, 0):
            parts.append(f"Page: {self.page_dimensions[0]:.0f}x{self.page_dimensions[1]:.0f}")
        parts.append(f"Position: ({self.bbox[0]:.0f}, {self.bbox[1]:.0f}) to ({self.bbox[2]:.0f}, {self.bbox[3]:.0f})")
        
        # Caption is highest priority (if available)
        if self.caption:
            parts.append(f"Caption: {self.caption}")
        
        # Description
        default_desc = f"Smart crop from {Path(self.source_file).name}, page {self.page_number}"
        if self.description and self.description != default_desc:
            parts.append(f"Description: {self.description}")
        
        # Anchor text with positions (OlmOCR-style rich context)
        if self.anchor_text:
            # Truncate if too long but keep structure
            anchor = self.anchor_text[:1500] if len(self.anchor_text) > 1500 else self.anchor_text
            parts.append(f"Layout:\n{anchor}")
        elif self.surrounding_text:
            # Fallback to surrounding text
            context = self.surrounding_text[:800] if len(self.surrounding_text) > 800 else self.surrounding_text
            parts.append(f"Context: {context}")
        
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
    Extracts image regions from PDF documents.
    
    Supports two modes:
    - pymu: PyMuPDF smart cropping (detects and crops diagrams/figures)
    - fullpage: Full-page rendering at 150% zoom when images detected
    
    Features:
    - Detects embedded bitmap images
    - Clusters vector drawings into diagrams (pymu mode)
    - Merges nearby elements (pymu mode)
    - Captures surrounding text as context
    """
    
    def __init__(self, config: Optional[ImageExtractionConfig] = None, mode: str = "pymu"):
        """
        Initialize the ImageExtractor.
        
        Args:
            config: Image extraction configuration.
            mode: Extraction mode - "pymu" for smart cropping or "fullpage" for page rendering
        """
        self.config = config or ImageExtractionConfig()
        self.mode = mode
        ensure_directories()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        if fitz is None:
            print("Warning: PyMuPDF (fitz) is not installed. Image extraction will not work.")
            print("Install with: pip install pymupdf")

    def extract_from_file(self, pdf_path: Path) -> list[ExtractedImage]:
        """
        Extract images from a PDF file based on configured mode.
        
        - pymu mode: Smart cropping of diagrams/figures  
        - fullpage mode: Full-page rendering at 150% zoom
        
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
        
        # Route to appropriate extraction method
        if self.mode == "fullpage":
            return self._extract_full_pages(pdf_path)
        else:  # Default to pymu mode
            return self.extract_from_pdf(pdf_path)
    
    # Backwards compatibility alias - DO NOT call extract_from_file from here!
    def extract_from_pdf(self, pdf_path: Path) -> list[ExtractedImage]:
        """
        Extract images using PyMuPDF smart cropping.
        
        This is the original smart cropping implementation.
        For mode-based routing, use extract_from_file() instead.
        """
        if fitz is None:
            print("  PyMuPDF not available. Install with: pip install pymupdf")
            return []
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        print(f"Extracting images from: {pdf_path.name}")
        # Call the actual smart cropping implementation
        return self._extract_smart_crops(pdf_path)


    def _get_anchor_text(self, page, image_rect: fitz.Rect, max_length: int = 2000) -> str:
        """
        Generate OlmOCR-style anchor text with positional information.
        
        This provides structural hints to help VLMs understand document layout.
        Format: [x_coord x y_coord]text_content
        
        Args:
            page: PyMuPDF page object
            image_rect: Bounding box of the image region
            max_length: Maximum length of anchor text
            
        Returns:
            Anchor text string with positional information
        """
        page_width = page.rect.width
        page_height = page.rect.height
        
        result = f"Page dimensions: {page_width:.0f}x{page_height:.0f}\n"
        result += f"[Image {image_rect.x0:.0f}x{image_rect.y0:.0f} to {image_rect.x1:.0f}x{image_rect.y1:.0f}]\n"
        
        # Get text blocks with positions
        try:
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        except Exception:
            return result
        
        text_elements = []
        for block in blocks:
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text and len(text) > 2:  # Skip very short text
                            bbox = span.get("bbox", [0, 0, 0, 0])
                            x, y = bbox[0], bbox[1]
                            text_elements.append((x, y, text))
        
        # Sort by position (top to bottom, left to right in PDF coords)
        # Note: PDF y-axis is bottom-up, but get_text returns top-down
        text_elements.sort(key=lambda t: (t[1], t[0]))
        
        # Build anchor text with coordinates
        current_length = len(result)
        for x, y, text in text_elements:
            line = f"[{x:.0f}x{y:.0f}]{text}\n"
            if current_length + len(line) > max_length:
                break
            result += line
            current_length += len(line)
        
        return result
    
    def _detect_content_type(self, page, rect: fitz.Rect) -> dict:
        """
        Detect if the region contains primarily table or diagram content.
        
        Args:
            page: PyMuPDF page object
            rect: Region bounding box
            
        Returns:
            dict with is_table and is_diagram flags
        """
        result = {"is_table": False, "is_diagram": False}
        
        try:
            # Get text in the region
            text = page.get_text(clip=rect)
            
            # Table detection heuristics
            # Tables often have: multiple "|" chars, aligned columns, grid patterns
            lines = text.split('\n')
            pipe_count = text.count('|')
            tab_count = text.count('\t')
            
            # Check for table-like patterns
            if pipe_count > 5 or tab_count > 10:
                result["is_table"] = True
            
            # Check for aligned numeric columns (common in tables)
            numeric_lines = sum(1 for line in lines if any(c.isdigit() for c in line))
            if len(lines) > 3 and numeric_lines / max(len(lines), 1) > 0.5:
                result["is_table"] = True
            
            # Diagram detection heuristics
            # Diagrams have: many vector drawings, less text, specific keywords
            drawings = page.get_drawings()
            drawings_in_rect = [d for d in drawings if fitz.Rect(d["rect"]).intersects(rect)]
            
            text_length = len(text.strip())
            drawing_count = len(drawings_in_rect)
            
            # High drawing density with low text = likely diagram
            if drawing_count > 10 and text_length < 500:
                result["is_diagram"] = True
            
            # Check for diagram keywords in surrounding text
            diagram_keywords = ["figure", "fig.", "diagram", "chart", "flow", "schema", "architecture"]
            text_lower = text.lower()
            if any(kw in text_lower for kw in diagram_keywords):
                result["is_diagram"] = True
                
        except Exception:
            pass
        
        return result

    def _extract_smart_crops(self, pdf_path: Path) -> list[ExtractedImage]:
        """
        Smartly detect and crop figures/diagrams from PDF pages using PyMuPDF.
        Enhanced with OlmOCR-style anchor text and content classification.
        """
        try:
            doc = fitz.open(str(pdf_path))
            extracted_images = []
            
            print(f"  Smart scanning {len(doc)} pages...")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_images = []
                page_dims = (page.rect.width, page.rect.height)
                
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
                
                # D. Render and Save with OlmOCR enhancements
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
                        
                        # Get context text (caption/surrounding) - EXPANDED WINDOW
                        context_rect = fitz.Rect(rect)
                        context_rect.y1 += 150  # Look 150px below for caption
                        context_rect.y0 -= 80   # Look 80px above for title
                        context_rect.x0 -= 20   # Expand horizontally for full text
                        context_rect.x1 += 20
                        surrounding_text = page.get_text(clip=context_rect)
                        
                        # OlmOCR Enhancement: Get anchor text with positions
                        anchor_text = self._get_anchor_text(page, rect)
                        
                        # OlmOCR Enhancement: Detect content type
                        content_type = self._detect_content_type(page, rect)
                        
                        extracted_images.append(ExtractedImage(
                            image_id=image_id,
                            image_path=str(image_path),
                            source_file=str(pdf_path),
                            page_number=page_num + 1,
                            bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                            region_type="smart_crop",
                            surrounding_text=surrounding_text,
                            caption="",
                            description=f"Smart crop from {pdf_path.name}, page {page_num + 1}",
                            # OlmOCR fields
                            anchor_text=anchor_text,
                            is_table=content_type["is_table"],
                            is_diagram=content_type["is_diagram"],
                            page_dimensions=page_dims,
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

    def _extract_full_pages(self, pdf_path: Path) -> list[ExtractedImage]:
        """
        Extract full-page renders for pages containing images.
        
        Renders pages at 150% zoom (DPI=225) when images are detected.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of ExtractedImage objects for full-page renders
        """
        if fitz is None:
            return []
        
        extracted = []
        
        try:
            doc = fitz.open(str(pdf_path))
            
            for page_num, page in enumerate(doc):
                # Check if page has images
                images = page.get_images()
                if not images:
                    continue
                
                # Render at 150% zoom (DPI=225 from 150 DPI)
                mat = fitz.Matrix(1.5, 1.5)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # Save full page
                image_id = f"{pdf_path.stem}_p{page_num+1}_fullpage"
                image_path = self.config.output_dir / f"{image_id}.png"
                pix.save(str(image_path))
                
                # Get all page text as context
                page_text = page.get_text()
                
                # Get page dimensions
                page_dims = (page.rect.width, page.rect.height)
                
                extracted.append(ExtractedImage(
                    image_id=image_id,
                    image_path=str(image_path),
                    source_file=str(pdf_path),
                    page_number=page_num + 1,
                    bbox=(0, 0, page.rect.width, page.rect.height),
                    region_type="fullpage",
                    surrounding_text=page_text[:2000],  # Limit context length
                    description=f"Full page from {pdf_path.name}, page {page_num+1} (150% zoom)",
                    page_dimensions=page_dims
                ))
                
            doc.close()
            
        except Exception as e:
            print(f"Error extracting full pages from {pdf_path}: {e}")
        
        return extracted

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
_current_mode: Optional[str] = None


def get_image_extractor(mode: str = "pymu") -> ImageExtractor:
    """
    Get or create the global image extractor instance.
    
    Args:
        mode: Extraction mode - "pymu" or "fullpage"
        
    Returns:
        ImageExtractor instance configured for the specified mode
    """
    global _extractor, _current_mode
    
    # Recreate if mode changed or first time
    if _extractor is None or _current_mode != mode:
        _extractor = ImageExtractor(mode=mode)
        _current_mode = mode
    
    return _extractor
