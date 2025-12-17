"""
Image Extraction Module

This module provides functionality to extract images and figures from documents
using either PyMuPDF (Smart Cropping) or PaddleOCR (AI Layout Analysis).
"""
from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field
import hashlib
import io
import numpy as np
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings, ensure_directories, EXTRACTED_IMAGES_DIR
from PIL import Image

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
    region_type: str  # "smart_crop", "figure", "table", "full_page_image"
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
    
    # Clustering config
    cluster_merge_distance: int = 50  # Pixels to merge nearby drawings
    min_drawing_size: int = 20  # Minimum size for a drawing path
    min_diagram_size: int = 100  # Minimum size for a clustered diagram
    
    # Config for extraction strategy
    extract_full_page: bool = False  # Set to False to prefer smart cropping
    smart_crop: bool = True  # Enable smart cropping of figures/diagrams
    extraction_mode: str = "pymupdf"  # "pymupdf" or "paddle"
    use_gpu: bool = False


class ImageExtractor:
    """
    Extracts image regions from documents using PyMuPDF or PaddleOCR.
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
        
        if fitz is None:
            print("Warning: PyMuPDF (fitz) is not installed. Image extraction will not work.")
            print("Install with: pip install pymupdf")

    def extract_from_pdf(self, pdf_path: Path) -> list[ExtractedImage]:
        """
        Extract images from a PDF file.
        
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
            
        print(f"Extracting images from: {pdf_path.name} (Mode: {self.config.extraction_mode})")
        
        # Mode Selection
        if self.config.extraction_mode == "paddle":
            return self._extract_with_paddle(pdf_path)
            
        # Default: PyMuPDF Smart Cropping
        if self.config.smart_crop:
            return self._extract_smart_crops(pdf_path)
            
        # Fallback: Full Page
        if self.config.extract_full_page:
            return self._extract_pages_with_images_pymupdf(pdf_path)
            
        return []

    def _extract_with_paddle(self, pdf_path: Path) -> list[ExtractedImage]:
        """
        Extract images using PaddleOCR PP-Structure via subprocess.
        
        Runs PaddleOCR in a separate process to avoid DLL conflicts with PyTorch
        and to isolate potential crashes from oneDNN/MKL-DNN backends.
        """
        import subprocess
        import json
        import tempfile
        import os
        import threading
        
        worker_script = Path(__file__).parent / "paddle_worker.py"
        if not worker_script.exists():
            print("  Error: paddle_worker.py not found")
            return []
        
        print(f"  PaddleOCR extraction starting (isolated subprocess)...")
        print(f"  Note: This may take a while on first run (model download)")
        
        # Create temp file for JSON output
        fd, json_output_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        
        try:
            # Build command
            cmd = [
                sys.executable,
                str(worker_script),
                "--pdf", str(pdf_path),
                "--output", str(self.config.output_dir),
                "--json-output", str(json_output_path)
            ]
            if self.config.use_gpu:
                cmd.append("--gpu")
            
            # Create isolated environment to prevent DLL conflicts
            env = os.environ.copy()
            
            # Disable oneDNN/MKL-DNN which causes crashes on Windows
            env["FLAGS_use_mkldnn"] = "0"
            env["FLAGS_use_gpu"] = "0"
            env["MKLDNN_VERBOSE"] = "0"
            env["DNNL_VERBOSE"] = "0"
            env["PADDLE_WITH_MKLDNN"] = "0"
            
            # Limit threads to prevent resource conflicts
            env["OMP_NUM_THREADS"] = "2"
            env["MKL_NUM_THREADS"] = "2"
            
            # Suppress excessive logging
            env["GLOG_v"] = "0"
            env["GLOG_logtostderr"] = "0"
            env["GLOG_minloglevel"] = "3"
            env["TF_CPP_MIN_LOG_LEVEL"] = "3"
            
            # For newer PaddlePaddle versions
            env["FLAGS_enable_pir_in_executor"] = "0"
            
            # Function to print stderr in real-time
            def print_stderr(pipe):
                for line in iter(pipe.readline, ''):
                    if line.strip():
                        print(f"  {line.strip()}")
                pipe.close()
            
            # Run subprocess with timeout
            # Timeout: 5 minutes per PDF (adjust as needed)
            timeout_seconds = 300
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            
            # Stream stderr output in a separate thread
            stderr_thread = threading.Thread(target=print_stderr, args=(process.stderr,))
            stderr_thread.daemon = True
            stderr_thread.start()
            
            try:
                stdout, _ = process.communicate(timeout=timeout_seconds)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                print(f"  PaddleOCR worker timed out after {timeout_seconds}s")
                self._cleanup_temp_file(json_output_path)
                return []
            
            # Wait for stderr thread to finish
            stderr_thread.join(timeout=1)
            
            if returncode != 0:
                print(f"  PaddleOCR worker exited with code {returncode}")
                # Try to read partial results anyway
                if not os.path.exists(json_output_path):
                    self._cleanup_temp_file(json_output_path)
                    return []
            
            # Parse output from file
            if not os.path.exists(json_output_path):
                print(f"  PaddleOCR output file not found")
                return []
                
            try:
                with open(json_output_path, "r", encoding="utf-8") as f:
                    output_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  Failed to parse PaddleOCR output: {e}")
                self._cleanup_temp_file(json_output_path)
                return []
            finally:
                self._cleanup_temp_file(json_output_path)
            
            # Check for error response
            if isinstance(output_data, dict) and "error" in output_data:
                error_msg = output_data['error']
                # Provide helpful error messages
                if "paddleocr" in error_msg.lower() or "paddle" in error_msg.lower():
                    print(f"  PaddleOCR error: {error_msg[:200]}")
                    print(f"  Tip: Try 'pip install paddlepaddle paddleocr' or use --mode pymupdf")
                else:
                    print(f"  PaddleOCR error: {error_msg[:500]}")
                return []
            
            # Convert to ExtractedImage objects
            extracted_images = []
            for item in output_data:
                try:
                    extracted_images.append(ExtractedImage(
                        image_id=item["image_id"],
                        image_path=item["image_path"],
                        source_file=item["source_file"],
                        page_number=item["page_number"],
                        bbox=tuple(item["bbox"]),
                        region_type=item["region_type"],
                        surrounding_text=item.get("surrounding_text", ""),
                        caption=item.get("caption", ""),
                        description=item.get("description", "")
                    ))
                except KeyError as e:
                    print(f"  Warning: Skipping malformed result: missing {e}")
                    continue
            
            print(f"  Extracted {len(extracted_images)} items with PaddleOCR from {pdf_path.name}")
            return extracted_images
            
        except Exception as e:
            print(f"  Error in PaddleOCR extraction: {e}")
            import traceback
            traceback.print_exc()
            self._cleanup_temp_file(json_output_path)
            return []
    
    def _cleanup_temp_file(self, filepath: str):
        """Safely remove a temporary file."""
        import os
        try:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)
        except Exception:
            pass

    def _extract_pages_with_images_pymupdf(self, pdf_path: Path) -> list[ExtractedImage]:
        """
        Extract full pages as images if they contain any images or drawings.
        """
        try:
            doc = fitz.open(str(pdf_path))
            extracted_pages = []
            
            print(f"  Scanning {len(doc)} pages in {pdf_path.name}...")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                try:
                    # Render full page
                    pix = page.get_pixmap(dpi=150)
                    
                    image_data = pix.tobytes("png")
                    image_id = hashlib.md5(image_data).hexdigest()[:8]
                    image_filename = f"{pdf_path.stem}_page_{page_num + 1}_{image_id}.png"
                    image_path = self.config.output_dir / image_filename
                    
                    # Save image
                    pix.save(str(image_path))
                    pix = None
                    
                    # Get full page text
                    text = page.get_text()
                    
                    extracted_pages.append(ExtractedImage(
                        image_id=image_id,
                        image_path=str(image_path),
                        source_file=str(pdf_path),
                        page_number=page_num + 1,
                        bbox=(0, 0, page.rect.width, page.rect.height),
                        region_type="full_page_image",
                        surrounding_text=text,
                        caption="",
                        description=f"Full page image from {pdf_path.name}, page {page_num + 1}"
                    ))
                    
                except Exception as e:
                    print(f"    Error capturing page {page_num + 1}: {e}")
            
            doc.close()
            print(f"  Converted {len(extracted_pages)} pages to images from {pdf_path.name}")
            return extracted_pages
            
        except Exception as e:
            print(f"  Error with PyMuPDF: {e}")
            return []

    def extract_from_file(self, file_path: Path) -> list[ExtractedImage]:
        """
        Extract images from a file.
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self.extract_from_pdf(file_path)
        else:
            print(f"Skipping non-PDF file: {file_path.name}")
            return []
    
    def extract_from_directory(
        self,
        directory: Path,
        recursive: bool = True
    ) -> list[ExtractedImage]:
        """
        Extract images from all documents in a directory.
        """
        directory = Path(directory)
        all_images = []
        
        pattern = '**/*' if recursive else '*'
        extensions = ['.pdf']
        
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    images = self.extract_from_file(file_path)
                    all_images.extend(images)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
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


def check_paddle_installation() -> dict:
    """
    Check if PaddleOCR and its dependencies are properly installed.
    
    Returns:
        dict with keys:
            - 'available': bool - True if paddle mode can be used
            - 'packages': dict - Status of each required package
            - 'message': str - Human-readable status message
            - 'tips': list[str] - Installation tips if packages are missing
    """
    result = {
        'available': False,
        'packages': {},
        'message': '',
        'tips': []
    }
    
    # Check each required package
    packages_to_check = [
        ('paddlepaddle', 'paddle', 'pip install paddlepaddle'),
        ('paddleocr', 'paddleocr', 'pip install paddleocr'),
        ('opencv-python', 'cv2', 'pip install opencv-python'),
        ('numpy', 'numpy', 'pip install numpy'),
        ('pymupdf', 'fitz', 'pip install pymupdf'),
    ]
    
    all_ok = True
    for pkg_name, import_name, install_cmd in packages_to_check:
        try:
            __import__(import_name)
            result['packages'][pkg_name] = {'installed': True, 'version': None}
            
            # Try to get version
            try:
                import importlib.metadata
                result['packages'][pkg_name]['version'] = importlib.metadata.version(pkg_name)
            except:
                pass
                
        except ImportError:
            result['packages'][pkg_name] = {'installed': False, 'version': None}
            result['tips'].append(install_cmd)
            all_ok = False
    
    result['available'] = all_ok
    
    if all_ok:
        result['message'] = "PaddleOCR is available and ready to use."
        
        # Additional check: try to create PPStructure
        try:
            import os
            os.environ["FLAGS_use_mkldnn"] = "0"
            os.environ["GLOG_minloglevel"] = "3"
            from paddleocr import PPStructure
            result['message'] += " PP-Structure can be initialized."
        except Exception as e:
            result['message'] += f" Warning: PP-Structure init test failed: {e}"
            result['tips'].append("Try: pip install --upgrade paddleocr paddlepaddle")
    else:
        missing = [k for k, v in result['packages'].items() if not v['installed']]
        result['message'] = f"Missing packages: {', '.join(missing)}"
    
    return result


def print_paddle_status():
    """Print PaddleOCR installation status to console."""
    print("=" * 60)
    print("PaddleOCR Installation Status")
    print("=" * 60)
    
    status = check_paddle_installation()
    
    print(f"\nStatus: {'✓ Available' if status['available'] else '✗ Not Available'}")
    print(f"\n{status['message']}")
    
    print("\nPackages:")
    for pkg, info in status['packages'].items():
        version = info['version'] or 'unknown'
        if info['installed']:
            print(f"  ✓ {pkg}: {version}")
        else:
            print(f"  ✗ {pkg}: NOT INSTALLED")
    
    if status['tips']:
        print("\nTo install missing packages:")
        for tip in status['tips']:
            print(f"  {tip}")
    
    print("\n" + "=" * 60)
    
    if status['available']:
        print("You can use: python -m src.main process --mode paddle")
    else:
        print("Paddle mode unavailable. Use: python -m src.main process --mode pymupdf")
    print("=" * 60)
    
    return status['available']
