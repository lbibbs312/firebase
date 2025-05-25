

import os
import io
import sys
import json
import tempfile
import logging
import argparse
import shutil
import re
import uuid
import base64
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union, Iterator
from datetime import datetime
from pathlib import Path

# Import colorama for cross-platform colored terminal text
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    print("colorama not available. Install with: pip install colorama")
    class DummyColors:
        def __getattr__(self, name): return ''
    Fore = Back = Style = DummyColors()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pdf_form_handler")

# Web scraping imports
try:
    import requests
    from bs4 import BeautifulSoup
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False
    logger.warning(f"{Fore.YELLOW}Web scraping libraries not available. Install with: pip install requests beautifulsoup4{Style.RESET_ALL}")

# Core PDF libraries
try:
    import PyPDF2
    from PyPDF2 import PdfReader, PdfWriter
    from PyPDF2.generic import NameObject, NumberObject, TextStringObject, DictionaryObject
except ImportError:
    logger.error(f"{Fore.RED}PyPDF2 is required. Install with: pip install PyPDF2{Style.RESET_ALL}")
    sys.exit(1)

# Google Cloud Vision API
try:
    from google.cloud import vision
    from google.cloud import storage
    GOOGLE_VISION_AVAILABLE = True
    logger.info(f"{Fore.GREEN}Google Cloud Vision API available{Style.RESET_ALL}")
except ImportError:
    GOOGLE_VISION_AVAILABLE = False
    logger.warning(f"{Fore.YELLOW}Google Cloud Vision API not available. Install with: pip install google-cloud-vision{Style.RESET_ALL}")

# Additional libraries with fallbacks
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning(f"{Fore.YELLOW}PyMuPDF not available. Install with: pip install pymupdf{Style.RESET_ALL}")

try:
    from fillpdf import fillpdfs
    FILLPDFS_AVAILABLE = True
except ImportError:
    FILLPDFS_AVAILABLE = False
    logger.warning(f"{Fore.YELLOW}fillpdf not available. Install with: pip install fillpdf{Style.RESET_ALL}")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning(f"{Fore.YELLOW}PIL/Pillow not available. Install with: pip install pillow{Style.RESET_ALL}")

# Try to import pdf2image
try:
    from pdf2image import convert_from_path, convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning(f"{Fore.YELLOW}pdf2image not available. Install with: pip install pdf2image{Style.RESET_ALL}")

# Try to import reportlab
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning(f"{Fore.YELLOW}reportlab not available. Install with: pip install reportlab{Style.RESET_ALL}")

# Try to import shapely for geometry operations
try:
    from shapely.geometry import Polygon, box
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    logger.warning(f"{Fore.YELLOW}shapely not available. Install with: pip install shapely{Style.RESET_ALL}")

# Try to import pdftk-python
try:
    from pdftk import PDFTK
    PDFTK_AVAILABLE = True
except ImportError:
    PDFTK_AVAILABLE = False
    # Check if pdftk executable is available
    try:
        import subprocess
        result = subprocess.run(['pdftk', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        PDFTK_AVAILABLE = result.returncode == 0
    except:
        PDFTK_AVAILABLE = False

# Try to import win32com for Adobe Acrobat automation
try:
    import win32com.client
    ACROBAT_AVAILABLE = True
except ImportError:
    ACROBAT_AVAILABLE = False
    logger.warning(f"{Fore.YELLOW}win32com not available. Adobe Acrobat automation disabled.{Style.RESET_ALL}")

# Create a simple file storage class for workspace
class FileStorage:
    """Simple file storage class to mimic workspace functionality"""
    
    def __init__(self, root_dir="."):
        self.root = Path(root_dir)
        os.makedirs(self.root, exist_ok=True)
    
    def get_path(self, path):
        """Get the full path of a file"""
        return self.root / path
    
    def exists(self, path):
        """Check if a file exists"""
        return (self.root / path).exists()
    
    def write_file(self, path, content):
        """Write content to a file"""
        full_path = self.root / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(content, str):
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            with open(full_path, 'wb') as f:
                f.write(content)
    
    def read_file(self, path, binary=False):
        """Read content from a file"""
        full_path = self.root / path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if binary:
            with open(full_path, 'rb') as f:
                return f.read()
        else:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def delete_file(self, path):
        """Delete a file"""
        full_path = self.root / path
        if full_path.exists():
            full_path.unlink()


# AutoGPT Output Functions
def thinking(message: str) -> None:
    """Display thinking output in yellow color."""
    if COLORAMA_AVAILABLE:
        print(f"{Fore.YELLOW}💭 THINKING: {message}{Style.RESET_ALL}")
    else:
        print(f"[💭 THINKING] {message}")

def reasoning(message: str) -> None:
    """Display reasoning output in green color."""
    if COLORAMA_AVAILABLE:
        print(f"{Fore.GREEN}🧠 REASONING: {message}{Style.RESET_ALL}")
    else:
        print(f"[🧠 REASONING] {message}")

def speaking(message: str) -> None:
    """Display speaking output in blue color."""
    if COLORAMA_AVAILABLE:
        print(f"{Fore.BLUE}🗣️ SPEAKING: {message}{Style.RESET_ALL}")
    else:
        print(f"[🗣️ SPEAKING] {message}")

def success(message: str) -> None:
    """Display success message in green color."""
    if COLORAMA_AVAILABLE:
        print(f"{Fore.GREEN}✅ SUCCESS: {message}{Style.RESET_ALL}")
    else:
        print(f"[✅ SUCCESS] {message}")

def warning(message: str) -> None:
    """Display warning message in yellow color."""
    if COLORAMA_AVAILABLE:
        print(f"{Fore.YELLOW}⚠️ WARNING: {message}{Style.RESET_ALL}")
    else:
        print(f"[⚠️ WARNING] {message}")

def error(message: str) -> None:
    """Display error message in red color."""
    if COLORAMA_AVAILABLE:
        print(f"{Fore.RED}❌ ERROR: {message}{Style.RESET_ALL}")
    else:
        print(f"[❌ ERROR] {message}")

def info(message: str) -> None:
    """Display info message in cyan color."""
    if COLORAMA_AVAILABLE:
        print(f"{Fore.CYAN}ℹ️ INFO: {message}{Style.RESET_ALL}")
    else:
        print(f"[ℹ️ INFO] {message}")


class GoogleVisionPDFStrategy:
    """PDF processing strategy using Google Vision."""
    
    def __init__(self, vision_client):
        self.vision_client = vision_client
        
    def repair_pdf(self, pdf_path: str) -> Optional[str]:
        """Repair a PDF using multiple strategies.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Path to the repaired PDF or None if repair failed
        """
        repair_strategies = [
            self._repair_with_pymupdf,
            self._repair_with_ghostscript,
            self._repair_with_qpdf
        ]
        
        # Try each repair strategy until one succeeds
        for strategy in repair_strategies:
            try:
                repaired_path = strategy(pdf_path)
                if repaired_path and os.path.exists(repaired_path):
                    if self._validate_pdf(repaired_path):
                        logger.info(f"PDF repaired successfully using {strategy.__name__}")
                        return repaired_path
            except Exception as e:
                logger.warning(f"Repair strategy {strategy.__name__} failed: {e}")
                
        logger.warning(f"All PDF repair strategies failed for {pdf_path}")
        return None
    
    def _repair_with_pymupdf(self, pdf_path: str) -> Optional[str]:
        """Repair PDF using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Path to the repaired PDF
        """
        if not PYMUPDF_AVAILABLE:
            return None
            
        try:
            output_path = os.path.join(
                tempfile.gettempdir(),
                f"repaired_pymupdf_{os.path.basename(pdf_path)}"
            )
            
            # Open the original document and check page count
            doc = fitz.open(pdf_path)
            original_page_count = len(doc)
            logger.info(f"Original PDF has {original_page_count} pages")
            
            # Get document metadata for preservation
            original_metadata = doc.metadata
            
            # Save with cleanup options
            doc.save(
                output_path,
                clean=True,      # Remove unreferenced objects
                garbage=4,       # Maximum garbage collection
                deflate=True,    # Compress streams
                pretty=False,    # No pretty printing (smaller file)
                linear=True      # Create linearized PDF
            )
            doc.close()
            
            # Verify the repaired document
            repaired_doc = fitz.open(output_path)
            repaired_page_count = len(repaired_doc)
            
            # If page count changed, log warning and return None
            if repaired_page_count != original_page_count:
                logger.warning(f"PyMuPDF repair changed page count: {original_page_count} to {repaired_page_count}")
                repaired_doc.close()
                return None
                
            # Restore original metadata
            repaired_doc.set_metadata(original_metadata)
            repaired_doc.save(output_path)
            repaired_doc.close()
            
            return output_path
        except Exception as e:
            logger.warning(f"PyMuPDF repair failed: {e}")
            return None
    
    def _repair_with_ghostscript(self, pdf_path: str) -> Optional[str]:
        """Repair PDF using Ghostscript (assumes ghostscript is installed).
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Path to the repaired PDF
        """
        try:
            import subprocess
            
            output_path = os.path.join(
                tempfile.gettempdir(),
                f"repaired_gs_{os.path.basename(pdf_path)}"
            )
            
            # Use ghostscript to repair
            cmd = [
                "gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER",
                "-sDEVICE=pdfwrite", "-dPDFSETTINGS=/prepress",
                f"-sOutputFile={output_path}", pdf_path
            ]
            
            subprocess.run(cmd, check=True)
            
            return output_path
        except Exception as e:
            logger.warning(f"Ghostscript repair failed: {e}")
            return None
    
    def _repair_with_qpdf(self, pdf_path: str) -> Optional[str]:
        """Repair PDF using qpdf (assumes qpdf is installed).
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Path to the repaired PDF
        """
        try:
            import subprocess
            
            output_path = os.path.join(
                tempfile.gettempdir(),
                f"repaired_qpdf_{os.path.basename(pdf_path)}"
            )
            
            # Use qpdf to repair
            cmd = [
                "qpdf", "--replace-input", 
                "--object-streams=generate",
                "--compress-streams=y",
                "--recompress-flate",
                "--linearize",
                pdf_path, output_path
            ]
            
            subprocess.run(cmd, check=True)
            
            return output_path
        except Exception as e:
            logger.warning(f"QPDF repair failed: {e}")
            return None
    
    def _validate_pdf(self, pdf_path: str) -> bool:
        """Validate that a PDF is readable.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if PDF is valid, False otherwise
        """
        try:
            # Try with PyPDF2 (always available)
            reader = PdfReader(pdf_path)
            if len(reader.pages) == 0:
                return False
                
            # Also try with PyMuPDF if available
            if PYMUPDF_AVAILABLE:
                doc = fitz.open(pdf_path)
                page_count = len(doc)
                doc.close()
                
                if page_count == 0:
                    return False
            
            # If we can read pages, it's probably valid
            return True
        except Exception as e:
            logger.warning(f"PDF validation failed: {e}")
            return False
    
    def convert_to_images(self, pdf_path: str, output_dir: str) -> List[str]:
        """Convert PDF to images using multiple strategies.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save images
            
        Returns:
            List of paths to the generated images
        """
        strategies = []
        
        # Add available strategies based on installed libraries
        if PDF2IMAGE_AVAILABLE:
            strategies.append(self._convert_with_pdf2image)
        
        if PYMUPDF_AVAILABLE:
            strategies.append(self._convert_with_pymupdf)
            
        # Always try ghostscript last if available
        strategies.append(self._convert_with_gs)
        
        # Try each conversion strategy until one succeeds
        for strategy in strategies:
            try:
                image_paths = strategy(pdf_path, output_dir)
                if image_paths and len(image_paths) > 0:
                    logger.info(f"PDF converted successfully using {strategy.__name__}")
                    return image_paths
            except Exception as e:
                logger.warning(f"Conversion strategy {strategy.__name__} failed: {e}")
                
        # If all strategies fail, try one last approach: repair and convert
        try:
            repaired_path = self.repair_pdf(pdf_path)
            if repaired_path:
                # Try converting the repaired PDF with pdf2image
                for strategy in strategies:
                    try:
                        image_paths = strategy(repaired_path, output_dir)
                        if image_paths and len(image_paths) > 0:
                            return image_paths
                    except:
                        pass
        except Exception as e:
            logger.warning(f"Repair and convert approach failed: {e}")
            
        logger.error(f"All PDF conversion strategies failed for {pdf_path}")
        return []
    
    def _convert_with_pdf2image(self, pdf_path: str, output_dir: str) -> List[str]:
        """Convert PDF to images using pdf2image.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save images
            
        Returns:
            List of paths to the generated images
        """
        if not PDF2IMAGE_AVAILABLE:
            return []
            
        try:
            # Convert PDF to images at 300 DPI
            images = convert_from_path(
                pdf_path,
                dpi=300,
                output_folder=output_dir,
                fmt="png",
                transparent=False
            )
            
            # Save and collect paths
            image_paths = []
            for i, img in enumerate(images):
                img_path = os.path.join(output_dir, f"page_{i}.png")
                img.save(img_path)
                image_paths.append(img_path)
                
            return image_paths
        except Exception as e:
            logger.warning(f"pdf2image conversion failed: {e}")
            return []
    
    def _convert_with_pymupdf(self, pdf_path: str, output_dir: str) -> List[str]:
        """Convert PDF to images using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save images
            
        Returns:
            List of paths to the generated images
        """
        if not PYMUPDF_AVAILABLE:
            return []
            
        try:
            doc = fitz.open(pdf_path)
            image_paths = []
            
            for i, page in enumerate(doc):
                # Render page to pixmap at 300 DPI (factor 3)
                pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
                img_path = os.path.join(output_dir, f"page_{i}.png")
                pix.save(img_path)
                image_paths.append(img_path)
                
            doc.close()
            return image_paths
        except Exception as e:
            logger.warning(f"PyMuPDF conversion failed: {e}")
            return []
    
    def _convert_with_gs(self, pdf_path: str, output_dir: str) -> List[str]:
        """Convert PDF to images using Ghostscript.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save images
            
        Returns:
            List of paths to the generated images
        """
        try:
            import subprocess
            
            # Define output pattern
            output_pattern = os.path.join(output_dir, "page_%d.png")
            
            # Use ghostscript to convert
            cmd = [
                "gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER",
                "-sDEVICE=png16m", "-r300",
                f"-sOutputFile={output_pattern}", pdf_path
            ]
            
            subprocess.run(cmd, check=True)
            
            # Collect the output files
            image_paths = []
            i = 0
            while True:
                img_path = os.path.join(output_dir, f"page_{i}.png")
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                    i += 1
                else:
                    break
                    
            return image_paths
        except Exception as e:
            logger.warning(f"Ghostscript conversion failed: {e}")
            return []
    
    def extract_text_with_positions(self, pdf_path: str) -> Dict[int, List[Dict]]:
        """Extract text with positions using Google Vision.
        
        This uses a multi-step approach:
        1. Convert PDF to images
        2. Use Google Vision to extract text with positions
        3. Organize results by page
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary mapping page numbers to lists of text blocks with positions
        """
        results = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Convert PDF to images
            image_paths = self.convert_to_images(pdf_path, temp_dir)
            
            if not image_paths:
                logger.error("Failed to convert PDF to images for text extraction")
                return results
                
            # Process each page with Google Vision
            for i, img_path in enumerate(image_paths):
                try:
                    with open(img_path, "rb") as image_file:
                        content = image_file.read()
                        
                    image = vision.Image(content=content)
                    response = self.vision_client.document_text_detection(image=image)
                    
                    if response.error.message:
                        logger.warning(f"Google Vision error on page {i}: {response.error.message}")
                        continue
                        
                    # Extract text blocks with positions
                    blocks = []
                    
                    # Get image dimensions for normalization
                    with Image.open(img_path) as img:
                        img_width, img_height = img.size
                        
                    # Process text blocks
                    for page in response.full_text_annotation.pages:
                        for block in page.blocks:
                            # Get block vertices 
                            vertices = []
                            for vertex in block.bounding_box.vertices:
                                # Normalize to [0,1] range
                                vertices.append({
                                    "x": vertex.x / img_width if img_width else 0,
                                    "y": vertex.y / img_height if img_height else 0
                                })
                                
                            # Extract block text
                            block_text = ""
                            for paragraph in block.paragraphs:
                                for word in paragraph.words:
                                    word_text = ''.join([symbol.text for symbol in word.symbols])
                                    block_text += word_text + " "
                                    
                            blocks.append({
                                "text": block_text.strip(),
                                "vertices": vertices,
                                "confidence": block.confidence
                            })
                            
                    results[i] = blocks
                    
                except Exception as e:
                    logger.error(f"Error extracting text from page {i}: {e}")
                    
        return results
    
    def detect_form_fields(self, pdf_path: str) -> Dict[int, List[Dict]]:
        """Detect form fields in a PDF using Google Vision.
        
        Uses a combination of approaches:
        1. PyMuPDF to get existing form fields
        2. Google Vision to detect potential form fields from visual cues
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary mapping page numbers to lists of form fields
        """
        results = {}
        
        # First, try to get existing form fields from the PDF
        if PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(pdf_path)
                
                # Check for existing form fields
                for i, page in enumerate(doc):
                    if page.widgets:
                        if i not in results:
                            results[i] = []
                            
                        # Extract form field info
                        for widget in page.widgets:
                            field = {
                                "type": widget.field_type_string,
                                "name": widget.field_name,
                                "value": widget.field_value,
                                "rect": [v for v in widget.rect],
                                "vertices": [
                                    {"x": widget.rect[0], "y": widget.rect[1]},
                                    {"x": widget.rect[2], "y": widget.rect[1]},
                                    {"x": widget.rect[2], "y": widget.rect[3]},
                                    {"x": widget.rect[0], "y": widget.rect[3]}
                                ],
                                "is_existing": True
                            }
                            results[i].append(field)
                            
                doc.close()
            except Exception as e:
                logger.warning(f"Error extracting existing form fields: {e}")
        
        # Next, use Google Vision to detect potential form fields
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Convert PDF to images
                image_paths = self.convert_to_images(pdf_path, temp_dir)
                
                if not image_paths:
                    logger.error("Failed to convert PDF to images for form field detection")
                    return results
                    
                # Process each page with Google Vision
                for i, img_path in enumerate(image_paths):
                    # Initialize page results if not exist
                    if i not in results:
                        results[i] = []
                        
                    # Detect form fields using Vision API
                    detected_fields = self._detect_form_fields_with_vision(img_path)
                    
                    # Mark these as detected (not existing in PDF)
                    for field in detected_fields:
                        field["is_existing"] = False
                        results[i].append(field)
                        
            except Exception as e:
                logger.error(f"Error detecting form fields with Vision: {e}")
                
        return results
    
    def _detect_form_fields_with_vision(self, image_path: str) -> List[Dict]:
        """Detect form fields in an image using Google Vision.
        
        Args:
            image_path: Path to the image
            
        Returns:
            List of detected form fields
        """
        try:
            with open(image_path, "rb") as image_file:
                content = image_file.read()
                
            image = vision.Image(content=content)
            
            # Get document text with layout
            response = self.vision_client.document_text_detection(image=image)
            
            if response.error.message:
                logger.warning(f"Google Vision error: {response.error.message}")
                return []
                
            # Get image dimensions for normalization
            with Image.open(image_path) as img:
                img_width, img_height = img.size
                
            # Detect potential form field indicators
            fields = []
            
            # Process text blocks to find form field indicators
            for page in response.full_text_annotation.pages:
                # Look for checkbox-like symbols
                self._detect_checkboxes(page, img_width, img_height, fields)
                
                # Look for underlines and blank spaces that might be form fields
                self._detect_text_fields(page, img_width, img_height, fields)
                
            return fields
            
        except Exception as e:
            logger.error(f"Error detecting form fields: {e}")
            return []
    
    def _detect_checkboxes(self, page, img_width, img_height, fields):
        """Detect checkboxes in a page.
        
        Args:
            page: Google Vision page
            img_width: Image width
            img_height: Image height
            fields: List to append fields to
        """
        # Look for checkbox symbols (□, ☐, etc.) or small square shapes
        checkbox_symbols = ["□", "☐", "▢", "▫", "⬜"]
        
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    
                    # Check if word contains checkbox symbol
                    if any(sym in word_text for sym in checkbox_symbols):
                        # Get bounding box
                        vertices = []
                        for vertex in word.bounding_box.vertices:
                            vertices.append({
                                "x": vertex.x / img_width if img_width else 0,
                                "y": vertex.y / img_height if img_height else 0
                            })
                            
                        # Add as a checkbox field
                        fields.append({
                            "type": "checkbox",
                            "vertices": vertices,
                            "value": False,
                            "confidence": 0.8
                        })
    
    def _detect_text_fields(self, page, img_width, img_height, fields):
        """Detect text fields in a page.
        
        Args:
            page: Google Vision page
            img_width: Image width
            img_height: Image height
            fields: List to append fields to
        """
        # Look for common form field indicators: text followed by colon, underline, or space
        for block in page.blocks:
            for paragraph in block.paragraphs:
                # Check if paragraph ends with colon
                para_text = ""
                para_vertices = []
                
                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    para_text += word_text + " "
                    
                    # Collect all word vertices to get paragraph bbox
                    for vertex in word.bounding_box.vertices:
                        para_vertices.append((vertex.x, vertex.y))
                        
                para_text = para_text.strip()
                
                # If paragraph ends with colon, look for potential form field to the right
                if para_text.endswith(":"):
                    # Calculate bounding box for paragraph
                    if para_vertices:
                        min_x = min(v[0] for v in para_vertices)
                        min_y = min(v[1] for v in para_vertices) 
                        max_x = max(v[0] for v in para_vertices)
                        max_y = max(v[1] for v in para_vertices)
                        
                        # Create a potential text field to the right
                        field_width = (max_x - min_x) * 2  # Make field 2x as wide as label
                        field_height = max_y - min_y
                        
                        # Field starts at the right of the label and has same height
                        field_vertices = [
                            {"x": max_x / img_width, "y": min_y / img_height},
                            {"x": (max_x + field_width) / img_width, "y": min_y / img_height},
                            {"x": (max_x + field_width) / img_width, "y": max_y / img_height},
                            {"x": max_x / img_width, "y": max_y / img_height}
                        ]
                        
                        # Add as a text field
                        fields.append({
                            "type": "text",
                            "label": para_text[:-1],  # Remove colon
                            "vertices": field_vertices,
                            "value": "",
                            "confidence": 0.7
                        })
    
    def edit_pdf(self, pdf_path: str, edits: Dict[int, List[Dict]], flatten: bool = True) -> str:
        """Edit a PDF with the given edits.
        
        Args:
            pdf_path: Path to the PDF file
            edits: Dictionary mapping page numbers to lists of edits
            flatten: Whether to flatten the PDF after editing (make annotations part of content)
            
        Returns:
            Path to the edited PDF
        """
        # Create output path
        output_path = os.path.join(
            tempfile.gettempdir(),
            f"edited_{os.path.basename(pdf_path)}"
        )
        
        # Use PyMuPDF if available
        if PYMUPDF_AVAILABLE:
            try:
                edited_path = self._edit_with_pymupdf(pdf_path, edits, output_path)
                if edited_path and os.path.exists(edited_path):
                    if self._validate_pdf(edited_path):
                        # If flattening is requested, flatten the PDF
                        if flatten:
                            flattened_path = self._flatten_pdf(edited_path)
                            if flattened_path:
                                return flattened_path
                        return edited_path
            except Exception as e:
                logger.warning(f"PyMuPDF edit failed: {e}")
        
        # Fallback to ReportLab + PyPDF
        if REPORTLAB_AVAILABLE:
            try:
                edited_path = self._edit_with_reportlab(pdf_path, edits, output_path)
                if edited_path and os.path.exists(edited_path):
                    if self._validate_pdf(edited_path):
                        # If flattening is requested, flatten the PDF
                        if flatten:
                            flattened_path = self._flatten_pdf(edited_path)
                            if flattened_path:
                                return flattened_path
                        return edited_path
            except Exception as e:
                logger.warning(f"ReportLab edit failed: {e}")
            
        logger.error(f"All PDF edit strategies failed for {pdf_path}")
        return pdf_path  # Return original if all edits fail
    
    def _flatten_pdf(self, pdf_path: str) -> Optional[str]:
        """Flatten a PDF, making annotations and form fields part of the content.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Path to the flattened PDF or None if flattening failed
        """
        if not PYMUPDF_AVAILABLE:
            return None
            
        try:
            output_path = os.path.join(
                tempfile.gettempdir(),
                f"flattened_{os.path.basename(pdf_path)}"
            )
            
            doc = fitz.open(pdf_path)
            
            # For each page, create a new page with all content rasterized
            new_doc = fitz.open()
            
            for i, page in enumerate(doc):
                # Create a pixmap of the page at high resolution (300 DPI)
                pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
                
                # Create a new page in the output document
                new_page = new_doc.new_page(width=page.rect.width, height=page.rect.height)
                
                # Insert the pixmap as an image covering the whole page
                new_page.insert_image(page.rect, pixmap=pix)
                
            # Save the new document
            new_doc.save(output_path)
            new_doc.close()
            doc.close()
            
            return output_path
            
        except Exception as e:
            logger.warning(f"PDF flattening failed: {e}")
            return None

    def _edit_with_pymupdf(self, pdf_path: str, edits: Dict[int, List[Dict]], output_path: str) -> str:
        """Edit PDF using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            edits: Dictionary mapping page numbers to lists of edits
            output_path: Path to save the edited PDF
            
        Returns:
            Path to the edited PDF
        """
        if not PYMUPDF_AVAILABLE:
            return None
            
        doc = fitz.open(pdf_path)
        
        # Apply edits to each page
        for page_num, page_edits in edits.items():
            if page_num >= len(doc):
                continue
                
            page = doc[page_num]
            
            for edit in page_edits:
                edit_type = edit.get("type", "text")
                
                if edit_type == "text":
                    # Get normalized coordinates and denormalize to page coordinates
                    vertices = edit.get("vertices", [])
                    if not vertices or len(vertices) < 4:
                        continue
                        
                    # Convert vertices to rectangle
                    rect = fitz.Rect(
                        vertices[0]["x"] * page.rect.width,
                        vertices[0]["y"] * page.rect.height,
                        vertices[2]["x"] * page.rect.width,
                        vertices[2]["y"] * page.rect.height
                    )
                    
                    # Add text annotation
                    text_value = edit.get("value", "")
                    if text_value:
                        # For form fields, update or create
                        if "is_existing" in edit and edit["is_existing"]:
                            # Find widget by name
                            field_name = edit.get("name")
                            for widget in page.widgets():
                                if widget.field_name == field_name:
                                    widget.field_value = text_value
                                    break
                        else:
                            # Insert as text
                            page.insert_text(
                                rect.tl,  # Top-left point
                                text_value,
                                fontsize=11,
                                color=(0, 0, 0)
                            )
                            
                elif edit_type == "checkbox":
                    # Handle checkboxes
                    vertices = edit.get("vertices", [])
                    if not vertices or len(vertices) < 4:
                        continue
                        
                    # Convert vertices to rectangle
                    rect = fitz.Rect(
                        vertices[0]["x"] * page.rect.width,
                        vertices[0]["y"] * page.rect.height,
                        vertices[2]["x"] * page.rect.width,
                        vertices[2]["y"] * page.rect.height
                    )
                    
                    checked = edit.get("value", False)
                    
                    # For existing form fields
                    if "is_existing" in edit and edit["is_existing"]:
                        field_name = edit.get("name")
                        for widget in page.widgets():
                            if widget.field_name == field_name:
                                widget.field_value = checked
                                break
                    else:
                        # Add a visual checkmark if checked
                        if checked:
                            page.insert_text(
                                rect.tl,  # Top-left point
                                "✓",
                                fontsize=12,
                                color=(0, 0, 0)
                            )
        
        # Save the document
        doc.save(output_path)
        doc.close()
        
        return output_path
    
    def _edit_with_reportlab(self, pdf_path: str, edits: Dict[int, List[Dict]], output_path: str) -> str:
        """Edit PDF using ReportLab and PyPDF.
        
        This approach:
        1. Creates an overlay PDF with edits for each page
        2. Merges the overlays with the original PDF
        
        Args:
            pdf_path: Path to the PDF file
            edits: Dictionary mapping page numbers to lists of edits
            output_path: Path to save the edited PDF
            
        Returns:
            Path to the edited PDF
        """
        if not REPORTLAB_AVAILABLE:
            return None
            
        # Read the input PDF
        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        
        # For each page with edits
        for page_num in range(len(reader.pages)):
            # Get the page
            page = reader.pages[page_num]
            
            # If there are edits for this page
            if page_num in edits and edits[page_num]:
                # Create in-memory buffer for overlay
                overlay_buffer = io.BytesIO()
                
                # Create canvas
                c = canvas.Canvas(
                    overlay_buffer, 
                    pagesize=(page.mediabox.width, page.mediabox.height)
                )
                
                # Apply edits
                for edit in edits[page_num]:
                    edit_type = edit.get("type", "text")
                    vertices = edit.get("vertices", [])
                    
                    if not vertices or len(vertices) < 4:
                        continue
                        
                    # Convert normalized coordinates to page coordinates
                    x1 = vertices[0]["x"] * page.mediabox.width
                    y1 = (1 - vertices[0]["y"]) * page.mediabox.height  # Invert Y for ReportLab
                    x2 = vertices[2]["x"] * page.mediabox.width
                    y2 = (1 - vertices[2]["y"]) * page.mediabox.height  # Invert Y for ReportLab
                    
                    # Apply edit based on type
                    if edit_type == "text":
                        text_value = edit.get("value", "")
                        if text_value:
                            c.setFont("Helvetica", 11)
                            c.setFillColorRGB(0, 0, 0)
                            # ReportLab text origin is at bottom left
                            c.drawString(x1, y2, text_value)
                            
                    elif edit_type == "checkbox":
                        checked = edit.get("value", False)
                        if checked:
                            c.setFont("Helvetica", 12)
                            c.setFillColorRGB(0, 0, 0)
                            c.drawString(x1, y2, "✓")
                
                # Finish the page
                c.save()
                
                # Move buffer position to the beginning
                overlay_buffer.seek(0)
                
                # Create PDF reader from overlay
                overlay = PdfReader(overlay_buffer)
                
                # Merge page with overlay
                page.merge_page(overlay.pages[0])
            
            # Add the page to the writer
            writer.add_page(page)
        
        # Write the result to file
        with open(output_path, "wb") as output_file:
            writer.write(output_file)
            
        return output_path


class GoogleVisionPDFHandler:
    """Comprehensive PDF handler using Google Vision API."""
    
    def __init__(self, vision_client, output_dir="./pdf_output", workspace=None):
        """Initialize with Google Vision client and options.
        
        Args:
            vision_client: Google Vision client
            output_dir: Directory for output files
            workspace: Optional workspace for file handling
        """
        self.vision_client = vision_client
        self.output_dir = output_dir
        self.pdf_strategy = GoogleVisionPDFStrategy(vision_client)
        self.workspace = workspace or FileStorage(output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a PDF file.
        
        Complete pipeline:
        1. Repair PDF
        2. Extract text with positions
        3. Detect form fields
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Processing results
        """
        if not os.path.exists(pdf_path):
            logger.error(f"File not found: {pdf_path}")
            return {"error": f"File not found: {pdf_path}"}
            
        results = {
            "original_path": pdf_path,
            "success": False,
            "repaired_path": None,
            "text_extraction": None,
            "form_fields": None,
            "errors": []
        }
        
        # Step 1: Repair PDF
        try:
            repaired_path = self.pdf_strategy.repair_pdf(pdf_path)
            if repaired_path:
                results["repaired_path"] = repaired_path
                logger.info(f"PDF repaired: {repaired_path}")
            else:
                results["errors"].append("PDF repair failed, using original file")
                repaired_path = pdf_path
        except Exception as e:
            results["errors"].append(f"PDF repair error: {str(e)}")
            repaired_path = pdf_path
            
        # Step 2: Extract text with positions
        try:
            text_results = self.pdf_strategy.extract_text_with_positions(repaired_path)
            results["text_extraction"] = text_results
            logger.info(f"Extracted text from {len(text_results)} pages")
        except Exception as e:
            results["errors"].append(f"Text extraction error: {str(e)}")
            
        # Step 3: Detect form fields
        try:
            form_fields = self.pdf_strategy.detect_form_fields(repaired_path)
            results["form_fields"] = form_fields
            form_field_count = sum(len(fields) for fields in form_fields.values())
            logger.info(f"Detected {form_field_count} form fields")
        except Exception as e:
            results["errors"].append(f"Form field detection error: {str(e)}")
            
        results["success"] = len(results["errors"]) == 0
        return results
    
    def fill_form(self, pdf_path: str, form_data: Dict[str, Any], output_path: str = None) -> str:
        """Fill a PDF form with data.
        
        Args:
            pdf_path: Path to the PDF file
            form_data: Dictionary mapping field names to values
            output_path: Path to save the filled PDF (optional)
            
        Returns:
            Path to the filled PDF
        """
        # Create edits dictionary from form data
        edits = {}
        
        # First detect form fields
        form_fields = self.pdf_strategy.detect_form_fields(pdf_path)
        
        # For each page with fields
        for page_num, fields in form_fields.items():
            page_edits = []
            
            # For each field
            for field in fields:
                field_name = field.get("name", "")
                
                # If this field is in the form data
                if field_name and field_name in form_data:
                    edit = {
                        "type": field.get("type", "text"),
                        "vertices": field.get("vertices", []),
                        "value": form_data[field_name],
                        "is_existing": field.get("is_existing", False)
                    }
                    
                    if "name" in field:
                        edit["name"] = field["name"]
                        
                    page_edits.append(edit)
            
            if page_edits:
                edits[page_num] = page_edits
        
        # If no output path provided, create one
        if not output_path:
            output_path = os.path.join(
                self.output_dir,
                f"filled_{os.path.basename(pdf_path)}"
            )
        
        # Apply edits
        return self.pdf_strategy.edit_pdf(pdf_path, edits, flatten=True)
    
    def verify_form_is_complete(self, pdf_path: str) -> Dict[str, Any]:
        """Verify that a form is complete.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Verification results
        """
        results = {
            "is_complete": False,
            "missing_fields": [],
            "filled_fields": [],
            "total_fields": 0
        }
        
        try:
            # Detect form fields
            form_fields = self.pdf_strategy.detect_form_fields(pdf_path)
            
            # Check each field
            for page_num, fields in form_fields.items():
                for field in fields:
                    results["total_fields"] += 1
                    
                    # Check if field has a value
                    field_name = field.get("name", field.get("label", f"Field {results['total_fields']}"))
                    field_value = field.get("value", "")
                    
                    if not field_value or field_value == "":
                        results["missing_fields"].append({
                            "name": field_name,
                            "page": page_num
                        })
                    else:
                        results["filled_fields"].append({
                            "name": field_name,
                            "value": field_value,
                            "page": page_num
                        })
                        
            results["is_complete"] = len(results["missing_fields"]) == 0
            return results
        except Exception as e:
            logger.error(f"Error verifying form: {e}")
            results["error"] = str(e)
            return results


class GoogleVisionComponent:
    """Component for handling Google Vision API operations."""
    
    def __init__(self, client=None, config=None, workspace=None):
        """Initialize with all required parameters.
        
        Args:
            client: Google Vision client
            config: Configuration object
            workspace: Workspace for file handling
        """
        self.client = client or vision.ImageAnnotatorClient()
        self.config = config or type('obj', (object,), {'pdf_output_dir': './pdf_output'})
        self.workspace = workspace or FileStorage()
        
        # Initialize PDF handler
        self.pdf_handler = GoogleVisionPDFHandler(
            vision_client=self.client, 
            output_dir=self.config.pdf_output_dir,
            workspace=self.workspace
        )
        
    def analyze_pdf(self, pdf_path: str, page_numbers: Optional[List[int]] = None) -> Dict[str, Any]:
        """Analyze a PDF document with Google Vision API.
        
        Args:
            pdf_path: Path to the PDF file
            page_numbers: Optional list of page numbers to analyze (0-based)
            
        Returns:
            Dict with analysis results
        """
        if not os.path.isabs(pdf_path):
            try:
                pdf_path = str(self.workspace.get_path(pdf_path))
            except:
                pdf_path = os.path.abspath(pdf_path)
        
        if not os.path.exists(pdf_path):
            return {"error": f"File not found: {pdf_path}"}
        
        try:
            # Process the PDF
            results = self.pdf_handler.process_pdf(pdf_path)
            
            # If specific pages were requested, filter results
            if page_numbers and "text_extraction" in results:
                filtered_text = {
                    i: results["text_extraction"][i] 
                    for i in page_numbers 
                    if i in results["text_extraction"]
                }
                results["text_extraction"] = filtered_text
                
                filtered_fields = {
                    i: results["form_fields"][i] 
                    for i in page_numbers 
                    if i in results["form_fields"]
                }
                results["form_fields"] = filtered_fields
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing PDF with Google Vision: {e}")
            return {"error": f"Google Vision analysis failed: {str(e)}"}
    
    def verify_form_is_complete(self, pdf_path: str) -> Dict[str, Any]:
        """Verify that a form PDF is complete with all fields filled.
        
        Args:
            pdf_path: Path to the PDF form
            
        Returns:
            Dict with verification results
        """
        if not os.path.isabs(pdf_path):
            try:
                pdf_path = str(self.workspace.get_path(pdf_path))
            except:
                pdf_path = os.path.abspath(pdf_path)
        
        if not os.path.exists(pdf_path):
            return {"error": f"File not found: {pdf_path}"}
        
        try:
            # Verify form
            return self.pdf_handler.verify_form_is_complete(pdf_path)
            
        except Exception as e:
            logger.error(f"Error verifying form: {e}")
            return {"error": f"Form verification failed: {str(e)}"}
    
    def fill_pdf_form(self, pdf_path: str, form_data: Dict[str, Any], output_path: str = None) -> Dict[str, Any]:
        """Fill a PDF form with the provided data.
        
        Args:
            pdf_path: Path to the PDF form
            form_data: Dictionary mapping field names to values
            output_path: Optional output path
            
        Returns:
            Dict with results including path to filled PDF
        """
        if not os.path.isabs(pdf_path):
            try:
                pdf_path = str(self.workspace.get_path(pdf_path))
            except:
                pdf_path = os.path.abspath(pdf_path)
        
        if not os.path.exists(pdf_path):
            return {"error": f"File not found: {pdf_path}"}
        
        # If output_path is provided, ensure it's an absolute path
        if output_path and not os.path.isabs(output_path):
            try:
                output_path = str(self.workspace.get_path(output_path))
            except:
                output_path = os.path.abspath(output_path)
        
        try:
            # Fill form
            filled_pdf_path = self.pdf_handler.fill_form(pdf_path, form_data, output_path)
            
            # Verify form is complete
            verification = self.pdf_handler.verify_form_is_complete(filled_pdf_path)
            
            return {
                "filled_pdf_path": filled_pdf_path,
                "verification": verification
            }
            
        except Exception as e:
            logger.error(f"Error filling form: {e}")
            return {"error": f"Form filling failed: {str(e)}"}


class GoogleVisionHelper:
    """Enhanced helper class for Google Cloud Vision API operations."""
    
    def __init__(self, credentials_path=None, output_dir="./pdf_output"):
        """Initialize Google Vision API client with proper configuration.
        
        Args:
            credentials_path: Path to Google Cloud credentials JSON file
            output_dir: Directory for output files
        """
        self.client = None
        self.output_dir = output_dir
        self.workspace = FileStorage(output_dir)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        if not GOOGLE_VISION_AVAILABLE:
            warning("Google Cloud Vision API is not available. Visual form detection disabled.")
            return
        
        try:
            if credentials_path:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            
            self.client = vision.ImageAnnotatorClient()
            self.component = GoogleVisionComponent(
                client=self.client,
                config=type('obj', (object,), {'pdf_output_dir': output_dir}),
                workspace=self.workspace
            )
            info("Google Vision API client initialized successfully")
        except Exception as e:
            error(f"Failed to initialize Google Vision API client: {e}")
    
    def is_available(self) -> bool:
        """Check if Google Vision API is available and initialized."""
        return GOOGLE_VISION_AVAILABLE and self.client is not None
    
    def detect_form_fields(self, image_path: str) -> List[Dict[str, Any]]:
        """Detect form fields in an image using Google Vision API."""
        if not self.is_available():
            warning("Google Vision API is not available. Cannot detect form fields.")
            return []
        
        thinking(f"Detecting form fields in image: {image_path}")
        
        try:
            # Create a temporary PDF from the image
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                temp_pdf_path = temp_pdf.name
                
            if PIL_AVAILABLE:
                img = Image.open(image_path)
                img.save(temp_pdf_path, "PDF")
            else:
                # Fallback using reportlab if PIL not available
                if REPORTLAB_AVAILABLE:
                    c = canvas.Canvas(temp_pdf_path, pagesize=letter)
                    c.drawImage(image_path, 0, 0)
                    c.save()
                else:
                    error("No library available to convert image to PDF")
                    return []
            
            # Use component to detect form fields
            result = self.component.analyze_pdf(temp_pdf_path)
            
            # Extract form fields
            form_fields = []
            if "form_fields" in result and 0 in result["form_fields"]:
                form_fields = result["form_fields"][0]
                
            # Clean up temporary PDF
            os.unlink(temp_pdf_path)
            
            return form_fields
        except Exception as e:
            error(f"Error detecting form fields with Google Vision: {e}")
            return []
    
    def detect_form_fields_from_pdf(self, pdf_path: str, page_num: int = 0) -> List[Dict[str, Any]]:
        """Extract a page from PDF and detect form fields using Google Vision."""
        if not self.is_available():
            warning("Google Vision API is not available. Cannot detect form fields.")
            return []
        
        thinking(f"Detecting form fields in PDF page {page_num}")
        
        try:
            # Use component to detect form fields
            result = self.component.analyze_pdf(pdf_path, [page_num])
            
            # Extract form fields for the specified page
            form_fields = []
            if "form_fields" in result and page_num in result["form_fields"]:
                form_fields = result["form_fields"][page_num]
                
            info(f"Detected {len(form_fields)} form fields on page {page_num}")
            return form_fields
        except Exception as e:
            error(f"Error detecting form fields from PDF: {e}")
            return []
    
    def verify_form_is_filled(self, pdf_path: str, page_num: int = 0) -> Dict[str, Any]:
        """Verify if a form's fields are filled using Google Vision API."""
        if not self.is_available():
            warning("Google Vision API is not available. Cannot verify form.")
            return {'status': 'error', 'message': 'Google Vision API not available'}
        
        thinking(f"Verifying if form is filled: {pdf_path}")
        
        try:
            # Use component to verify form
            verification = self.component.verify_form_is_complete(pdf_path)
            
            # Map to expected response format
            result = {
                'status': 'success',
                'is_complete': verification.get('is_complete', False),
                'total_fields': verification.get('total_fields', 0),
                'filled_fields_count': len(verification.get('filled_fields', [])),
                'empty_fields_count': len(verification.get('missing_fields', [])),
                'empty_field_names': [field.get('name') for field in verification.get('missing_fields', [])]
            }
            
            # Add appropriate message
            if result['is_complete']:
                result['message'] = "Form is complete. All fields are filled."
            else:
                result['message'] = f"Form is incomplete. {result['empty_fields_count']} fields are empty."
                
            return result
        except Exception as e:
            error(f"Error verifying form: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def fill_pdf_form(self, pdf_path: str, form_data: Dict[str, Any], output_path: str = None) -> Dict[str, Any]:
        """Fill a PDF form with the provided data."""
        if not self.is_available():
            warning("Google Vision API is not available. Cannot fill form.")
            return {'status': 'error', 'message': 'Google Vision API not available'}
        
        thinking(f"Filling PDF form: {pdf_path}")
        
        try:
            # Use component to fill form
            result = self.component.fill_pdf_form(pdf_path, form_data, output_path)
            
            if 'error' in result:
                return {'status': 'error', 'message': result['error']}
                
            filled_path = result.get('filled_pdf_path')
            verification = result.get('verification', {})
            
            return {
                'status': 'success',
                'message': 'Form filled successfully',
                'filled_pdf_path': filled_path,
                'is_complete': verification.get('is_complete', False),
                'missing_fields': verification.get('missing_fields', [])
            }
        except Exception as e:
            error(f"Error filling form: {e}")
            return {'status': 'error', 'message': str(e)}


class PDFFormHandler:
    """Handler for PDF form operations with Google Vision integration."""
    
    def __init__(self, verbose=True, google_vision_credentials=None, output_dir="./pdf_output"):
        self.verbose = verbose
        self.output_dir = output_dir
        self.available_libraries = {
            'PyPDF2': True,
            'PyMuPDF': PYMUPDF_AVAILABLE,
            'fillpdf': FILLPDFS_AVAILABLE,
            'pdftk': PDFTK_AVAILABLE,
            'Acrobat': ACROBAT_AVAILABLE,
            'GoogleVision': GOOGLE_VISION_AVAILABLE
        }
        
        # Create workspace
        self.workspace = FileStorage(output_dir)
        
        # Initialize Google Vision helper
        self.vision_helper = GoogleVisionHelper(
            credentials_path=google_vision_credentials,
            output_dir=output_dir
        )
        
        if verbose:
            info(f"Initialized PDFFormHandler with available libraries: {self.available_libraries}")
    
    def get_form_fields(self, file_path: str, detailed: bool = False, use_vision: bool = False) -> Dict:
        """Get all form fields from a PDF file with comprehensive detection."""
        if not os.path.exists(file_path):
            error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Results will store form fields from all available methods
        all_results = {}
        
        # Method 1: PyPDF2 (always available)
        try:
            reader = PdfReader(file_path)
            fields = reader.get_fields()
            
            if fields:
                form_fields = {}
                
                for name, field in fields.items():
                    field_type = "Unknown"
                    field_value = None
                    
                    # Determine field type and value
                    if hasattr(field, '/FT'):
                        ft = field['/FT']
                        if ft == '/Tx': field_type = "Text"
                        elif ft == '/Btn': field_type = "Button/Checkbox"
                        elif ft == '/Ch': field_type = "Choice/Dropdown"
                        elif ft == '/Sig': field_type = "Signature"
                    
                    # Get field value
                    if hasattr(field, '/V'):
                        field_value = field['/V']
                        # Convert PDF-specific objects to Python types
                        if isinstance(field_value, PyPDF2.generic.ByteStringObject):
                            field_value = str(field_value)
                        elif isinstance(field_value, PyPDF2.generic.NumberObject):
                            field_value = float(field_value)
                        elif isinstance(field_value, PyPDF2.generic.BooleanObject):
                            field_value = bool(field_value)
                    
                    # Store field info
                    if detailed:
                        form_fields[name] = {
                            'type': field_type,
                            'value': field_value,
                            'source': 'PyPDF2'
                        }
                    else:
                        form_fields[name] = field_value
                
                all_results.update(form_fields)
                info(f"Found {len(form_fields)} fields using PyPDF2")
        except Exception as e:
            warning(f"Error getting form fields with PyPDF2: {e}")
        
        # Method 2: PyMuPDF (if available)
        if PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(file_path)
                form_fields = {}
                
                # Get all widgets for all field types
                for page in doc:
                    for widget in page.widgets():
                        field_name = widget.field_name
                        field_type = widget.field_type_string
                        field_value = widget.field_value
                        
                        # Store field info
                        if detailed:
                            form_fields[field_name] = {
                                'type': field_type,
                                'value': field_value,
                                'source': 'PyMuPDF'
                            }
                        else:
                            form_fields[field_name] = field_value
                
                doc.close()
                
                # Update results
                all_results.update(form_fields)
                info(f"Found {len(form_fields)} fields using PyMuPDF")
            except Exception as e:
                warning(f"Error getting form fields with PyMuPDF: {e}")
        
        # Method 3: Use Google Vision for visual form field detection
        if use_vision and self.vision_helper.is_available():
            try:
                # Process first page for now
                vision_fields = self.vision_helper.detect_form_fields_from_pdf(file_path, 0)
                
                if vision_fields:
                    # Convert Google Vision fields to match our format
                    form_fields = {}
                    
                    for idx, field in enumerate(vision_fields):
                        field_name = field.get('name', f"Field_{idx+1}").replace(' ', '_')
                        field_value = "Filled" if field.get('is_filled', False) else None
                        
                        if detailed:
                            form_fields[field_name] = {
                                'type': field.get('type', 'Unknown'),
                                'value': field_value,
                                'source': 'GoogleVision',
                                'bbox': field.get('bbox', {})
                            }
                        else:
                            form_fields[field_name] = field_value
                    
                    # Add Google Vision fields that don't already exist
                    for name, value in form_fields.items():
                        if name not in all_results:
                            all_results[name] = value
                    
                    info(f"Found {len(form_fields)} fields using Google Vision")
            except Exception as e:
                warning(f"Error detecting form fields with Google Vision: {e}")
        
        return all_results
    
    def verify_form_is_filled(self, file_path: str, page_num: int = 0) -> Dict[str, Any]:
        """Verify if a PDF form's fields are properly filled using Google Vision."""
        if not self.vision_helper.is_available():
            return {'status': 'error', 'message': 'Google Vision API not available'}
            
        return self.vision_helper.verify_form_is_filled(file_path, page_num)
    
    def fill_form(
        self, 
        file_path: str, 
        form_data: Dict[str, Any], 
        output_path: str,
        flatten: bool = False,
        verify: bool = False
    ) -> Tuple[bool, str]:
        """Fill a PDF form with data using multiple fallback methods."""
        if not os.path.exists(file_path):
            error(f"File not found: {file_path}")
            return False, f"File not found: {file_path}"
            
        if not form_data:
            warning("No form data provided")
            return False, "No form data provided"
        
        thinking(f"Filling form with {len(form_data)} fields")
        
        # Create output directory if needed
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        
        # Try Google Vision first if available
        if self.vision_helper.is_available():
            try:
                result = self.vision_helper.fill_pdf_form(file_path, form_data, output_path)
                
                if result.get('status') == 'success':
                    filled_path = result.get('filled_pdf_path')
                    if verify:
                        verification = self.vision_helper.verify_form_is_filled(filled_path)
                        if verification.get('is_complete', False):
                            success(f"Successfully filled and verified form using Google Vision")
                        else:
                            warning(f"Form filled but verification found missing fields: {verification.get('empty_field_names', [])}")
                    else:
                        success(f"Successfully filled form using Google Vision")
                    
                    return True, "Successfully filled form using Google Vision"
            except Exception as e:
                warning(f"Error filling form with Google Vision: {e}")
        
        # Try multiple methods for form filling
        methods_tried = ["Google Vision"]  # Already tried above
        error_messages = []
        
        # Try Adobe Acrobat if available
        if ACROBAT_AVAILABLE:
            methods_tried.append("Adobe Acrobat")
            try:
                # Create Acrobat application
                acrobat = win32com.client.Dispatch('AcroExch.App')
                avdoc = win32com.client.Dispatch('AcroExch.AVDoc')
                
                # Open the PDF
                if avdoc.Open(os.path.abspath(file_path), ''):
                    pdfdoc = avdoc.GetPDDoc()
                    jso = pdfdoc.GetJSObject()
                    
                    # Fill fields
                    filled_count = 0
                    for field_name, value in form_data.items():
                        try:
                            field = jso.getField(field_name)
                            if field:
                                field.value = value
                                filled_count += 1
                        except:
                            pass
                    
                    # Save the filled PDF
                    if pdfdoc.Save(1, os.path.abspath(output_path)):
                        # Close the PDF
                        avdoc.Close(True)
                        
                        # Flatten if requested
                        if flatten:
                            if avdoc.Open(os.path.abspath(output_path), ''):
                                pdfdoc = avdoc.GetPDDoc()
                                jso = pdfdoc.GetJSObject()
                                try:
                                    jso.flattenPages()
                                    pdfdoc.Save(1, os.path.abspath(output_path))
                                except:
                                    warning("Could not flatten with Acrobat JavaScript")
                                avdoc.Close(True)
                        
                        acrobat.Exit()
                        
                        success(f"Successfully filled {filled_count} form fields using Adobe Acrobat")
                        
                        # Verify if requested
                        if verify:
                            self.verify_form_is_filled(output_path, 0)
                        
                        return True, f"Successfully filled form using Adobe Acrobat"
                    else:
                        avdoc.Close(True)
                        acrobat.Exit()
                        error_messages.append("Failed to save PDF with Adobe Acrobat")
                else:
                    acrobat.Exit()
                    error_messages.append("Failed to open PDF with Adobe Acrobat")
            except Exception as e:
                error_messages.append(f"Error using Adobe Acrobat: {str(e)}")
        
        # Use fillpdf if available
        if FILLPDFS_AVAILABLE:
            methods_tried.append("fillpdf")
            try:
                # Fill the form
                fillpdfs.write_fillable_pdf(file_path, output_path, form_data, flatten=flatten)
                
                # Verify the form was filled correctly
                if os.path.exists(output_path):
                    success(f"Successfully filled form fields using fillpdf")
                    
                    # Verify if requested
                    if verify:
                        self.verify_form_is_filled(output_path, 0)
                    
                    return True, f"Successfully filled form using fillpdf"
            except Exception as e:
                error_messages.append(f"Error using fillpdf: {str(e)}")
        
        # Use PyMuPDF if available
        if PYMUPDF_AVAILABLE:
            methods_tried.append("PyMuPDF")
            try:
                doc = fitz.open(file_path)
                
                # Get all form fields
                form_fields = {}
                # Get text fields
                text_fields = doc.get_form_text_fields()
                form_fields.update(text_fields)
                
                # Get all widgets for other field types
                widgets_by_name = {}
                for page in doc:
                    for widget in page.widgets():
                        widgets_by_name[widget.field_name] = widget
                
                # Fill form fields
                filled_fields = []
                for field_name, value in form_data.items():
                    if field_name in text_fields:
                        # Fill text field
                        doc.set_form_text_fields({field_name: value})
                        filled_fields.append(field_name)
                    elif field_name in widgets_by_name:
                        # Fill other field types
                        widget = widgets_by_name[field_name]
                        
                        # Handle different field types
                        if widget.field_type == fitz.PDF_WIDGET_TYPE_CHECKBOX:
                            # For checkbox, value should be True/False or string from choice_options
                            if isinstance(value, bool):
                                widget.field_value = value
                            elif isinstance(value, str) and value.lower() in ('yes', 'true', '1', 'on'):
                                widget.field_value = True
                            else:
                                widget.field_value = False
                            filled_fields.append(field_name)
                        elif widget.field_type == fitz.PDF_WIDGET_TYPE_CHOICE:
                            if value in widget.choice_options:
                                widget.field_value = value
                                filled_fields.append(field_name)
                
                # Save the document
                if flatten:
                    doc.flatten()
                doc.save(output_path)
                doc.close()
                
                if filled_fields:
                    success(f"Successfully filled {len(filled_fields)} form fields using PyMuPDF")
                    
                    # Verify if requested
                    if verify:
                        self.verify_form_is_filled(output_path, 0)
                    
                    return True, f"Successfully filled form using PyMuPDF"
            except Exception as e:
                error_messages.append(f"Error using PyMuPDF: {str(e)}")
        
        # Fall back to PyPDF2
        methods_tried.append("PyPDF2")
        try:
            reader = PdfReader(file_path)
            writer = PdfWriter()
            
            # Copy all pages from the original PDF
            for page in reader.pages:
                writer.add_page(page)
            
            # Update form fields
            writer.update_page_form_field_values(
                writer.pages[0], {k: v for k, v in form_data.items()}
            )
            
            # Flatten form if requested
            if flatten:
                writer.flatten_form_fields()
            
            # Write the filled PDF
            with open(output_path, "wb") as f:
                writer.write(f)
            
            success(f"Successfully filled form fields using PyPDF2")
            
            # Verify if requested
            if verify:
                self.verify_form_is_filled(output_path, 0)
            
            return True, f"Successfully filled form using PyPDF2"
        except Exception as e:
            error_messages.append(f"Error using PyPDF2: {str(e)}")
        
        # If all methods failed, return the errors
        error(f"Failed to fill form fields using any available method ({', '.join(methods_tried)})")
        return False, f"Failed to fill form fields. Tried: {', '.join(methods_tried)}"


class WebScraper:
    """Component for web scraping operations."""
    
    def __init__(self, user_agent=None):
        self.user_agent = user_agent or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    
    def extract_data(self, url: str, extract_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract data from a webpage."""
        if not WEB_SCRAPING_AVAILABLE:
            return {"error": "Web scraping libraries not available"}
            
        if extract_options is None:
            extract_options = {
                'title': True,
                'description': True,
                'links': True,
                'text': True,
                'tables': True,
                'images': False,
                'selectors': {}
            }
        
        thinking(f"Extracting data from {url}")
        
        try:
            headers = {'User-Agent': self.user_agent}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            result = {
                'url': url,
                'status': response.status_code,
                'content_type': response.headers.get('Content-Type', '')
            }
            
            # Extract title
            if extract_options.get('title', True):
                result['title'] = soup.title.text.strip() if soup.title else "No title found"
            
            # Extract meta description
            if extract_options.get('description', True):
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc and 'content' in meta_desc.attrs:
                    result['description'] = meta_desc['content']
                else:
                    result['description'] = None
            
            # Extract links
            if extract_options.get('links', True):
                links = soup.find_all('a', href=True)
                result['links'] = [{'text': link.text.strip(), 'href': link['href']} for link in links]
            
            # Extract main content text
            if extract_options.get('text', True):
                main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
                if main_content:
                    result['main_text'] = main_content.get_text(strip=True)
                else:
                    result['main_text'] = soup.body.get_text(strip=True) if soup.body else ""
            
            # Extract tables
            if extract_options.get('tables', True):
                tables = soup.find_all('table')
                result['tables'] = []
                
                for table in tables:
                    table_data = {'headers': [], 'rows': []}
                    
                    # Extract headers
                    headers = [th.text.strip() for th in table.find_all('th')]
                    if headers:
                        table_data['headers'] = headers
                    
                    # Extract rows
                    rows = []
                    for tr in table.find_all('tr'):
                        row = [td.text.strip() for td in tr.find_all('td')]
                        if row:
                            rows.append(row)
                    
                    table_data['rows'] = rows
                    result['tables'].append(table_data)
            
            # Extract images
            if extract_options.get('images', False):
                images = soup.find_all('img', src=True)
                result['images'] = [{'alt': img.get('alt', ''), 'src': img['src']} for img in images]
            
            # Extract data based on custom selectors
            custom_selectors = extract_options.get('selectors', {})
            if custom_selectors:
                result['custom'] = {}
                
                for name, selector in custom_selectors.items():
                    elements = soup.select(selector)
                    if elements:
                        result['custom'][name] = [element.text.strip() for element in elements]
                    else:
                        result['custom'][name] = []
            
            success(f"Successfully extracted data from {url}")
            return result
            
        except requests.exceptions.RequestException as e:
            error(f"Error fetching URL {url}: {e}")
            return {'error': f"Error fetching URL: {str(e)}"}
        
        except Exception as e:
            error(f"Error extracting data from {url}: {e}")
            return {'error': f"Error extracting data: {str(e)}"}


class SystemComponent:
    """Component for system operations and commands."""
    
    def __init__(self, allow_shell=True):
        self.allow_shell = allow_shell
    
    def execute_command(self, command: str, timeout: int = 60) -> Dict[str, Any]:
        """Execute a shell command with safety checks."""
        if not self.allow_shell:
            warning("Shell command execution is disabled")
            return {
                'success': False,
                'stderr': 'Shell command execution is disabled',
                'returncode': -1
            }
        
        thinking(f"Executing command: {command}")
        
        try:
            import subprocess
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
        except subprocess.TimeoutExpired:
            error(f"Command timed out after {timeout} seconds")
            return {
                'success': False,
                'stderr': f'Command timed out after {timeout} seconds',
                'returncode': -1
            }
        except Exception as e:
            error(f"Error executing command: {e}")
            return {
                'success': False,
                'stderr': f'Error executing command: {str(e)}',
                'returncode': -1
            }


def main():
    """Command line interface for the PDF form operations."""
    parser = argparse.ArgumentParser(description="PDF Form Handler with Google Vision Integration")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List form fields command
    list_parser = subparsers.add_parser("list-fields", help="List all form fields in a PDF")
    list_parser.add_argument("input", help="Input PDF file")
    list_parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed field information")
    list_parser.add_argument("--use-vision", "-v", action="store_true", help="Use Google Vision to detect fields visually")
    
    # Fill form command
    fill_parser = subparsers.add_parser("fill-form", help="Fill a PDF form with data")
    fill_parser.add_argument("input", help="Input PDF file")
    fill_parser.add_argument("--output", "-o", required=True, help="Output PDF file")
    fill_parser.add_argument("--data", "-d", required=True, help="JSON string or file with form data")
    fill_parser.add_argument("--flatten", "-f", action="store_true", help="Flatten the form after filling")
    fill_parser.add_argument("--verify", "-v", action="store_true", help="Verify form is properly filled using Google Vision")
    
    # Verify form fields command
    verify_parser = subparsers.add_parser("verify-form", help="Verify if a form is properly filled")
    verify_parser.add_argument("input", help="Input PDF file")
    
    # Scrape webpage command
    scrape_parser = subparsers.add_parser("scrape", help="Scrape data from a webpage")
    scrape_parser.add_argument("url", help="URL to scrape")
    scrape_parser.add_argument("--output", "-o", help="Output JSON file for scraped data")
    
    # Execute system command
    sys_parser = subparsers.add_parser("exec", help="Execute a system command")
    sys_parser.add_argument("command", help="Command to execute")
    sys_parser.add_argument("--timeout", "-t", type=int, default=60, help="Command timeout in seconds")
    
    # Google Vision API credentials
    parser.add_argument("--vision-credentials", help="Path to Google Cloud Vision API credentials JSON file")
    parser.add_argument("--output-dir", help="Directory for output files", default="./pdf_output")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create handler instances
    form_handler = PDFFormHandler(
        google_vision_credentials=args.vision_credentials,
        output_dir=args.output_dir
    )
    web_scraper = WebScraper()
    system = SystemComponent()
    
    try:
        if args.command == "list-fields":
            # List form fields
            speaking(f"Listing form fields in {args.input}")
            fields = form_handler.get_form_fields(args.input, detailed=args.detailed, use_vision=args.use_vision)
            
            print(f"\n{Fore.CYAN}Form fields in {args.input}:{Style.RESET_ALL}")
            for name, value in fields.items():
                if args.detailed and isinstance(value, dict):
                    print(f"{Fore.CYAN}Field: {name}{Style.RESET_ALL}")
                    print(f"  Type: {value['type']}")
                    print(f"  Value: {value['value']}")
                    print(f"  Source: {value['source']}")
                    print()
                else:
                    print(f"  {name}: {value}")
        
        elif args.command == "fill-form":
            # Parse form data
            speaking(f"Filling form {args.input} with provided data")
            import json
            if os.path.exists(args.data):
                thinking(f"Loading form data from file: {args.data}")
                with open(args.data, 'r') as f:
                    form_data = json.load(f)
            else:
                thinking("Parsing form data from JSON string")
                try:
                    form_data = json.loads(args.data)
                except json.JSONDecodeError:
                    error("Invalid JSON data")
                    return 1
            
            # Fill the form
            success_flag, message = form_handler.fill_form(
                args.input, 
                form_data, 
                args.output,
                flatten=args.flatten,
                verify=args.verify
            )
            
            if success_flag:
                success(message)
            else:
                error(message)
                return 1
        
        elif args.command == "verify-form":
            # Verify if a form is properly filled
            speaking(f"Verifying if form is properly filled: {args.input}")
            
            # Verify the form
            results = form_handler.verify_form_is_filled(args.input)
            
            print(f"\n{Fore.CYAN}Form verification results:{Style.RESET_ALL}")
            print(f"  Status: {results.get('status', 'unknown')}")
            print(f"  Message: {results.get('message', '')}")
            
            if results.get('empty_fields'):
                print(f"\n{Fore.YELLOW}Empty fields:{Style.RESET_ALL}")
                for field in results.get('empty_fields', []):
                    print(f"  - {field}")
        
        elif args.command == "scrape":
            # Scrape webpage
            speaking(f"Scraping data from {args.url}")
            result = web_scraper.extract_data(args.url)
            
            # Output results
            if args.output:
                thinking(f"Saving scraped data to {args.output}")
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)
                success(f"Scraped data saved to {args.output}")
            else:
                success("Scraped data retrieved successfully")
                print(json.dumps(result, indent=2))
        
        elif args.command == "exec":
            # Execute system command
            speaking(f"Executing system command: {args.command}")
            result = system.execute_command(args.command, timeout=args.timeout)
            
            if result['success']:
                success("Command executed successfully")
                print(result['stdout'])
            else:
                error(f"Command failed with error:")
                print(result['stderr'])
                return result['returncode']
        
        else:
            # No command or unrecognized command
            parser.print_help()
            return 1
        
        return 0
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    
    except Exception as e:
        error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())