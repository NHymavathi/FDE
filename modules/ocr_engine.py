"""
modules/ocr_engine.py

Provides OCR capabilities using EasyOCR for scanned PDFs and image files (PNG, JPG, JPEG).
Decouples OCR initialization and image processing from text extraction routing.
"""
import io
from typing import Dict, Any
import numpy as np
from PIL import Image
import pymupdf

# Lazy singleton EasyOCR reader instance to prevent repeated model loading
_ocr_reader = None

def get_ocr_reader():
    """
    Lazily initializes and returns the EasyOCR Reader instance.
    Uses English ('en') language model on CPU for maximum compatibility.
    """
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        # gpu=False ensures robust CPU execution across environments; verbose=False prevents stdout charmap errors on Windows
        _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _ocr_reader

def ocr_image_bytes(image_bytes: bytes) -> str:
    """
    Performs OCR on raw image bytes (PNG, JPG, JPEG).

    Args:
        image_bytes (bytes): Binary image content.

    Returns:
        str: Extracted text from the image.
    """
    reader = get_ocr_reader()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    text_lines = reader.readtext(image_np, detail=0)
    return "\n".join(text_lines).strip()

def ocr_pdf_bytes(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Rasterizes PDF pages into in-memory images and applies EasyOCR on each page.

    Args:
        pdf_bytes (bytes): Binary PDF content.

    Returns:
        dict: Result containing extracted text, page count, and status.
    """
    try:
        reader = get_ocr_reader()
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(doc)
        full_text = []

        for page_num in range(page_count):
            page = doc.load_page(page_num)
            # Render page to an image pixmap at 150 DPI for good OCR accuracy
            pix = page.get_pixmap(dpi=150)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                
            img_np = np.array(img)
            lines = reader.readtext(img_np, detail=0)
            if lines:
                full_text.append(f"--- Page {page_num + 1} ---\n" + "\n".join(lines))

        doc.close()
        combined_text = "\n\n".join(full_text).strip()

        return {
            "success": True,
            "text": combined_text,
            "page_count": page_count,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "text": "",
            "page_count": 0,
            "error": f"EasyOCR error on PDF: {str(e)}"
        }
