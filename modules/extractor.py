"""
modules/extractor.py

Main Document Extraction Router for Phase 2.
Attempts direct text extraction using PyMuPDF first.
If text is missing or below minimum character threshold, falls back to EasyOCR.
Supports both PDF and Image files (PNG, JPG, JPEG).
"""
import os
from typing import Dict, Any
import pymupdf
from modules.ocr_engine import ocr_pdf_bytes, ocr_image_bytes

# Default minimum character threshold to trigger OCR fallback
DEFAULT_MIN_CHAR_THRESHOLD = 50

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def extract_text_from_pdf_pymupdf(file_bytes: bytes) -> Dict[str, Any]:
    """
    Extracts text directly from a PDF using PyMuPDF text layer.
    """
    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        extracted_text = ""
        page_count = len(doc)
        
        for page_num in range(page_count):
            page = doc.load_page(page_num)
            page_text = page.get_text("text")
            if page_text:
                extracted_text += page_text + "\n"
        
        doc.close()
        cleaned_text = extracted_text.strip()
        return {
            "success": True,
            "text": cleaned_text,
            "character_count": len(cleaned_text),
            "page_count": page_count,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "text": "",
            "character_count": 0,
            "page_count": 0,
            "error": str(e)
        }

def process_document(
    file_bytes: bytes, 
    filename: str, 
    min_char_threshold: int = DEFAULT_MIN_CHAR_THRESHOLD
) -> Dict[str, Any]:
    """
    Unified document processing function with OCR fallback decision logic.

    Decision Logic:
    1. If file is an Image (.png, .jpg, .jpeg, etc.):
       -> Route directly to EasyOCR.
       -> Method = "ocr"
    2. If file is a PDF:
       -> Step 2A: Attempt direct text extraction with PyMuPDF.
       -> Step 2B: If extracted text count >= min_char_threshold:
          Use PyMuPDF text.
          -> Method = "pymupdf"
       -> Step 2C: If text is empty or below min_char_threshold (scanned/image PDF):
          Fallback to EasyOCR page rendering.
          -> Method = "ocr"

    Args:
        file_bytes (bytes): Raw binary content of the file.
        filename (str): Name of the uploaded file.
        min_char_threshold (int): Minimum characters required to consider direct text valid.

    Returns:
        dict: Containing extracted_text, character_count, extraction_method ("pymupdf" or "ocr"),
              page_count, and status information.
    """
    ext = os.path.splitext(filename.lower())[1]

    # --- Case 1: Image Files ---
    if ext in IMAGE_EXTENSIONS:
        try:
            ocr_text = ocr_image_bytes(file_bytes)
            return {
                "success": True,
                "text": ocr_text,
                "character_count": len(ocr_text),
                "extraction_method": "ocr",
                "page_count": 1,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "text": "",
                "character_count": 0,
                "extraction_method": "ocr",
                "page_count": 1,
                "error": f"Image OCR extraction failed: {str(e)}"
            }

    # --- Case 2: PDF Files ---
    # Attempt 1: Direct text extraction with PyMuPDF
    pymupdf_res = extract_text_from_pdf_pymupdf(file_bytes)
    
    if pymupdf_res["success"] and pymupdf_res["character_count"] >= min_char_threshold:
        return {
            "success": True,
            "text": pymupdf_res["text"],
            "character_count": pymupdf_res["character_count"],
            "extraction_method": "pymupdf",
            "page_count": pymupdf_res["page_count"],
            "error": None
        }

    # Attempt 2: Fallback to EasyOCR for scanned / low-character PDFs
    ocr_res = ocr_pdf_bytes(file_bytes)
    if ocr_res["success"]:
        ocr_text = ocr_res["text"]
        return {
            "success": True,
            "text": ocr_text,
            "character_count": len(ocr_text),
            "extraction_method": "ocr",
            "page_count": ocr_res["page_count"],
            "error": None
        }

    # Failure case
    return {
        "success": False,
        "text": "",
        "character_count": 0,
        "extraction_method": "unknown",
        "page_count": pymupdf_res.get("page_count", 0),
        "error": f"Extraction failed. PyMuPDF error: {pymupdf_res.get('error')}. OCR error: {ocr_res.get('error')}"
    }
