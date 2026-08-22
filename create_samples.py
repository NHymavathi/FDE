"""
create_samples.py

Generates synthetic PDF files for testing the extraction system:
1. Standard Text-based PDFs (PyMuPDF target)
2. Scanned Image PDF & PNG (EasyOCR target)
"""
import os
import pymupdf
from PIL import Image, ImageDraw, ImageFont

def create_sample_invoice(output_path: str):
    doc = pymupdf.open()
    page = doc.new_page()
    text = """
INVOICE

Invoice Number: INV-2026-001
Date: 2026-08-20
Vendor: Acme Tech Solutions Inc.
Vendor Address: 100 Innovation Way, Tech City, CA 94016

Bill To: Global Enterprises Corp.
Buyer Address: 500 Corporate Ave, New York, NY 10001

Itemized Breakdown:
1. Cloud Architecture Consulting Services - 40 hrs @ $150/hr = $6,000.00
2. Database Optimization & Indexing - 20 hrs @ $125/hr = $2,500.00
3. API Gateway Integration - 15 hrs @ $100/hr = $1,500.00

Subtotal: $10,000.00
Tax Amount (10%): $1,000.00
Total Amount Due: $11,000.00

Payment Terms: Net 30
Bank Details: Silicon Valley Bank, Account #987654321, Routing #123456789
Thank you for your business!
"""
    page.insert_text((50, 50), text.strip(), fontsize=11)
    doc.save(output_path)
    doc.close()
    print(f"Created {output_path}")

def create_sample_po(output_path: str):
    doc = pymupdf.open()
    page = doc.new_page()
    text = """
PURCHASE ORDER

PO Number: PO-998822
Order Date: 2026-08-15
Buyer Name: Apex Global Logistics
Supplier Name: Industrial Hardware Supplies Co.

Delivery Address: 750 Shipping Lane, Dock B, Chicago, IL 60601

Requested Items:
1. High-Density Server Racks - Qty: 5 @ $800 each = $4,000.00
2. Cat6e Ethernet Cables (100m spool) - Qty: 10 @ $150 each = $1,500.00
3. Uninterruptible Power Supply (UPS 3000VA) - Qty: 2 @ $1,250 each = $2,500.00

Total Amount: $8,000.00

Approved By: Sarah Jenkins, Procurement Manager
Notes: Please deliver before 2026-08-30.
"""
    page.insert_text((50, 50), text.strip(), fontsize=11)
    doc.save(output_path)
    doc.close()
    print(f"Created {output_path}")

def create_sample_resume(output_path: str):
    doc = pymupdf.open()
    page = doc.new_page()
    text = """
Alex Mercer
Email: alex.mercer@devmail.com | Phone: (555) 234-5678 | San Francisco, CA

SUMMARY
Experienced Senior Software Engineer with 6+ years specializing in Python, AI agents, cloud architectures, and full-stack web applications.

SKILLS
Programming Languages: Python, JavaScript, TypeScript, SQL
Frameworks & Libraries: Streamlit, Pydantic, PyMuPDF, EasyOCR, FastAPI, React
Cloud & Databases: SQLite, PostgreSQL, AWS (S3, Lambda), Docker
LLM Tools: Groq API, OpenAI SDK, LangChain, Prompt Engineering

EDUCATION
Bachelor of Science in Computer Science
University of California, Berkeley (2016 - 2020)

EXPERIENCE
Senior AI Systems Engineer | Tech Innovators Inc. | 2022 - Present
- Architected document extraction pipelines processing 50k+ PDFs daily using Python and LLMs.
- Optimized OCR fallback workflows reducing latency by 35%.

Software Engineer | DataCorp Solutions | 2020 - 2022
- Developed scalable REST APIs with FastAPI and SQLite/PostgreSQL.
"""
    page.insert_text((50, 50), text.strip(), fontsize=11)
    doc.save(output_path)
    doc.close()
    print(f"Created {output_path}")

def create_scanned_sample(output_png_path: str, output_pdf_path: str):
    """
    Renders text onto a PNG raster image, and packages it into a PDF without a text layer.
    Used specifically to test OCR fallback (EasyOCR).
    """
    img = Image.new("RGB", (750, 1000), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    text = """
SCANNED INVOICE (IMAGE ONLY)

Invoice Number: INV-SCAN-900
Date: 2026-08-21
Vendor: Scanned Imaging Systems LLC

Bill To: Test Client Corporation

Items:
1. Document Scanner Hardware - $1,200.00
2. OCR License Key - $300.00

Total Amount: $1,500.00
Tax: $0.00

Note: This file contains NO embedded text layer.
PyMuPDF direct extraction will return 0 characters.
EasyOCR fallback must be triggered to read this!
"""
    draw.text((40, 40), text.strip(), fill=(0, 0, 0))
    img.save(output_png_path)
    print(f"Created {output_png_path}")

    # Embed PNG in a PDF page to simulate a scanned PDF
    doc = pymupdf.open()
    rect = pymupdf.Rect(0, 0, 750, 1000)
    page = doc.new_page(width=750, height=1000)
    page.insert_image(rect, filename=output_png_path)
    doc.save(output_pdf_path)
    doc.close()
    print(f"Created {output_pdf_path}")

if __name__ == "__main__":
    os.makedirs("sample_docs", exist_ok=True)
    create_sample_invoice("sample_docs/invoice.pdf")
    create_sample_po("sample_docs/purchase_order.pdf")
    create_sample_resume("sample_docs/resume.pdf")
    create_scanned_sample("sample_docs/scanned_invoice.png", "sample_docs/scanned_invoice.pdf")
