# Intelligent Document Extraction & Structuring Agent

An enterprise-grade, modular MVP built with Python, Streamlit, PyMuPDF, EasyOCR, Groq LLM, Pydantic, and SQLite.

## Core Features
1. **Multi-Format Document Parsing**: Parses text-based PDFs, scanned image PDFs, and image files (`.png`, `.jpg`, `.jpeg`).
2. **PyMuPDF Text Extraction with EasyOCR Fallback**: Automatically evaluates extracted character count against a configurable threshold (default: 50 characters) to switch between direct text parsing and OCR fallback.
3. **Groq LLM Classification (`temperature=0`)**: Classifies documents into `INVOICE`, `PURCHASE_ORDER`, `RESUME`, or `UNKNOWN`.
4. **Pydantic Schema Validation**: Enforces strict structured JSON extraction matching schema models.
5. **Field Completeness Scoring & Quality Routing**: Calculates a heuristic score (`populated fields / total fields`). Automatically flags low-completeness (<70%) or invalid documents as `REQUIRES_REVIEW`.
6. **Interactive Human Review Workflow**: Streamlit UI allowing users to inspect, edit, approve, and save extracted fields.
7. **SQLite Persistence**: Stores document metadata, extraction method, structured JSON payload, completeness score, status, and created timestamp with a built-in **Processing History** dashboard.

---

## Architecture Diagram

```
Upload Document (PDF / Image)
            │
            ▼
┌─────────────────────────┐
│ PyMuPDF Text Extraction │
└───────────┬─────────────┘
            │
  ┌─────────┴─────────┐
  │ Text >= Threshold │
  │    (50 chars)?    │
  └─────────┬─────────┘
      YES   │   NO (Scanned)
 ┌──────────┴──────────┐
 ▼                     ▼
[ Direct PyMuPDF ]   [ EasyOCR Fallback ]
 (method: pymupdf)     (method: ocr)
 └──────────┬──────────┘
            ▼
  [ Groq LLM Classification ]
  (INVOICE | PURCHASE_ORDER | RESUME | UNKNOWN)
            │
            ▼
  [ Schema Router & LLM Extraction ]
  (Structured JSON matching schema)
            │
            ▼
  [ Pydantic Validation & Completeness Score ]
  (Completeness = Populated Fields / Total Fields)
            │
  ┌─────────┴─────────┐
  │ Score >= 0.7 &    │
  │ Valid Schema?     │
  └─────────┬─────────┘
      YES   │   NO
 ┌──────────┴──────────┐
 ▼                     ▼
[ Ready for Approval ] [ Requires Human Review ]
 │                     │ (Editable UI Form)
 └──────────┬──────────┘
            ▼
   [ SQLite Persistence ]
   (Save record & view in History UI)
```

---

## Project Structure

```
FDE/
├── .env.example                # Template for GROQ_API_KEY
├── .env                        # Environment variables
├── requirements.txt            # Project dependencies
├── README.md                   # Complete documentation
├── app.py                      # Main Streamlit Dashboard UI
├── database.py                 # SQLite persistence & query manager
├── schemas.py                  # Pydantic schemas (Invoice, PO, Resume)
├── create_samples.py           # Synthetic sample PDF/image generator
├── modules/                    # Business logic modules
│   ├── __init__.py
│   ├── extractor.py            # Document extraction router (PyMuPDF + OCR)
│   ├── ocr_engine.py           # EasyOCR engine wrapper
│   ├── classifier.py           # Groq LLM classifier (temperature=0)
│   ├── structured_extractor.py # Groq LLM schema extraction engine
│   └── validator.py            # Field completeness score & review status logic
└── data/                       # Database storage folder
    └── processed_docs.db
```

---

## Installation & Setup

1. **Clone/Navigate to Project Folder**:
   ```bash
   cd FDE
   ```

2. **Install Dependencies**:
   ```bash
   python -m pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the project root:
   ```ini
   GROQ_API_KEY=gsk_your_groq_api_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   ```
   *(Or provide your API key directly in the Streamlit sidebar).*

4. **Generate Synthetic Test Documents**:
   ```bash
   python create_samples.py
   ```
   This generates:
   - `sample_docs/invoice.pdf` (Digital PDF Invoice)
   - `sample_docs/purchase_order.pdf` (Digital PDF Purchase Order)
   - `sample_docs/resume.pdf` (Digital PDF Resume)
   - `sample_docs/scanned_invoice.pdf` (Scanned Image PDF - Triggers EasyOCR)
   - `sample_docs/scanned_invoice.png` (Direct Image - Triggers EasyOCR)

5. **Launch Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```

---

## Project Assumptions

1. **MVP Scope**: Due to the 24-hour assessment constraint, the MVP supports Invoice, Purchase Order, and Resume documents only.
2. **Language Support**: The MVP primarily supports English-language documents.
3. **File Format Support**: Documents may be text-based PDFs or scanned PDFs/images.
4. **Extraction Routing**: Direct text extraction is attempted first, with OCR used as a fallback when meaningful text cannot be extracted.
5. **Semantic Processing**: The LLM performs semantic classification and information extraction.
6. **Data Validation**: Pydantic validates extracted structured data.
7. **Field Completeness Metric**: The completeness score is a heuristic based on populated fields and is not a true model confidence score.
8. **Human Review Workflow**: Documents below the configured completeness threshold are routed to a human review workflow.
9. **Synthetic Data**: All documents and data used in the MVP are synthetic.
