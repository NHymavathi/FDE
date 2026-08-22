"""
modules/structured_extractor.py

Handles schema-based structured information extraction using Groq LLM API.
Maps classified document categories to Pydantic schemas and enforces strict JSON output.
"""
import os
import json
import re
from typing import Dict, Any, Tuple
from dotenv import load_dotenv
from groq import Groq
from schemas import SCHEMA_MAP, InvoiceSchema, PurchaseOrderSchema, ResumeSchema, UnknownSchema

load_dotenv()

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "groq/compound-mini")
MAX_EXTRACTION_CHARS = 4000

SCHEMA_PROMPTS = {
    "INVOICE": """Extract the following fields from the invoice document into a valid JSON object:
- invoice_number (string or null): Unique invoice identifier/number.
- vendor_name (string or null): Name of the vendor or issuing company.
- invoice_date (string or null): Date of invoice issue (format YYYY-MM-DD if possible).
- total_amount (number/float or null): Final total amount due (numeric value only, e.g. 11000.00).
- tax_amount (number/float or null): Tax portion of amount (numeric value only, e.g. 1000.00).""",

    "PURCHASE_ORDER": """Extract the following fields from the purchase order document into a valid JSON object:
- po_number (string or null): Purchase Order (PO) identifier/number.
- buyer_name (string or null): Name of the buyer/purchasing organization.
- supplier_name (string or null): Name of the supplier/vendor.
- total_amount (number/float or null): Total monetary amount of the PO (numeric value only, e.g. 8000.00).""",

    "RESUME": """Extract the following fields from the resume/CV document into a valid JSON object:
- candidate_name (string or null): Full name of the candidate.
- email (string or null): Primary email address of candidate.
- skills (list of strings): Array of key technical and professional skills mentioned (e.g. ["Python", "Streamlit"]).
- education (string or null): Summary of highest degree or university education.""",

    "UNKNOWN": """Extract a brief summary from the unclassified document into a valid JSON object:
- summary (string or null): One sentence summary of document content."""
}


def sanitize_parsed_data(data: Dict[str, Any], category: str) -> Dict[str, Any]:
    """
    Sanitizes raw parsed LLM JSON before Pydantic validation:
    - Cleans currency symbols ($) and formatting commas (,) from float fields.
    - Converts comma-separated string skills to a list for RESUME schema.
    """
    if not isinstance(data, dict):
        return {}
    
    sanitized = dict(data)
    
    # Clean numeric fields
    numeric_fields = ["total_amount", "tax_amount"]
    for field in numeric_fields:
        if field in sanitized and sanitized[field] is not None:
            val = sanitized[field]
            if isinstance(val, str):
                cleaned = re.sub(r'[^\d.-]', '', val)
                try:
                    sanitized[field] = float(cleaned) if cleaned else None
                except ValueError:
                    sanitized[field] = None

    # Clean skills field for RESUME
    if category == "RESUME" and "skills" in sanitized:
        val = sanitized["skills"]
        if isinstance(val, str):
            sanitized["skills"] = [s.strip() for s in val.split(",") if s.strip()]
        elif val is None:
            sanitized["skills"] = []
            
    return sanitized


def extract_structured_data(text: str, category: str, api_key: str = None, model: str = None) -> Dict[str, Any]:
    """
    Extracts structured JSON data matching the category schema using Groq LLM.

    Args:
        text (str): Extracted document text.
        category (str): Document category ("INVOICE", "PURCHASE_ORDER", "RESUME", "UNKNOWN").
        api_key (str, optional): Groq API key.
        model (str, optional): Groq LLM model.

    Returns:
        dict: Extracted JSON data and status information.
    """
    load_dotenv(override=True)
    effective_api_key = api_key.strip() if (api_key and api_key.strip()) else os.getenv("GROQ_API_KEY")
    target_model = model.strip() if (model and model.strip()) else os.getenv("GROQ_MODEL", "groq/compound-mini")
    schema_cls = SCHEMA_MAP.get(category, UnknownSchema)
    schema_prompt = SCHEMA_PROMPTS.get(category, SCHEMA_PROMPTS["UNKNOWN"])

    if not effective_api_key or effective_api_key.strip() == "" or effective_api_key.startswith("your_"):
        # Default empty schema dict on missing API key
        empty_instance = schema_cls()
        return {
            "data": empty_instance.model_dump(),
            "schema_name": category,
            "is_valid": False,
            "validation_error": "GROQ_API_KEY is missing.",
            "raw_json_str": "{}"
        }

    truncated_text = text.strip()[:MAX_EXTRACTION_CHARS]

    system_instruction = f"""You are a precise data extraction AI.
{schema_prompt}

CRITICAL RULES:
1. Output MUST be valid JSON only. Do not wrap in markdown code blocks like ```json.
2. If a field is not explicitly present in the document text, return null (or [] for skills).
3. NEVER invent, guess, or hallucinate missing data values.
4. Extract numeric values for amounts (e.g., 1250.50 instead of "$1,250.50")."""

    try:
        client = Groq(api_key=effective_api_key)
        
        max_retries = 3
        backoff = 2
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": f"Document Text:\n\n{truncated_text}"}
                    ],
                    model=target_model,
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                break
            except Exception as call_err:
                err_str = str(call_err)
                if ("429" in err_str or "rate_limit" in err_str.lower()) and attempt < max_retries - 1:
                    match = re.search(r"try again in (\d+\.?\d*)s", err_str, re.IGNORECASE)
                    wait_time = float(match.group(1)) + 0.5 if match else backoff * (2 ** attempt)
                    if wait_time <= 8.0:
                        import time
                        time.sleep(wait_time)
                        continue
                raise call_err

        raw_json_str = response.choices[0].message.content.strip()

        # Clean potential markdown backticks
        clean_str = raw_json_str
        if "```" in clean_str:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_str, re.DOTALL)
            if match:
                clean_str = match.group(1)
            else:
                clean_str = clean_str.replace("```json", "").replace("```", "").strip()

        # Parse JSON
        parsed_dict = json.loads(clean_str)

        # Sanitize values before Pydantic validation
        sanitized_dict = sanitize_parsed_data(parsed_dict, category)

        # Validate with Pydantic model
        validated_model = schema_cls.model_validate(sanitized_dict)
        
        return {
            "data": validated_model.model_dump(),
            "schema_name": category,
            "is_valid": True,
            "validation_error": None,
            "raw_json_str": raw_json_str
        }

    except Exception as e:
        # If Pydantic validation or JSON parsing fails
        try:
            fallback_str = clean_str if 'clean_str' in locals() else (raw_json_str if 'raw_json_str' in locals() else "{}")
            raw_parsed = json.loads(fallback_str)
            fallback_dict = sanitize_parsed_data(raw_parsed, category)
        except Exception:
            fallback_dict = schema_cls().model_dump()

        return {
            "data": fallback_dict,
            "schema_name": category,
            "is_valid": False,
            "validation_error": f"Extraction / Validation error: {str(e)}",
            "raw_json_str": raw_json_str if 'raw_json_str' in locals() else "{}"
        }
