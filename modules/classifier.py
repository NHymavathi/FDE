"""
modules/classifier.py

Handles document classification using Groq LLM API.
Classifies extracted document text into exactly one of:
- INVOICE
- PURCHASE_ORDER
- RESUME
- UNKNOWN
"""
import os
import re
from typing import Dict, Any
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

VALID_CATEGORIES = {"INVOICE", "PURCHASE_ORDER", "RESUME", "UNKNOWN"}
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "groq/compound-mini")
MAX_INPUT_CHARS = 3000

SYSTEM_PROMPT = """You are an expert document classification AI.
Classify the provided document text into EXACTLY ONE of the following valid categories:
- INVOICE: Invoices, billing statements, payment requests from vendors.
- PURCHASE_ORDER: Purchase orders, PO requisitions, commercial orders issued by buyers.
- RESUME: CVs, resumes, candidate work histories, professional profiles.
- UNKNOWN: Any document that does not fit INVOICE, PURCHASE_ORDER, or RESUME.

CRITICAL REQUIREMENT:
Respond with ONLY the exact category name in uppercase (INVOICE, PURCHASE_ORDER, RESUME, or UNKNOWN).
Do NOT include any punctuation, explanation, markdown formatting, or extra text."""

def classify_document(text: str, api_key: str = None, model: str = None) -> Dict[str, Any]:
    """
    Classifies extracted document text into a standard document category using Groq LLM.

    Args:
        text (str): Extracted document text.
        api_key (str, optional): Groq API key.
        model (str, optional): Groq LLM model name.

    Returns:
        dict: A dictionary containing category, model, and status information.
    """
    load_dotenv(override=True)
    effective_api_key = api_key.strip() if (api_key and api_key.strip()) else os.getenv("GROQ_API_KEY")
    target_model = model.strip() if (model and model.strip()) else os.getenv("GROQ_MODEL", "groq/compound-mini")

    if not effective_api_key or effective_api_key.strip() == "" or effective_api_key.startswith("your_"):
        return {
            "category": "UNKNOWN",
            "model": target_model,
            "error": "GROQ_API_KEY is missing or invalid. Please check your API key."
        }

    if not text or not text.strip():
        return {
            "category": "UNKNOWN",
            "model": target_model,
            "error": "Input text is empty. Cannot classify document."
        }

    truncated_text = text.strip()[:MAX_INPUT_CHARS]

    try:
        client = Groq(api_key=effective_api_key)
        
        max_retries = 3
        backoff = 2
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Document Text:\n\n{truncated_text}"}
                    ],
                    model=target_model,
                    temperature=0,
                    max_tokens=20
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

        raw_output = response.choices[0].message.content.strip().upper()
        
        # Clean potential markdown backticks or quotes
        clean_output = re.sub(r'[^A-Z_]', '', raw_output.replace(" ", "_"))

        if clean_output in VALID_CATEGORIES:
            final_category = clean_output
        else:
            # Fallback search for category word inside raw response
            final_category = "UNKNOWN"
            for cat in ["INVOICE", "PURCHASE_ORDER", "RESUME"]:
                if cat in raw_output:
                    final_category = cat
                    break

        return {
            "category": final_category,
            "model": target_model,
            "raw_response": raw_output,
            "error": None
        }

    except Exception as e:
        # Fallback gracefully to UNKNOWN if LLM call fails
        return {
            "category": "UNKNOWN",
            "model": target_model,
            "error": f"Groq API Error: {str(e)}"
        }
