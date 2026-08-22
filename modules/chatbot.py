"""
modules/chatbot.py

AI Document Chatbot module for the Intelligent Document Extraction Agent.
Handles conversational multi-turn Q&A using Groq LLM API with context injection
from current document extractions and SQLite processing history.
"""
import os
import re
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "groq/compound-mini")

SYSTEM_PROMPT_TEMPLATE = """You are an expert AI Document Assistant embedded inside the Intelligent Document Extraction & Structuring platform.
Your job is to answer user questions accurately, concisely, and professionally based on the provided document context and processing database history.

CONTEXT INFORMATION:
==================================================
{context_str}
==================================================

INSTRUCTIONS:
1. Always prioritize answers derived directly from the document text or structured extraction data above.
2. If the user asks a question about details contained in the document, quote or reference the specific field/value where helpful.
3. If the answer cannot be found in the document context, clearly inform the user while providing any relevant general assistance.
4. Keep answers clean, well-formatted (using markdown formatting like bolding, lists, tables), and easy to read.
"""

def build_context_string(
    document_context: Dict[str, Any] = None,
    history_context: List[Dict[str, Any]] = None
) -> str:
    """
    Builds a structured text block of document & database context to inject into LLM system prompt.
    """
    parts = []

    if document_context:
        filename = document_context.get("filename", "Uploaded Document")
        category = document_context.get("category", "UNKNOWN")
        text = document_context.get("text", "")
        extracted_data = document_context.get("extracted_data", {})

        doc_summary = f"CURRENTLY ACTIVE DOCUMENT:\n- Filename: {filename}\n- Category/Schema: {category}\n"
        if extracted_data:
            doc_summary += f"- Extracted Fields (JSON):\n{json.dumps(extracted_data, indent=2)}\n"
        if text:
            truncated_text = text.strip()[:3000]
            doc_summary += f"- Document Raw Text (Excerpt):\n{truncated_text}\n"
        parts.append(doc_summary)

    if history_context:
        hist_str = f"PERSISTED SQLITE DOCUMENT RECORDS ({len(history_context)} documents):\n"
        for i, item in enumerate(history_context[:10], 1):
            hist_str += f"{i}. ID: {item.get('document_id')} | File: {item.get('original_filename')} | Type: {item.get('document_type')} | Status: {item.get('status')} | Score: {round(item.get('completeness_score', 0)*100, 1)}%\n"
            if item.get("structured_data"):
                hist_str += f"   Data: {json.dumps(item.get('structured_data'))}\n"
        parts.append(hist_str)

    if not parts:
        return "No specific document uploaded or active in current context."

    return "\n".join(parts)


def generate_chatbot_response(
    messages: List[Dict[str, str]],
    document_context: Dict[str, Any] = None,
    history_context: List[Dict[str, Any]] = None,
    api_key: str = None,
    model: str = None
) -> Dict[str, Any]:
    """
    Generates a chatbot response using Groq LLM with context injection.

    Args:
        messages (list): Conversation messages list [{"role": "user"|"assistant", "content": "..."}].
        document_context (dict, optional): Current document text and extracted JSON.
        history_context (list, optional): List of document records from SQLite DB.
        api_key (str, optional): Groq API key.
        model (str, optional): Groq model name.

    Returns:
        dict: {"response": str, "error": str or None, "model": str}
    """
    load_dotenv(override=True)
    effective_api_key = api_key.strip() if (api_key and api_key.strip()) else os.getenv("GROQ_API_KEY")
    target_model = model.strip() if (model and model.strip()) else os.getenv("GROQ_MODEL", DEFAULT_MODEL)

    if not effective_api_key or effective_api_key.startswith("your_"):
        return {
            "response": "⚠️ GROQ_API_KEY is missing or invalid. Please configure your API key in the sidebar settings.",
            "error": "GROQ_API_KEY missing",
            "model": target_model
        }

    # Build context string
    context_str = build_context_string(document_context, history_context)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context_str=context_str)

    formatted_messages = [{"role": "system", "content": system_prompt}]
    
    # Append recent conversation history (up to last 10 messages)
    for msg in messages[-10:]:
        formatted_messages.append({"role": msg["role"], "content": msg["content"]})

    try:
        client = Groq(api_key=effective_api_key)
        
        max_retries = 3
        backoff = 2
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    messages=formatted_messages,
                    model=target_model,
                    temperature=0.3,
                    max_tokens=1000
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

        reply_content = response.choices[0].message.content.strip()
        return {
            "response": reply_content,
            "error": None,
            "model": target_model
        }

    except Exception as e:
        return {
            "response": f"⚠️ Chatbot Error: {str(e)}",
            "error": str(e),
            "model": target_model
        }
