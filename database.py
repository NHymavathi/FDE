"""
database.py

SQLite persistence module for Intelligent Document Extraction & Structuring Agent.
Manages database initialization, saving processed documents, and querying processing history.
"""
import os
import sqlite3
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

DB_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DB_DIR, "processed_docs.db")

def get_connection() -> sqlite3.Connection:
    """Returns a connection to the SQLite database."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Creates the SQLite database schema if it does not exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processed_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT UNIQUE NOT NULL,
            original_filename TEXT NOT NULL,
            document_type TEXT NOT NULL,
            extraction_method TEXT NOT NULL,
            structured_data TEXT NOT NULL,
            completeness_score REAL NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_document(
    document_id: str,
    original_filename: str,
    document_type: str,
    extraction_method: str,
    structured_data: Dict[str, Any],
    completeness_score: float,
    status: str
) -> bool:
    """
    Saves or updates a processed document record in SQLite.

    Args:
        document_id (str): Unique document identifier.
        original_filename (str): Original uploaded file name.
        document_type (str): Category (INVOICE, PURCHASE_ORDER, RESUME, UNKNOWN).
        extraction_method (str): Method used ("pymupdf" or "ocr").
        structured_data (dict): Extracted JSON object.
        completeness_score (float): Completeness score between 0.0 and 1.0.
        status (str): "READY_FOR_APPROVAL", "REQUIRES_REVIEW", or "APPROVED".

    Returns:
        bool: True if saved successfully.
    """
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    json_str = json.dumps(structured_data, ensure_ascii=False)
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO processed_documents (
            document_id, original_filename, document_type, 
            extraction_method, structured_data, completeness_score, status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(document_id) DO UPDATE SET
            structured_data = excluded.structured_data,
            completeness_score = excluded.completeness_score,
            status = excluded.status,
            created_at = excluded.created_at
    """, (
        document_id, original_filename, document_type,
        extraction_method, json_str, completeness_score, status, created_at
    ))

    conn.commit()
    conn.close()
    return True

def get_all_documents() -> List[Dict[str, Any]]:
    """Fetches all processed document records sorted by created_at descending."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, document_id, original_filename, document_type, 
               extraction_method, structured_data, completeness_score, status, created_at
        FROM processed_documents
        ORDER BY created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    result = []
    for row in rows:
        item = dict(row)
        try:
            item["structured_data"] = json.loads(item["structured_data"])
        except Exception:
            item["structured_data"] = {}
        result.append(item)
    return result

def get_document_by_id(document_id: str) -> Optional[Dict[str, Any]]:
    """Fetches a single document by document_id."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, document_id, original_filename, document_type, 
               extraction_method, structured_data, completeness_score, status, created_at
        FROM processed_documents
        WHERE document_id = ?
    """, (document_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        item = dict(row)
        item["structured_data"] = json.loads(item["structured_data"])
        return item
    return None
