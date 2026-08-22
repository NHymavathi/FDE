"""
app.py

Intelligent Document Extraction & Structuring Agent - Full MVP Application.
Integrates PyMuPDF, EasyOCR Fallback, Groq LLM Classification, Pydantic Schemas,
Field Completeness Scoring, Human Review Workflow, and SQLite Persistence.
"""
import os
import uuid
import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

import re
from modules.extractor import process_document, DEFAULT_MIN_CHAR_THRESHOLD
from modules.classifier import classify_document
from modules.structured_extractor import extract_structured_data
from modules.validator import evaluate_document_quality
from modules.chatbot import generate_chatbot_response
from database import init_db, save_document, get_all_documents

# Load environment variables dynamically
load_dotenv(override=True)

def safe_float(val) -> float:
    """Safely converts input value to float for UI number inputs."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        cleaned = re.sub(r'[^\d.-]', '', val)
        try:
            return float(cleaned) if cleaned else 0.0
        except ValueError:
            return 0.0
    return 0.0

# Page Configuration
st.set_page_config(
    page_title="Intelligent Document Extraction Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize SQLite Database
init_db()

# Custom CSS for Modern Glassmorphic Dark UI & Typography
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700;800&display=swap');

/* Global Fonts & Background Reset */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: -0.02em;
}

.stApp {
    background: radial-gradient(circle at 10% 10%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 90% 90%, rgba(168, 85, 247, 0.12) 0%, transparent 40%),
                linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0f172a 100%) !important;
    background-attachment: fixed !important;
    color: #f1f5f9;
}

/* Hero Header Banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 2.2rem 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(99, 102, 241, 0.18);
    border: 1px solid rgba(129, 140, 248, 0.35);
    color: #c7d2fe;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.3rem 0.85rem;
    border-radius: 9999px;
    margin-bottom: 0.8rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 45%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0.2rem 0 0.6rem 0;
    line-height: 1.2;
}

.hero-sub {
    color: #94a3b8;
    font-size: 1.02rem;
    line-height: 1.6;
    margin: 0;
}

/* Glassmorphism Card Container */
.glass-container {
    background: rgba(30, 41, 59, 0.55);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
}

/* Custom Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.75rem;
    background-color: rgba(15, 23, 42, 0.6) !important;
    padding: 0.5rem;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.stTabs [data-baseweb="tab"] {
    height: 48px;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0 1.25rem !important;
    border: none !important;
    transition: all 0.2s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #f1f5f9 !important;
    background: rgba(255, 255, 255, 0.05) !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

/* Custom Metric Cards */
[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    padding: 1.2rem 1.4rem !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    transition: all 0.2s ease !important;
}

[data-testid="stMetric"]:hover {
    border-color: rgba(129, 140, 248, 0.4) !important;
    transform: translateY(-2px);
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-weight: 700 !important;
    font-size: 1.5rem !important;
}

/* File Uploader Custom Styling */
[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.5) !important;
    border: 2px dashed rgba(99, 102, 241, 0.3) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    transition: border-color 0.3s ease, background 0.3s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: #818cf8 !important;
    background: rgba(99, 102, 241, 0.05) !important;
}

/* Glowing Status Badges */
.badge-approved {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(52, 211, 153, 0.4);
    color: #6ee7b7;
    padding: 0.6rem 1.2rem;
    border-radius: 12px;
    font-weight: 700;
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
}

.badge-review {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(251, 191, 36, 0.4);
    color: #fde68a;
    padding: 0.6rem 1.2rem;
    border-radius: 12px;
    font-weight: 700;
    box-shadow: 0 0 15px rgba(245, 158, 11, 0.2);
}

/* Buttons */
.stButton > button, div.stFormSubmitButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.5rem !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover, div.stFormSubmitButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(168, 85, 247, 0.5) !important;
    opacity: 0.95 !important;
}

/* Inputs */
.stTextInput input, .stTextArea textarea, .stSelectbox select {
    background-color: rgba(15, 23, 42, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 10px !important;
    color: #f8fafc !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #090d16 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #6366f1 0%, #10b981 100%) !important;
    border-radius: 9999px !important;
}
</style>
""", unsafe_allow_html=True)

# Executive Hero Banner
st.markdown("""
<div class="hero-banner">
    <div class="hero-pill">⚡ Enterprise MVP • PyMuPDF + EasyOCR + Groq LLM + Pydantic</div>
    <div class="hero-title">Intelligent Document Extraction & Structuring Agent</div>
    <div class="hero-sub">Automated PDF/Image Parsing → OCR Fallback Routing → LLM Semantic Classification → Pydantic Validation → Human Verification → SQLite Persistence</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Settings & Configuration")
load_dotenv(override=True)
env_api_key = os.getenv("GROQ_API_KEY", "")
groq_api_key = st.sidebar.text_input(
    "Groq API Key", 
    value=env_api_key if (env_api_key and not env_api_key.startswith("your_")) else "",
    type="password",
    help="Enter your Groq API Key or set GROQ_API_KEY in .env"
)

groq_model = st.sidebar.selectbox(
    "Groq Model",
    ["groq/compound-mini", "groq/compound", "qwen/qwen3.6-27b", "openai/gpt-oss-20b"],
    index=0
)

min_char_threshold = st.sidebar.slider(
    "OCR Fallback Min Character Threshold",
    min_value=10,
    max_value=200,
    value=DEFAULT_MIN_CHAR_THRESHOLD,
    help="If PyMuPDF extracts fewer characters than this threshold, EasyOCR will be triggered."
)

st.sidebar.divider()
st.sidebar.caption("🎯 Supported Schemas: Invoice, Purchase Order, Resume")

# Main Layout Navigation Tabs
tab_process, tab_history, tab_chatbot, tab_assumptions = st.tabs([
    "📄 Process & Review Document", 
    "📊 Processing History (SQLite)",
    "💬 AI Document Chatbot",
    "ℹ️ Architecture & Documentation"
])

# ==============================================================================
# TAB 1: PROCESS & REVIEW DOCUMENT
# ==============================================================================
with tab_process:
    st.subheader("1. Document Upload")
    uploaded_file = st.file_uploader(
        "Upload document (PDF, PNG, JPG, JPEG)", 
        type=["pdf", "png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        doc_id = f"doc_{uuid.uuid4().hex[:10]}"

        # --- STEP 1: Text Extraction & OCR Fallback ---
        with st.spinner("Extracting text layer / evaluating EasyOCR fallback..."):
            ext_res = process_document(file_bytes, uploaded_file.name, min_char_threshold=min_char_threshold)

        if not ext_res["success"]:
            st.error(f"Text extraction failed: {ext_res['error']}")
        else:
            # Display Extraction Metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Filename", uploaded_file.name)
            with c2:
                method_label = "⚡ PyMuPDF" if ext_res["extraction_method"] == "pymupdf" else "🔍 EasyOCR"
                st.metric("Extraction Method", method_label)
            with c3:
                st.metric("Character Count", f"{ext_res['character_count']:,}")
            with c4:
                st.metric("Pages / Frames", ext_res["page_count"])

            st.divider()

            # --- STEP 2: Document Classification ---
            st.subheader("2. Semantic Document Classification")
            with st.spinner("Classifying document using Groq LLM (temperature=0)..."):
                class_res = classify_document(ext_res["text"], api_key=groq_api_key, model=groq_model)

            category = class_res["category"]
            cat_icons = {
                "INVOICE": "🧾 INVOICE",
                "PURCHASE_ORDER": "📦 PURCHASE ORDER",
                "RESUME": "👤 RESUME",
                "UNKNOWN": "❓ UNKNOWN"
            }
            category_display = cat_icons.get(category, "❓ UNKNOWN")

            col_cat, col_info = st.columns([1, 2])
            with col_cat:
                if category in ["INVOICE", "RESUME"]:
                    st.success(f"### {category_display}")
                elif category == "PURCHASE_ORDER":
                    st.info(f"### {category_display}")
                else:
                    st.warning(f"### {category_display}")
            with col_info:
                st.write(f"- **Schema Selected**: `{category}`")
                st.write(f"- **LLM Engine**: `{class_res.get('model')}`")
                if class_res.get("error"):
                    st.error(f"⚠️ {class_res['error']}")

            st.divider()

            # --- STEP 3: Schema-based Structured Extraction & Validation ---
            st.subheader("3. Structured Extraction & Quality Scoring")
            with st.spinner("Extracting structured fields matching Pydantic schema..."):
                struct_res = extract_structured_data(ext_res["text"], category, api_key=groq_api_key, model=groq_model)

            extracted_fields = struct_res["data"]
            is_pydantic_valid = struct_res["is_valid"]

            # Save active document context for Chatbot tab
            st.session_state["active_doc_context"] = {
                "filename": uploaded_file.name,
                "category": category,
                "text": ext_res["text"],
                "extracted_data": extracted_fields
            }

            # Calculate Completeness Score & Review Status
            eval_res = evaluate_document_quality(extracted_fields, category, is_pydantic_valid)
            score_pct = eval_res["score_percentage"]
            status = eval_res["status"]

            # Quality Banner
            col_score, col_status = st.columns([1, 1])
            with col_score:
                st.write(f"**Field Completeness Score**: `{score_pct}%` ({eval_res['populated_fields']}/{eval_res['total_fields']} populated fields)")
                st.progress(eval_res["completeness_score"])
            with col_status:
                if status == "READY_FOR_APPROVAL":
                    st.markdown('<div class="badge-approved">🟢 Status: READY_FOR_APPROVAL (High Completeness & Valid Pydantic Schema)</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge-review">🔴 Status: REQUIRES_REVIEW (Low Completeness or Schema Errors)</div>', unsafe_allow_html=True)

            st.caption(f"ℹ️ *Completeness Metric*: {eval_res['explanation']}")
            if struct_res.get("validation_error"):
                st.error(f"Pydantic Validation Message: {struct_res['validation_error']}")

            st.divider()

            # --- STEP 4: Human Review & Interactive Edit Workflow ---
            st.subheader("4. Human Review & Verification Workflow")
            st.info("Inspect or adjust extracted fields below before approving and persisting to SQLite database.")

            # Create Interactive Editing Form
            with st.form("human_review_form"):
                edited_fields = {}
                
                if category == "INVOICE":
                    edited_fields["invoice_number"] = st.text_input("Invoice Number", value=extracted_fields.get("invoice_number") or "")
                    edited_fields["vendor_name"] = st.text_input("Vendor Name", value=extracted_fields.get("vendor_name") or "")
                    edited_fields["invoice_date"] = st.text_input("Invoice Date", value=extracted_fields.get("invoice_date") or "")
                    
                    raw_total = extracted_fields.get("total_amount")
                    edited_fields["total_amount"] = st.number_input("Total Amount ($)", value=safe_float(raw_total), format="%.2f")
                    
                    raw_tax = extracted_fields.get("tax_amount")
                    edited_fields["tax_amount"] = st.number_input("Tax Amount ($)", value=safe_float(raw_tax), format="%.2f")

                elif category == "PURCHASE_ORDER":
                    edited_fields["po_number"] = st.text_input("PO Number", value=extracted_fields.get("po_number") or "")
                    edited_fields["buyer_name"] = st.text_input("Buyer Name", value=extracted_fields.get("buyer_name") or "")
                    edited_fields["supplier_name"] = st.text_input("Supplier Name", value=extracted_fields.get("supplier_name") or "")
                    
                    raw_total = extracted_fields.get("total_amount")
                    edited_fields["total_amount"] = st.number_input("Total Amount ($)", value=safe_float(raw_total), format="%.2f")

                elif category == "RESUME":
                    edited_fields["candidate_name"] = st.text_input("Candidate Name", value=extracted_fields.get("candidate_name") or "")
                    edited_fields["email"] = st.text_input("Email", value=extracted_fields.get("email") or "")
                    
                    skills_raw = extracted_fields.get("skills", [])
                    skills_str = ", ".join(skills_raw) if isinstance(skills_raw, list) else str(skills_raw)
                    skills_input = st.text_input("Skills (comma separated)", value=skills_str)
                    edited_fields["skills"] = [s.strip() for s in skills_input.split(",") if s.strip()]
                    
                    edited_fields["education"] = st.text_area("Education", value=extracted_fields.get("education") or "")
                
                else:  # UNKNOWN
                    edited_fields["summary"] = st.text_area("Document Summary", value=extracted_fields.get("summary") or "")

                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    submit_approve = st.form_submit_button("🟢 Approve & Save Result", width="stretch")
                with col_btn2:
                    submit_draft = st.form_submit_button("💾 Save Changes / Draft", width="stretch")

            if submit_approve or submit_draft:
                # Recalculate completeness score on updated fields
                updated_eval = evaluate_document_quality(edited_fields, category, is_pydantic_valid=True)
                final_status = "APPROVED" if submit_approve else updated_eval["status"]
                
                saved = save_document(
                    document_id=doc_id,
                    original_filename=uploaded_file.name,
                    document_type=category,
                    extraction_method=ext_res["extraction_method"],
                    structured_data=edited_fields,
                    completeness_score=updated_eval["completeness_score"],
                    status=final_status
                )
                
                if saved:
                    st.balloons()
                    st.success(f"✅ Document successfully saved to SQLite! Document ID: `{doc_id}` | Status: `{final_status}`")
                else:
                    st.error("Failed to save document to SQLite database.")

            with st.expander("🔍 View Raw Extracted Text Layer"):
                st.text_area("Raw Text", value=ext_res["text"], height=200, disabled=True)

    else:
        st.info("Upload a PDF or Image document in Step 1 to run the full extraction, classification, and review workflow.")


# ==============================================================================
# TAB 2: PROCESSING HISTORY (SQLITE)
# ==============================================================================
with tab_history:
    st.subheader("📊 SQLite Document Processing History")
    
    docs = get_all_documents()
    if not docs:
        st.info("No processed documents stored in SQLite yet. Process and save a document in Tab 1 to see records here.")
    else:
        # Filtering Options
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            status_filter = st.selectbox("Filter by Status", ["ALL", "APPROVED", "READY_FOR_APPROVAL", "REQUIRES_REVIEW"])
        with f_col2:
            type_filter = st.selectbox("Filter by Document Type", ["ALL", "INVOICE", "PURCHASE_ORDER", "RESUME", "UNKNOWN"])

        # Apply Filters
        filtered_docs = docs
        if status_filter != "ALL":
            filtered_docs = [d for d in filtered_docs if d["status"] == status_filter]
        if type_filter != "ALL":
            filtered_docs = [d for d in filtered_docs if d["document_type"] == type_filter]

        st.caption(f"Showing {len(filtered_docs)} document records stored in SQLite.")

        # Render Table Summary
        table_data = []
        for d in filtered_docs:
            table_data.append({
                "Document ID": d["document_id"],
                "Original Filename": d["original_filename"],
                "Category": d["document_type"],
                "Extraction Method": d["extraction_method"].upper(),
                "Completeness Score": f"{round(d['completeness_score'] * 100, 1)}%",
                "Status": d["status"],
                "Processed Date": d["created_at"]
            })

        df = pd.DataFrame(table_data)
        st.dataframe(df, width="stretch")

        st.divider()
        st.subheader("🔍 Inspect Saved Structured Data")
        selected_doc_id = st.selectbox("Select Document ID to View JSON Payload", [d["document_id"] for d in filtered_docs])
        
        selected_record = next((d for d in filtered_docs if d["document_id"] == selected_doc_id), None)
        if selected_record:
            st.json(selected_record["structured_data"])


# ==============================================================================
# TAB 3: AI DOCUMENT CHATBOT
# ==============================================================================
with tab_chatbot:
    st.subheader("💬 Interactive AI Document Assistant")
    st.caption("Ask questions about the currently uploaded document or search through past processed records stored in SQLite.")

    # Initialize chat session history
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "content": "👋 Hello! I am your AI Document Assistant. Upload a document or ask me any questions about your documents, extracted fields, or processing history!"
            }
        ]

    # Context Controls & Status Summary
    active_doc = st.session_state.get("active_doc_context")
    all_db_docs = get_all_documents()

    ctx_col1, ctx_col2, ctx_col3 = st.columns([2, 2, 1])
    with ctx_col1:
        use_active_doc = st.checkbox("📄 Include Active Upload Context", value=True if active_doc else False)
        if active_doc:
            st.caption(f"Active file: `{active_doc['filename']}` ({active_doc['category']})")
        else:
            st.caption("No document uploaded in Tab 1 yet.")
            
    with ctx_col2:
        use_db_history = st.checkbox("📊 Include SQLite DB History Context", value=True if all_db_docs else False)
        st.caption(f"Database records available: `{len(all_db_docs)}` documents")
        
    with ctx_col3:
        if st.button("🗑️ Clear Chat", help="Reset chatbot conversation history"):
            st.session_state["chat_messages"] = [
                {
                    "role": "assistant",
                    "content": "Chat history cleared. How can I help you now?"
                }
            ]
            st.rerun()

    st.divider()

    # Render Chat Conversation History
    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input Form
    user_query = st.chat_input("Type your question here (e.g., 'What is the total amount due on the invoice?')...")
    if user_query:
        # Append User Message
        st.session_state["chat_messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking & analyzing document context..."):
                doc_ctx = active_doc if use_active_doc else None
                db_ctx = all_db_docs if use_db_history else None

                bot_res = generate_chatbot_response(
                    messages=st.session_state["chat_messages"],
                    document_context=doc_ctx,
                    history_context=db_ctx,
                    api_key=groq_api_key,
                    model=groq_model
                )
                
                reply_text = bot_res["response"]
                st.markdown(reply_text)
                
                # Append Assistant Message
                st.session_state["chat_messages"].append({"role": "assistant", "content": reply_text})


# ==============================================================================
# TAB 4: ASSUMPTIONS & ARCHITECTURE
# ==============================================================================
with tab_assumptions:
    st.subheader("ℹ️ Project Assumptions & System Architecture")
    
    st.markdown("""
    ### System Architecture & Design Assumptions
    1. **MVP Scope**: Supports **Invoice**, **Purchase Order**, and **Resume** document schemas out-of-the-box.
    2. **Multi-Format Extraction**: Supports digital text PDFs, scanned image PDFs, and images (`.png`, `.jpg`, `.jpeg`).
    3. **Extraction Routing**: Attempts direct **PyMuPDF** text layer extraction first. Triggers **EasyOCR** fallback when character count falls below threshold ($< 50$ characters).
    4. **Semantic Classification**: Uses **Groq LLM** at `temperature=0` for deterministic category classification.
    5. **Pydantic Validation**: Validates extracted fields against structured schemas (`InvoiceSchema`, `PurchaseOrderSchema`, `ResumeSchema`).
    6. **Completeness Heuristic**: Calculates field completeness score ($ populated\_fields / total\_expected\_fields $). Automatically routes documents below $70\%$ completeness or with schema errors to Human Review.
    7. **Human Review & SQLite**: Provides an interactive Streamlit form to review, modify, approve, and save documents into an SQLite persistent store.
    """)

