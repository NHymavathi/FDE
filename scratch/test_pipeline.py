import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database import init_db, save_document, get_all_documents
from modules.extractor import process_document
from modules.classifier import classify_document
from modules.structured_extractor import extract_structured_data
from modules.validator import evaluate_document_quality

init_db()

files = [
    'sample_docs/invoice.pdf', 
    'sample_docs/purchase_order.pdf', 
    'sample_docs/resume.pdf', 
    'sample_docs/scanned_invoice.pdf'
]

for filepath in files:
    if not os.path.exists(filepath):
        print(f"Missing {filepath}")
        continue
    
    with open(filepath, 'rb') as f:
        content = f.read()
    
    filename = os.path.basename(filepath)
    print(f"\n==========================================")
    print(f"Processing: {filename}")
    print(f"==========================================")
    
    # 1. Extract text
    doc_res = process_document(content, filename)
    print(f"Extraction Method: {doc_res['extraction_method']} | Chars: {doc_res['character_count']}")
    
    # 2. Classify
    cls_res = classify_document(doc_res['text'])
    category = cls_res['category']
    print(f"Document Category: {category}")
    
    # 3. Structured Extract
    ext_res = extract_structured_data(doc_res['text'], category)
    print(f"Schema Valid: {ext_res['is_valid']}")
    print(f"Structured Data: {ext_res['data']}")
    
    val_res = evaluate_document_quality(ext_res['data'], category, ext_res['is_valid'])
    print(f"Completeness Score: {val_res['score_percentage']}% | Status: {val_res['status']}")
    
    # 5. Save to SQLite DB
    doc_id = filename.replace('.', '_')
    save_document(
        document_id=doc_id,
        original_filename=filename,
        document_type=category,
        extraction_method=doc_res['extraction_method'],
        structured_data=ext_res['data'],
        completeness_score=val_res['completeness_score'],
        status=val_res['status']
    )

print("\n==========================================")
print("SQLite Database Saved Documents")
print("==========================================")
docs = get_all_documents()
print(f"Total documents in database: {len(docs)}")
for d in docs:
    print(f"- {d['original_filename']}: Category={d['document_type']}, Method={d['extraction_method']}, Score={d['completeness_score']*100:.1f}%, Status={d['status']}")
