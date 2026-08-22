"""
modules/validator.py

Calculates field completeness heuristic score and determines document approval status.
Routes low completeness (<0.7) or invalid documents to Human Review.
"""
from typing import Dict, Any
from schemas import SCHEMA_MAP, UnknownSchema

COMPLETENESS_THRESHOLD = 0.7

def calculate_completeness(extracted_data: Dict[str, Any], category: str) -> Dict[str, Any]:
    """
    Calculates the field completeness score for extracted document data.

    Completeness Score Formula:
    completeness_score = (number of populated fields) / (total expected fields for schema)

    Populated field criteria:
    - Non-null (not None)
    - Non-empty string (str.strip() != "")
    - Non-empty list (len(list) > 0)
    - Valid numbers/floats

    Args:
        extracted_data (dict): Extracted field dictionary.
        category (str): Document schema category.

    Returns:
        dict: A dictionary containing:
            - completeness_score (float): Score between 0.0 and 1.0.
            - score_percentage (float): Score as a percentage (0 to 100%).
            - populated_fields (int): Number of fields successfully extracted.
            - total_fields (int): Total expected fields for category schema.
            - status (str): "READY_FOR_APPROVAL" or "REQUIRES_REVIEW".
            - explanation (str): Explicit note that this is a heuristic score.
    """
    schema_cls = SCHEMA_MAP.get(category, UnknownSchema)
    total_fields = schema_cls.get_field_count()

    populated_count = 0
    field_details = {}

    for field_name, value in extracted_data.items():
        is_populated = False
        if value is not None:
            if isinstance(value, str) and value.strip() != "":
                is_populated = True
            elif isinstance(value, list) and len(value) > 0:
                is_populated = True
            elif isinstance(value, (int, float)):
                is_populated = True
            elif isinstance(value, bool):
                is_populated = True

        field_details[field_name] = is_populated
        if is_populated:
            populated_count += 1

    score = populated_count / total_fields if total_fields > 0 else 0.0
    score_percentage = round(score * 100, 1)

    # Status Determination Rule
    if score >= COMPLETENESS_THRESHOLD:
        status = "READY_FOR_APPROVAL"
    else:
        status = "REQUIRES_REVIEW"

    explanation = (
        f"Completeness score ({score_percentage}%) measures the ratio of populated fields "
        f"({populated_count}/{total_fields}). It is a field presence heuristic, not LLM confidence."
    )

    return {
        "completeness_score": score,
        "score_percentage": score_percentage,
        "populated_fields": populated_count,
        "total_fields": total_fields,
        "field_details": field_details,
        "status": status,
        "explanation": explanation
    }

def evaluate_document_quality(
    extracted_data: Dict[str, Any], 
    category: str, 
    is_pydantic_valid: bool
) -> Dict[str, Any]:
    """
    Combines Pydantic schema validation status with field completeness scoring.

    If Pydantic validation failed, status is automatically forced to REQUIRES_REVIEW.
    """
    comp_res = calculate_completeness(extracted_data, category)

    if not is_pydantic_valid:
        comp_res["status"] = "REQUIRES_REVIEW"
        comp_res["explanation"] += " (Forced to REQUIRES_REVIEW due to Pydantic schema validation errors)."

    return comp_res
