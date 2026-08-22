"""
schemas.py

Pydantic data validation schemas for document extraction MVP.
Defines explicit field structures for Invoice, Purchase Order, and Resume documents.
"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class InvoiceSchema(BaseModel):
    """Schema for INVOICE document extraction."""
    invoice_number: Optional[str] = Field(None, description="Unique invoice identification number")
    vendor_name: Optional[str] = Field(None, description="Name of the selling vendor/company")
    invoice_date: Optional[str] = Field(None, description="Date the invoice was issued (YYYY-MM-DD or raw string)")
    total_amount: Optional[float] = Field(None, description="Total amount due including tax")
    tax_amount: Optional[float] = Field(None, description="Total tax amount charged")

    @classmethod
    def get_field_count(cls) -> int:
        return 5


class PurchaseOrderSchema(BaseModel):
    """Schema for PURCHASE_ORDER document extraction."""
    po_number: Optional[str] = Field(None, description="Purchase order reference number")
    buyer_name: Optional[str] = Field(None, description="Name of the purchasing buyer/company")
    supplier_name: Optional[str] = Field(None, description="Name of the supplier receiving the PO")
    total_amount: Optional[float] = Field(None, description="Total monetary amount of the purchase order")

    @classmethod
    def get_field_count(cls) -> int:
        return 4


class ResumeSchema(BaseModel):
    """Schema for RESUME document extraction."""
    candidate_name: Optional[str] = Field(None, description="Full name of the job candidate")
    email: Optional[str] = Field(None, description="Primary email address of the candidate")
    skills: List[str] = Field(default_factory=list, description="List of technical/professional skills")
    education: Optional[str] = Field(None, description="Educational background or highest degree achieved")

    @classmethod
    def get_field_count(cls) -> int:
        return 4


class UnknownSchema(BaseModel):
    """Fallback schema for UNKNOWN document category."""
    summary: Optional[str] = Field(None, description="Brief summary of unclassified document content")

    @classmethod
    def get_field_count(cls) -> int:
        return 1


# Type alias for schema models
DocumentSchemaType = Union[InvoiceSchema, PurchaseOrderSchema, ResumeSchema, UnknownSchema]

SCHEMA_MAP = {
    "INVOICE": InvoiceSchema,
    "PURCHASE_ORDER": PurchaseOrderSchema,
    "RESUME": ResumeSchema,
    "UNKNOWN": UnknownSchema
}
