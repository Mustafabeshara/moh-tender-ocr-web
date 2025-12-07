"""
Pydantic models for API request/response schemas
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ExtractionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TenderItem(BaseModel):
    """Single item from a tender"""
    item_number: str = ""
    description: str = ""
    quantity: str = ""
    unit: str = ""
    specifications: str = ""
    language: str = "unknown"
    has_arabic: bool = False


class TenderData(BaseModel):
    """Complete tender extraction result"""
    id: Optional[str] = None
    reference_number: str = ""
    title: str = ""
    closing_date: str = ""
    posting_date: str = ""
    department: str = ""
    items: List[TenderItem] = []
    specifications_text: str = ""
    items_count: int = 0
    ocr_confidence: float = 0.0
    source_files: List[str] = []
    extraction_timestamp: str = ""
    extraction_method: str = "ocr"
    language: str = "unknown"
    has_arabic_content: bool = False
    errors: List[str] = []

    class Config:
        json_schema_extra = {
            "example": {
                "reference_number": "5TN2024",
                "closing_date": "15/03/2024",
                "items_count": 12,
                "ocr_confidence": 87.5
            }
        }


class ExtractionJob(BaseModel):
    """Extraction job status"""
    job_id: str
    status: ExtractionStatus
    progress: int = 0
    total_files: int = 0
    processed_files: int = 0
    results: List[TenderData] = []
    errors: List[str] = []
    created_at: datetime
    completed_at: Optional[datetime] = None


class FileUploadResponse(BaseModel):
    """Response after file upload"""
    job_id: str
    message: str
    files_received: int
    status: ExtractionStatus


class ExtractionRequest(BaseModel):
    """Request to start extraction"""
    languages: List[str] = ["eng", "ara"]
    dpi: int = Field(default=300, ge=100, le=600)
    max_pages: int = Field(default=10, ge=1, le=50)
    department: str = "Biomedical Engineering"


class ExtractionSummary(BaseModel):
    """Summary of extraction results"""
    total_tenders: int
    total_items: int
    arabic_tenders: int
    tenders_with_specs: int
    average_confidence: float
    extraction_date: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    tesseract_available: bool
    poppler_available: bool
    version: str
