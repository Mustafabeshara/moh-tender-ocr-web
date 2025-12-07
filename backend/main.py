"""
MOH Tender OCR Web Application - FastAPI Backend
"""
import os
import json
import shutil
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict, field

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Import OCR components from local ocr_engine module (improved copy)
try:
    from ocr_engine import (
        OCREngine, TenderExtractor, TenderData, TenderItem,
        BatchProcessor, CONFIG
    )
    from dataclasses import asdict
    OCR_AVAILABLE = True
    print("✅ OCR components loaded from local ocr_engine module")
except ImportError as e:
    print(f"Warning: Could not import OCR components: {e}")
    OCR_AVAILABLE = False
    TenderExtractor = None

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports"

for dir_path in [UPLOAD_DIR, DATA_DIR, EXPORT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Database file for tender data
TENDERS_DB = DATA_DIR / "tenders.json"

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class TenderItemModel(BaseModel):
    item_number: str = ""
    description: str = ""
    quantity: str = ""
    unit: str = ""
    language: str = "unknown"
    has_arabic: bool = False

class TenderModel(BaseModel):
    id: str
    reference: str
    department: str = ""
    title: str = ""
    closing_date: str = ""
    items: List[TenderItemModel] = []
    ocr_confidence: float = 0.0
    has_arabic_content: bool = False
    specifications: str = ""
    source_file: str = ""
    status: str = "pending"  # pending, processing, completed, failed
    created_at: str = ""
    updated_at: str = ""

class TenderUpdateModel(BaseModel):
    reference: Optional[str] = None
    department: Optional[str] = None
    title: Optional[str] = None
    closing_date: Optional[str] = None
    items: Optional[List[TenderItemModel]] = None
    specifications: Optional[str] = None
    status: Optional[str] = None

class OCRProgressModel(BaseModel):
    tender_id: str
    status: str
    progress: float
    message: str

class StatsModel(BaseModel):
    total_tenders: int
    pending: int
    processing: int
    completed: int
    failed: int
    total_items: int
    avg_confidence: float

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def load_tenders() -> Dict[str, dict]:
    """Load tenders from JSON database"""
    if TENDERS_DB.exists():
        try:
            with open(TENDERS_DB, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading tenders: {e}")
    return {}

def save_tenders(tenders: Dict[str, dict]):
    """Save tenders to JSON database"""
    try:
        with open(TENDERS_DB, 'w', encoding='utf-8') as f:
            json.dump(tenders, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving tenders: {e}")

def generate_tender_id() -> str:
    """Generate unique tender ID"""
    return f"TND-{datetime.now().strftime('%Y%m%d%H%M%S')}-{os.urandom(4).hex()}"

# ============================================================================
# OCR PROCESSING
# ============================================================================

# Global OCR extractor (uses TenderExtractor from moh_ocr_extractor.py)
tender_extractor = None
processing_status: Dict[str, OCRProgressModel] = {}

def get_tender_extractor():
    """Get or create TenderExtractor instance"""
    global tender_extractor
    if tender_extractor is None and OCR_AVAILABLE and TenderExtractor is not None:
        tender_extractor = TenderExtractor()
    return tender_extractor

async def process_pdf_ocr(tender_id: str, pdf_path: str):
    """Process PDF with OCR in background using TenderExtractor from moh_ocr_extractor.py"""
    global processing_status

    tenders = load_tenders()
    if tender_id not in tenders:
        return

    try:
        processing_status[tender_id] = OCRProgressModel(
            tender_id=tender_id,
            status="processing",
            progress=0.0,
            message="Starting OCR..."
        )

        tenders[tender_id]["status"] = "processing"
        save_tenders(tenders)

        extractor = get_tender_extractor()
        if not extractor:
            raise Exception("OCR extractor not available - check pytesseract/poppler installation")

        processing_status[tender_id].message = "Converting PDF to images..."
        processing_status[tender_id].progress = 10.0

        # Run OCR using TenderExtractor.process_pdf() in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        tender_data = await loop.run_in_executor(
            None,
            lambda: extractor.process_pdf(pdf_path)
        )

        processing_status[tender_id].message = "Extracting tender data..."
        processing_status[tender_id].progress = 70.0

        # Check for Arabic content in raw text
        has_arabic = bool(tender_data.raw_text and any('\u0600' <= c <= '\u06FF' for c in tender_data.raw_text))

        processing_status[tender_id].message = "Saving results..."
        processing_status[tender_id].progress = 90.0

        # Convert items from TenderItem dataclass to dict
        items_list = []
        for item in tender_data.items:
            items_list.append({
                "item_number": item.item_number,
                "description": item.description,
                "quantity": item.quantity,
                "unit": item.unit,
                "language": "unknown",
                "has_arabic": bool(any('\u0600' <= c <= '\u06FF' for c in item.description))
            })

        # Update tender data
        tenders[tender_id].update({
            "reference": tender_data.reference or tenders[tender_id].get("reference", ""),
            "closing_date": tender_data.closing_date,
            "items": items_list,
            "ocr_confidence": tender_data.ocr_confidence,
            "has_arabic_content": has_arabic,
            "specifications": tender_data.specifications[:2000] if tender_data.specifications else "",
            "status": "completed",
            "updated_at": datetime.now().isoformat()
        })

        save_tenders(tenders)

        processing_status[tender_id] = OCRProgressModel(
            tender_id=tender_id,
            status="completed",
            progress=100.0,
            message=f"Completed! Extracted {len(items_list)} items with {tender_data.ocr_confidence:.1f}% confidence"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        tenders = load_tenders()
        if tender_id in tenders:
            tenders[tender_id]["status"] = "failed"
            tenders[tender_id]["updated_at"] = datetime.now().isoformat()
            save_tenders(tenders)

        processing_status[tender_id] = OCRProgressModel(
            tender_id=tender_id,
            status="failed",
            progress=0.0,
            message=f"Error: {str(e)}"
        )

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="MOH Tender OCR API",
    description="API for processing and managing MOH tender documents with OCR",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded files
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "MOH Tender OCR API",
        "version": "1.0.0",
        "ocr_available": OCR_AVAILABLE,
        "endpoints": {
            "tenders": "/api/tenders",
            "upload": "/api/upload",
            "stats": "/api/stats",
            "export": "/api/export"
        }
    }

@app.get("/api/stats", response_model=StatsModel)
async def get_stats():
    """Get tender statistics"""
    tenders = load_tenders()

    total = len(tenders)
    pending = sum(1 for t in tenders.values() if t.get("status") == "pending")
    processing = sum(1 for t in tenders.values() if t.get("status") == "processing")
    completed = sum(1 for t in tenders.values() if t.get("status") == "completed")
    failed = sum(1 for t in tenders.values() if t.get("status") == "failed")

    total_items = sum(len(t.get("items", [])) for t in tenders.values())

    confidences = [t.get("ocr_confidence", 0) for t in tenders.values() if t.get("ocr_confidence", 0) > 0]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    return StatsModel(
        total_tenders=total,
        pending=pending,
        processing=processing,
        completed=completed,
        failed=failed,
        total_items=total_items,
        avg_confidence=avg_confidence
    )

@app.get("/api/tenders")
async def list_tenders(
    status: Optional[str] = None,
    department: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(default=50, le=500),
    offset: int = 0
):
    """List all tenders with optional filtering"""
    tenders = load_tenders()

    results = list(tenders.values())

    # Filter by status
    if status:
        results = [t for t in results if t.get("status") == status]

    # Filter by department
    if department:
        results = [t for t in results if department.lower() in t.get("department", "").lower()]

    # Search in reference and title
    if search:
        search_lower = search.lower()
        results = [t for t in results if
                   search_lower in t.get("reference", "").lower() or
                   search_lower in t.get("title", "").lower()]

    # Sort by created_at descending
    results.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    # Pagination
    total = len(results)
    results = results[offset:offset + limit]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "tenders": results
    }

@app.get("/api/tenders/{tender_id}")
async def get_tender(tender_id: str):
    """Get a specific tender by ID"""
    tenders = load_tenders()

    if tender_id not in tenders:
        raise HTTPException(status_code=404, detail="Tender not found")

    return tenders[tender_id]

@app.post("/api/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    department: str = "",
    background_tasks: BackgroundTasks = None
):
    """Upload a PDF and optionally start OCR processing"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    tender_id = generate_tender_id()

    # Save file
    safe_filename = f"{tender_id}_{file.filename.replace(' ', '_')}"
    file_path = UPLOAD_DIR / safe_filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Create tender record
    tender = {
        "id": tender_id,
        "reference": Path(file.filename).stem,
        "department": department,
        "title": "",
        "closing_date": "",
        "items": [],
        "ocr_confidence": 0.0,
        "has_arabic_content": False,
        "specifications": "",
        "source_file": safe_filename,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }

    tenders = load_tenders()
    tenders[tender_id] = tender
    save_tenders(tenders)

    return {
        "message": "File uploaded successfully",
        "tender_id": tender_id,
        "filename": safe_filename
    }

@app.post("/api/tenders/{tender_id}/process")
async def process_tender(tender_id: str, background_tasks: BackgroundTasks):
    """Start OCR processing for a tender"""
    tenders = load_tenders()

    if tender_id not in tenders:
        raise HTTPException(status_code=404, detail="Tender not found")

    tender = tenders[tender_id]

    if tender["status"] == "processing":
        raise HTTPException(status_code=400, detail="Tender is already being processed")

    pdf_path = UPLOAD_DIR / tender["source_file"]
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Source PDF not found")

    # Start background processing
    background_tasks.add_task(process_pdf_ocr, tender_id, str(pdf_path))

    return {"message": "OCR processing started", "tender_id": tender_id}

@app.get("/api/tenders/{tender_id}/status")
async def get_processing_status(tender_id: str):
    """Get OCR processing status"""
    if tender_id in processing_status:
        return processing_status[tender_id]

    tenders = load_tenders()
    if tender_id in tenders:
        return OCRProgressModel(
            tender_id=tender_id,
            status=tenders[tender_id].get("status", "unknown"),
            progress=100.0 if tenders[tender_id].get("status") == "completed" else 0.0,
            message=""
        )

    raise HTTPException(status_code=404, detail="Tender not found")

@app.put("/api/tenders/{tender_id}")
async def update_tender(tender_id: str, update: TenderUpdateModel):
    """Update tender data"""
    tenders = load_tenders()

    if tender_id not in tenders:
        raise HTTPException(status_code=404, detail="Tender not found")

    update_data = update.dict(exclude_unset=True)

    # Convert items if provided
    if "items" in update_data and update_data["items"]:
        update_data["items"] = [item.dict() if hasattr(item, 'dict') else item
                                 for item in update_data["items"]]

    tenders[tender_id].update(update_data)
    tenders[tender_id]["updated_at"] = datetime.now().isoformat()

    save_tenders(tenders)

    return tenders[tender_id]

@app.delete("/api/tenders/{tender_id}")
async def delete_tender(tender_id: str):
    """Delete a tender"""
    tenders = load_tenders()

    if tender_id not in tenders:
        raise HTTPException(status_code=404, detail="Tender not found")

    # Delete associated file
    source_file = tenders[tender_id].get("source_file")
    if source_file:
        file_path = UPLOAD_DIR / source_file
        if file_path.exists():
            file_path.unlink()

    del tenders[tender_id]
    save_tenders(tenders)

    return {"message": "Tender deleted successfully"}

@app.post("/api/tenders/{tender_id}/items")
async def add_item(tender_id: str, item: TenderItemModel):
    """Add an item to a tender"""
    tenders = load_tenders()

    if tender_id not in tenders:
        raise HTTPException(status_code=404, detail="Tender not found")

    if "items" not in tenders[tender_id]:
        tenders[tender_id]["items"] = []

    tenders[tender_id]["items"].append(item.dict())
    tenders[tender_id]["updated_at"] = datetime.now().isoformat()

    save_tenders(tenders)

    return tenders[tender_id]

@app.delete("/api/tenders/{tender_id}/items/{item_index}")
async def delete_item(tender_id: str, item_index: int):
    """Delete an item from a tender"""
    tenders = load_tenders()

    if tender_id not in tenders:
        raise HTTPException(status_code=404, detail="Tender not found")

    items = tenders[tender_id].get("items", [])
    if item_index < 0 or item_index >= len(items):
        raise HTTPException(status_code=404, detail="Item not found")

    items.pop(item_index)
    tenders[tender_id]["items"] = items
    tenders[tender_id]["updated_at"] = datetime.now().isoformat()

    save_tenders(tenders)

    return tenders[tender_id]

@app.get("/api/export/{format}")
async def export_tenders(
    format: str,
    status: Optional[str] = None,
    tender_ids: Optional[str] = None
):
    """Export tenders to Excel or JSON"""
    if format not in ["excel", "json"]:
        raise HTTPException(status_code=400, detail="Format must be 'excel' or 'json'")

    tenders = load_tenders()

    # Filter tenders
    if tender_ids:
        ids = tender_ids.split(",")
        export_data = [tenders[tid] for tid in ids if tid in tenders]
    elif status:
        export_data = [t for t in tenders.values() if t.get("status") == status]
    else:
        export_data = list(tenders.values())

    if not export_data:
        raise HTTPException(status_code=404, detail="No tenders to export")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if format == "json":
        export_path = EXPORT_DIR / f"tenders_export_{timestamp}.json"
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        return FileResponse(
            path=str(export_path),
            filename=f"tenders_export_{timestamp}.json",
            media_type="application/json"
        )

    else:  # Excel
        export_path = EXPORT_DIR / f"tenders_export_{timestamp}.xlsx"

        # Convert to TenderData objects for export function
        if OCR_AVAILABLE:
            tender_objects = []
            for t in export_data:
                td = TenderData(
                    reference=t.get("reference", ""),
                    department=t.get("department", ""),
                    title=t.get("title", ""),
                    closing_date=t.get("closing_date", ""),
                    ocr_confidence=t.get("ocr_confidence", 0),
                    has_arabic_content=t.get("has_arabic_content", False),
                    specifications=t.get("specifications", ""),
                    source_files=[t.get("source_file", "")]
                )
                for item_data in t.get("items", []):
                    td.items.append(TenderItem(
                        item_number=item_data.get("item_number", ""),
                        description=item_data.get("description", ""),
                        quantity=item_data.get("quantity", ""),
                        unit=item_data.get("unit", ""),
                        language=item_data.get("language", ""),
                        has_arabic=item_data.get("has_arabic", False)
                    ))
                tender_objects.append(td)

            export_to_excel(tender_objects, str(export_path))
        else:
            # Simple Excel export without the full function
            import pandas as pd
            rows = []
            for t in export_data:
                for item in t.get("items", [{"description": "No items"}]):
                    rows.append({
                        "Reference": t.get("reference", ""),
                        "Department": t.get("department", ""),
                        "Title": t.get("title", ""),
                        "Closing Date": t.get("closing_date", ""),
                        "Item #": item.get("item_number", ""),
                        "Description": item.get("description", ""),
                        "Quantity": item.get("quantity", ""),
                        "Unit": item.get("unit", ""),
                        "OCR Confidence": f"{t.get('ocr_confidence', 0):.1f}%"
                    })
            df = pd.DataFrame(rows)
            df.to_excel(str(export_path), index=False)

        return FileResponse(
            path=str(export_path),
            filename=f"tenders_export_{timestamp}.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

@app.get("/api/pdf/{tender_id}")
async def get_pdf(tender_id: str):
    """Get the PDF file for a tender"""
    tenders = load_tenders()

    if tender_id not in tenders:
        raise HTTPException(status_code=404, detail="Tender not found")

    source_file = tenders[tender_id].get("source_file")
    if not source_file:
        raise HTTPException(status_code=404, detail="No PDF file associated")

    file_path = UPLOAD_DIR / source_file
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found")

    return FileResponse(
        path=str(file_path),
        filename=source_file,
        media_type="application/pdf"
    )

@app.post("/api/batch-process")
async def batch_process(background_tasks: BackgroundTasks, status: str = "pending"):
    """Start batch OCR processing for all pending tenders"""
    tenders = load_tenders()

    pending = [tid for tid, t in tenders.items() if t.get("status") == status]

    if not pending:
        return {"message": "No tenders to process", "count": 0}

    for tender_id in pending:
        pdf_path = UPLOAD_DIR / tenders[tender_id]["source_file"]
        if pdf_path.exists():
            background_tasks.add_task(process_pdf_ocr, tender_id, str(pdf_path))

    return {"message": f"Started processing {len(pending)} tenders", "count": len(pending)}

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Starting MOH Tender OCR API Server...")
    print(f"Upload directory: {UPLOAD_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"OCR Available: {OCR_AVAILABLE}")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
