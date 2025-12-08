"""
MOH Tender OCR Web Application - FastAPI Backend
With integrated scraper support
"""
import os
import sys
import json
import shutil
import asyncio
import subprocess
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

# ============================================================================
# SCRAPER CONFIGURATION
# ============================================================================
SCRAPER_SCRIPT = os.path.expanduser("~/Documents/moh_tender_scraper_with_ocr.py")
SCRAPER_PYTHON = os.path.expanduser("~/Documents/MOH Tenders/scripts/moh_env/bin/python3")
MOH_TENDERS_DIR = os.path.expanduser("~/Documents/MOH Tenders")
OCR_RESULTS_DIR = os.path.expanduser("~/Documents/MOH Tenders/ocr_results")

# Scraper status tracking
scraper_status = {
    "running": False,
    "progress": 0.0,
    "message": "Idle",
    "pid": None,
    "last_run": None,
    "tenders_found": 0
}

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
    """Export tenders to Excel - appends to master file"""
    if format != "excel":
        raise HTTPException(status_code=400, detail="Only Excel export is supported")

    import pandas as pd

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

    # Master Excel file - one file for all exports
    export_path = EXPORT_DIR / "tenders_master.xlsx"

    # Build new rows from export data
    new_rows = []
    for t in export_data:
        items = t.get("items", [])
        if not items:
            items = [{"item_number": "", "description": "No items", "quantity": "", "unit": ""}]
        for item in items:
            new_rows.append({
                "Tender ID": t.get("id", ""),
                "Reference": t.get("reference", ""),
                "Department": t.get("department", ""),
                "Title": t.get("title", ""),
                "Closing Date": t.get("closing_date", ""),
                "Item #": item.get("item_number", ""),
                "Description": item.get("description", ""),
                "Quantity": item.get("quantity", ""),
                "Unit": item.get("unit", ""),
                "OCR Confidence": f"{t.get('ocr_confidence', 0):.1f}%",
                "Export Date": datetime.now().strftime("%Y-%m-%d %H:%M")
            })

    new_df = pd.DataFrame(new_rows)

    # If master file exists, load it and append (avoiding duplicates by Tender ID + Item #)
    if export_path.exists():
        try:
            existing_df = pd.read_excel(str(export_path))
            # Create unique key for deduplication
            existing_df['_key'] = existing_df['Tender ID'].astype(str) + '_' + existing_df['Item #'].astype(str)
            new_df['_key'] = new_df['Tender ID'].astype(str) + '_' + new_df['Item #'].astype(str)

            # Only add rows that don't already exist
            existing_keys = set(existing_df['_key'].tolist())
            new_df_filtered = new_df[~new_df['_key'].isin(existing_keys)]

            # Drop the helper column
            existing_df = existing_df.drop(columns=['_key'])
            new_df_filtered = new_df_filtered.drop(columns=['_key'])

            # Combine existing + new rows
            combined_df = pd.concat([existing_df, new_df_filtered], ignore_index=True)
        except Exception as e:
            print(f"Error reading existing Excel: {e}, creating new file")
            combined_df = new_df.drop(columns=['_key'], errors='ignore')
    else:
        combined_df = new_df.drop(columns=['_key'], errors='ignore')

    # Save to master file
    combined_df.to_excel(str(export_path), index=False)

    return FileResponse(
        path=str(export_path),
        filename="tenders_master.xlsx",
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
# SCRAPER ENDPOINTS
# ============================================================================

class ScraperRequest(BaseModel):
    department: Optional[str] = None  # 'Medical Store', 'Biomedical Engineering', or None for all
    headless: bool = True
    max_pages: int = 10

async def run_scraper_task(mode: str = "scrape", pdf_folder: str = None, output_folder: str = None):
    """Run the scraper script as a background task"""
    global scraper_status

    scraper_status["running"] = True
    scraper_status["progress"] = 0.0
    scraper_status["message"] = "Starting scraper..."

    try:
        cmd = [SCRAPER_PYTHON, SCRAPER_SCRIPT]

        if mode == "batch" and pdf_folder:
            cmd.extend(["--batch", pdf_folder])
            if output_folder:
                cmd.extend(["--output", output_folder])
            cmd.extend(["--pages", "10"])
        else:
            # Normal scraping mode
            cmd.extend(["--headless"])

        scraper_status["message"] = f"Running: {' '.join(cmd)}"

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        scraper_status["pid"] = process.pid

        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            scraper_status["message"] = "Scraper completed successfully"
            scraper_status["progress"] = 100.0
        else:
            error_msg = stderr.decode() if stderr else "Unknown error"
            scraper_status["message"] = f"Scraper failed: {error_msg[:200]}"

    except Exception as e:
        scraper_status["message"] = f"Error: {str(e)}"

    finally:
        scraper_status["running"] = False
        scraper_status["last_run"] = datetime.now().isoformat()
        scraper_status["pid"] = None


@app.get("/api/scraper/status")
async def get_scraper_status():
    """Get current scraper status"""
    # Check OCR results for recent files
    tenders_found = 0
    if os.path.exists(OCR_RESULTS_DIR):
        for f in os.listdir(OCR_RESULTS_DIR):
            if f.endswith('.xlsx') or f.endswith('.json'):
                tenders_found += 1

    return {
        **scraper_status,
        "script_exists": os.path.exists(SCRAPER_SCRIPT),
        "python_exists": os.path.exists(SCRAPER_PYTHON),
        "tenders_dir": MOH_TENDERS_DIR,
        "results_dir": OCR_RESULTS_DIR,
        "result_files": tenders_found
    }


@app.post("/api/scraper/start")
async def start_scraper(background_tasks: BackgroundTasks):
    """Start the MOH website scraper to download new tenders"""
    global scraper_status

    if scraper_status["running"]:
        raise HTTPException(status_code=400, detail="Scraper is already running")

    if not os.path.exists(SCRAPER_SCRIPT):
        raise HTTPException(status_code=404, detail=f"Scraper script not found: {SCRAPER_SCRIPT}")

    if not os.path.exists(SCRAPER_PYTHON):
        raise HTTPException(status_code=404, detail=f"Python not found: {SCRAPER_PYTHON}")

    background_tasks.add_task(run_scraper_task, "scrape")

    return {
        "message": "Scraper started",
        "script": SCRAPER_SCRIPT
    }


@app.post("/api/scraper/batch-ocr")
async def start_batch_ocr(
    background_tasks: BackgroundTasks,
    folder: str = Query(default=None, description="PDF folder to process"),
    department: str = Query(default=None, description="Department: 'medical' or 'biomedical'")
):
    """Start batch OCR processing on existing PDFs using your script"""
    global scraper_status

    if scraper_status["running"]:
        raise HTTPException(status_code=400, detail="Scraper/OCR is already running")

    # Determine which folder to process
    if folder:
        pdf_folder = folder
    elif department:
        if "medical" in department.lower():
            pdf_folder = os.path.join(MOH_TENDERS_DIR, "Medical_Store")
        elif "biomedical" in department.lower():
            pdf_folder = os.path.join(MOH_TENDERS_DIR, "Biomedical_Engineering")
        else:
            pdf_folder = MOH_TENDERS_DIR
    else:
        pdf_folder = MOH_TENDERS_DIR

    if not os.path.exists(pdf_folder):
        raise HTTPException(status_code=404, detail=f"Folder not found: {pdf_folder}")

    background_tasks.add_task(run_scraper_task, "batch", pdf_folder, OCR_RESULTS_DIR)

    return {
        "message": "Batch OCR started",
        "pdf_folder": pdf_folder,
        "output_folder": OCR_RESULTS_DIR
    }


@app.post("/api/scraper/stop")
async def stop_scraper():
    """Stop the running scraper"""
    global scraper_status

    if not scraper_status["running"]:
        return {"message": "Scraper is not running"}

    if scraper_status["pid"]:
        try:
            import signal
            os.kill(scraper_status["pid"], signal.SIGTERM)
            scraper_status["message"] = "Scraper stopped by user"
        except ProcessLookupError:
            pass
        except Exception as e:
            return {"message": f"Failed to stop: {str(e)}"}

    scraper_status["running"] = False
    scraper_status["pid"] = None

    return {"message": "Scraper stopped"}


@app.get("/api/scraper/results")
async def get_scraper_results():
    """Get list of OCR result files"""
    if not os.path.exists(OCR_RESULTS_DIR):
        return {"results": [], "total": 0}

    results = []
    for f in sorted(os.listdir(OCR_RESULTS_DIR), reverse=True):
        if f.endswith('.xlsx') or f.endswith('.json'):
            filepath = os.path.join(OCR_RESULTS_DIR, f)
            stat = os.stat(filepath)
            results.append({
                "filename": f,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "type": "excel" if f.endswith('.xlsx') else "json"
            })

    return {"results": results[:50], "total": len(results)}


@app.get("/api/scraper/download/{filename}")
async def download_result_file(filename: str):
    """Download an OCR result file"""
    filepath = os.path.join(OCR_RESULTS_DIR, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if filename.endswith('.xlsx') else "application/json"

    return FileResponse(
        path=filepath,
        filename=filename,
        media_type=media_type
    )


@app.get("/api/scraper/pdfs")
async def list_pdf_folders():
    """List available PDF folders for batch processing"""
    folders = []

    if os.path.exists(MOH_TENDERS_DIR):
        for item in os.listdir(MOH_TENDERS_DIR):
            item_path = os.path.join(MOH_TENDERS_DIR, item)
            if os.path.isdir(item_path):
                pdf_count = sum(1 for f in os.listdir(item_path) if f.lower().endswith('.pdf'))
                if pdf_count > 0:
                    folders.append({
                        "name": item,
                        "path": item_path,
                        "pdf_count": pdf_count
                    })

    return {"folders": folders, "base_dir": MOH_TENDERS_DIR}


# ============================================================================
# TENDER MANAGEMENT API (with AI Enhancement)
# ============================================================================

# Import tender manager
try:
    from tender_manager import get_tender_manager, get_ai_engine
    TENDER_MANAGER_AVAILABLE = True
    print("✅ Tender Manager loaded with AI enhancement")
except ImportError as e:
    print(f"Warning: Tender Manager not available: {e}")
    TENDER_MANAGER_AVAILABLE = False


@app.get("/api/manager/dashboard")
async def get_manager_dashboard():
    """Get tender management dashboard statistics"""
    if not TENDER_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Tender Manager not available")

    manager = get_tender_manager()
    return manager.get_dashboard_stats()


@app.get("/api/manager/tenders")
async def get_managed_tenders(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=200),
    search: str = Query(default=None),
    department: str = Query(default=None)
):
    """Get paginated list of tenders with filtering"""
    if not TENDER_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Tender Manager not available")

    manager = get_tender_manager()
    return manager.get_all_tenders(page=page, limit=limit, search=search, department=department)


@app.get("/api/manager/tenders/{reference}")
async def get_managed_tender(reference: str):
    """Get a single tender by reference"""
    if not TENDER_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Tender Manager not available")

    manager = get_tender_manager()
    tender = manager.get_tender(reference)
    if not tender:
        raise HTTPException(status_code=404, detail="Tender not found")
    return tender


@app.post("/api/manager/enhance/{reference}")
async def enhance_single_tender(reference: str):
    """Apply AI enhancement to a single tender"""
    if not TENDER_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Tender Manager not available")

    manager = get_tender_manager()
    result = manager.enhance_tender(reference)
    if 'error' in result:
        raise HTTPException(status_code=404, detail=result['error'])
    return result


class EnhanceBatchRequest(BaseModel):
    references: Optional[List[str]] = None
    limit: int = 10


@app.post("/api/manager/enhance-batch")
async def enhance_batch_tenders(request: EnhanceBatchRequest, background_tasks: BackgroundTasks):
    """Apply AI enhancement to multiple tenders"""
    if not TENDER_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Tender Manager not available")

    manager = get_tender_manager()

    # Run in background for large batches
    if request.limit > 5:
        background_tasks.add_task(manager.enhance_batch, request.references, request.limit)
        return {"message": f"Started enhancing up to {request.limit} tenders in background"}

    result = manager.enhance_batch(request.references, request.limit)
    return result


@app.get("/api/manager/search")
async def search_tender_items(
    q: str = Query(..., min_length=2),
    limit: int = Query(default=50, ge=1, le=200)
):
    """Search across all tender items"""
    if not TENDER_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Tender Manager not available")

    manager = get_tender_manager()
    return {"results": manager.search_items(q, limit)}


@app.get("/api/manager/departments")
async def get_departments():
    """Get list of unique departments"""
    if not TENDER_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Tender Manager not available")

    manager = get_tender_manager()
    return {"departments": manager.get_departments()}


@app.post("/api/manager/reload")
async def reload_tender_data():
    """Reload all tender data from OCR results"""
    if not TENDER_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Tender Manager not available")

    manager = get_tender_manager()
    manager.reload_from_ocr()
    return {"message": "Reloaded", "total_tenders": len(manager.tenders)}


@app.get("/api/manager/ai-stats")
async def get_ai_stats():
    """Get AI engine statistics"""
    if not TENDER_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Tender Manager not available")

    engine = get_ai_engine()
    return engine.get_stats()


class ExportRequest(BaseModel):
    references: Optional[List[str]] = None


@app.post("/api/manager/export")
async def export_managed_tenders(request: ExportRequest):
    """Export tenders to Excel file"""
    if not TENDER_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Tender Manager not available")

    manager = get_tender_manager()
    try:
        filepath = manager.export_to_excel(request.references)
        return {"filepath": filepath, "filename": os.path.basename(filepath)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/manager/export/download/{filename}")
async def download_export_file(filename: str):
    """Download an exported file"""
    export_dir = os.path.expanduser("~/Documents/MOH Tenders/ai_enhanced")
    filepath = os.path.join(export_dir, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=filepath,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Starting MOH Tender OCR API Server...")
    print(f"Upload directory: {UPLOAD_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"OCR Available: {OCR_AVAILABLE}")
    print(f"Tender Manager Available: {TENDER_MANAGER_AVAILABLE}")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
