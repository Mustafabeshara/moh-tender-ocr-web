"""
MOH Tender OCR Web API
FastAPI application for extracting data from tender PDFs
"""

import os
import uuid
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .models import (
    TenderData,
    ExtractionJob,
    ExtractionStatus,
    FileUploadResponse,
    ExtractionRequest,
    ExtractionSummary,
    HealthResponse
)
from .ocr_engine import (
    OCREngine,
    TenderExtractor,
    check_tesseract,
    check_poppler,
    OCRConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Storage for jobs (in production, use Redis or database)
jobs_storage: Dict[str, ExtractionJob] = {}

# Upload directory
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/tmp/moh_uploads"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/tmp/moh_results"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting MOH Tender OCR API...")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Check dependencies
    tesseract_ok = check_tesseract()
    poppler_ok = check_poppler()

    if not tesseract_ok:
        logger.warning("Tesseract OCR not found - OCR features will be limited")
    if not poppler_ok:
        logger.warning("Poppler not found - PDF conversion will fail")

    logger.info("API startup complete")

    yield

    # Shutdown
    logger.info("Shutting down MOH Tender OCR API...")


# Create FastAPI app
app = FastAPI(
    title="MOH Tender OCR API",
    description="API for extracting structured data from MOH Kuwait tender PDFs using OCR",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """API root endpoint"""
    return {
        "name": "MOH Tender OCR API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and dependencies"""
    return HealthResponse(
        status="healthy",
        tesseract_available=check_tesseract(),
        poppler_available=check_poppler(),
        version="1.0.0"
    )


# ============================================================================
# FILE UPLOAD ENDPOINTS
# ============================================================================

@app.post("/upload", response_model=FileUploadResponse, tags=["Upload"])
async def upload_files(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload PDF files for OCR processing"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Validate files
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.filename}. Only PDF files are accepted."
            )

    # Create job
    job_id = str(uuid.uuid4())
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded files
    saved_files = []
    for file in files:
        file_path = job_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        saved_files.append(str(file_path))
        logger.info(f"Saved file: {file.filename}")

    # Create job record
    job = ExtractionJob(
        job_id=job_id,
        status=ExtractionStatus.PENDING,
        total_files=len(saved_files),
        created_at=datetime.now()
    )
    jobs_storage[job_id] = job

    return FileUploadResponse(
        job_id=job_id,
        message=f"Successfully uploaded {len(files)} file(s)",
        files_received=len(files),
        status=ExtractionStatus.PENDING
    )


# ============================================================================
# EXTRACTION ENDPOINTS
# ============================================================================

def process_extraction_job(
    job_id: str,
    languages: List[str],
    dpi: int,
    max_pages: int,
    department: str
):
    """Background task to process OCR extraction"""
    job = jobs_storage.get(job_id)
    if not job:
        logger.error(f"Job not found: {job_id}")
        return

    try:
        job.status = ExtractionStatus.PROCESSING
        job_dir = UPLOAD_DIR / job_id

        extractor = TenderExtractor(languages=languages, dpi=dpi)

        pdf_files = list(job_dir.glob("*.pdf"))
        results = []

        for i, pdf_path in enumerate(pdf_files):
            logger.info(f"Processing {pdf_path.name} ({i+1}/{len(pdf_files)})")

            try:
                tender_data = extractor.process_pdf(
                    str(pdf_path),
                    department=department,
                    max_pages=max_pages
                )
                tender_data.id = str(uuid.uuid4())
                results.append(tender_data)
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {e}")
                job.errors.append(f"Error processing {pdf_path.name}: {str(e)}")

            job.processed_files = i + 1
            job.progress = int((i + 1) / len(pdf_files) * 100)

        job.results = results
        job.status = ExtractionStatus.COMPLETED
        job.completed_at = datetime.now()

        logger.info(f"Job {job_id} completed: {len(results)} tenders extracted")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        job.status = ExtractionStatus.FAILED
        job.errors.append(str(e))


@app.post("/extract/{job_id}", response_model=ExtractionJob, tags=["Extraction"])
async def start_extraction(
    job_id: str,
    request: ExtractionRequest,
    background_tasks: BackgroundTasks
):
    """Start OCR extraction for uploaded files"""
    job = jobs_storage.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status not in [ExtractionStatus.PENDING, ExtractionStatus.FAILED]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot start extraction. Job status: {job.status}"
        )

    # Start background processing
    background_tasks.add_task(
        process_extraction_job,
        job_id,
        request.languages,
        request.dpi,
        request.max_pages,
        request.department
    )

    job.status = ExtractionStatus.PROCESSING
    return job


@app.get("/jobs/{job_id}", response_model=ExtractionJob, tags=["Jobs"])
async def get_job_status(job_id: str):
    """Get extraction job status and results"""
    job = jobs_storage.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/jobs", response_model=List[ExtractionJob], tags=["Jobs"])
async def list_jobs(
    status: Optional[ExtractionStatus] = None,
    limit: int = Query(default=50, le=100)
):
    """List all extraction jobs"""
    jobs = list(jobs_storage.values())

    if status:
        jobs = [j for j in jobs if j.status == status]

    # Sort by creation date (newest first)
    jobs.sort(key=lambda x: x.created_at, reverse=True)

    return jobs[:limit]


@app.delete("/jobs/{job_id}", tags=["Jobs"])
async def delete_job(job_id: str):
    """Delete a job and its files"""
    job = jobs_storage.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Remove uploaded files
    job_dir = UPLOAD_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)

    # Remove from storage
    del jobs_storage[job_id]

    return {"message": f"Job {job_id} deleted"}


# ============================================================================
# RESULTS ENDPOINTS
# ============================================================================

@app.get("/results/{job_id}", response_model=List[TenderData], tags=["Results"])
async def get_results(job_id: str):
    """Get extraction results for a job"""
    job = jobs_storage.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != ExtractionStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Results not ready. Job status: {job.status}"
        )

    return job.results


@app.get("/results/{job_id}/summary", response_model=ExtractionSummary, tags=["Results"])
async def get_results_summary(job_id: str):
    """Get summary of extraction results"""
    job = jobs_storage.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != ExtractionStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Results not ready. Job status: {job.status}"
        )

    results = job.results

    total_items = sum(len(t.items) for t in results)
    arabic_tenders = len([t for t in results if t.has_arabic_content])
    with_specs = len([t for t in results if t.specifications_text])
    avg_confidence = (
        sum(t.ocr_confidence for t in results) / len(results)
        if results else 0
    )

    return ExtractionSummary(
        total_tenders=len(results),
        total_items=total_items,
        arabic_tenders=arabic_tenders,
        tenders_with_specs=with_specs,
        average_confidence=round(avg_confidence, 2),
        extraction_date=job.completed_at.isoformat() if job.completed_at else ""
    )


@app.get("/results/{job_id}/export", tags=["Results"])
async def export_results(
    job_id: str,
    format: str = Query(default="json", regex="^(json|csv)$")
):
    """Export results as JSON or CSV"""
    job = jobs_storage.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != ExtractionStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Results not ready. Job status: {job.status}"
        )

    if format == "json":
        return JSONResponse(
            content={
                "extraction_date": job.completed_at.isoformat() if job.completed_at else "",
                "total_tenders": len(job.results),
                "tenders": [t.model_dump() for t in job.results]
            }
        )

    elif format == "csv":
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "Reference", "Closing Date", "Posting Date", "Items Count",
            "Language", "OCR Confidence", "Has Specs", "Department"
        ])

        # Data
        for tender in job.results:
            writer.writerow([
                tender.reference_number,
                tender.closing_date,
                tender.posting_date,
                tender.items_count,
                tender.language,
                f"{tender.ocr_confidence:.1f}%",
                "Yes" if tender.specifications_text else "No",
                tender.department
            ])

        output.seek(0)

        return FileResponse(
            path=output,
            media_type="text/csv",
            filename=f"tender_results_{job_id}.csv"
        )


# ============================================================================
# SINGLE FILE EXTRACTION (Quick API)
# ============================================================================

@app.post("/extract-single", response_model=TenderData, tags=["Quick Extract"])
async def extract_single_file(
    file: UploadFile = File(...),
    languages: str = Query(default="eng+ara"),
    dpi: int = Query(default=300, ge=100, le=600),
    max_pages: int = Query(default=10, ge=1, le=50),
    department: str = Query(default="Biomedical Engineering")
):
    """Quick extraction from a single PDF (synchronous)"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Save temporarily
    temp_path = UPLOAD_DIR / f"temp_{uuid.uuid4()}.pdf"

    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Process
        lang_list = languages.split("+")
        extractor = TenderExtractor(languages=lang_list, dpi=dpi)

        result = extractor.process_pdf(
            str(temp_path),
            department=department,
            max_pages=max_pages
        )
        result.id = str(uuid.uuid4())

        return result

    finally:
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
