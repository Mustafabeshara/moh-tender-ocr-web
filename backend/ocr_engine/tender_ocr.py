# -*- coding: utf-8 -*-
"""
MOH Tender OCR Extractor - Enhanced Version
============================================
Extracts structured data from MOH tender PDFs using OCR:
- Reference Number
- Closing Date
- Items (Description, Quantity)
- Specifications (for Biomedical Engineering V2 files)

Requirements:
    pip install pytesseract pdf2image Pillow pandas openpyxl
    brew install tesseract tesseract-lang poppler (macOS)
"""

import os
import sys
import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Use web app directories for this improved copy
_BASE_DIR = Path(__file__).parent.parent.parent
_LOG_DIR = _BASE_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "base_folder": os.path.expanduser("~/Documents/MOH Tenders"),
    "output_folder": str(_BASE_DIR / "exports"),
    "log_file": str(_LOG_DIR / "ocr_extractor.log"),
    "tesseract_lang": "eng+ara",  # English + Arabic
    "dpi": 300,  # Higher DPI = better OCR accuracy
    "max_pages": 10,  # Max pages to process per PDF
    "debug_mode": False,  # Save intermediate images
}

# ============================================================================
# GARBAGE DETECTION - Filter non-item text
# ============================================================================

GARBAGE_PATTERNS = [
    # Page markers
    r'page\s*\d+\s*(?:of|/)\s*\d+',
    r'printed\s*on\s*:?\s*\d',
    # URLs and emails
    r'https?://[^\s]+',
    r'www\.[^\s]+',
    r'\.gov\.kw',
    r'\.com',
    r'\.net',
    # Phone/fax numbers
    r'(?:tel|fax|phone)\s*:?\s*[\d\-\s]+',
    r'\b\d{7,}\b',  # Long number sequences (phone numbers)
    # Addresses
    r'p\.?o\.?\s*box',
    r'code\s*no\s*:?',
    r'safat',
    r'kuwait\s*-',
    # MOH header/footer text
    r'ministry\s*of\s*health',
    r'medical\s*store',
    r'control\s*administration',
    r'biomedical\s*engineering',
    # Common document text
    r'total\s*value',
    r'estimated\s*value',
    r'unit\s*price',
    r'terms\s*and\s*conditions',
    r'tender\s*notice',
    r'invitation\s*to\s*bid',
]

def is_garbage_text(text: str) -> bool:
    """Check if text matches garbage patterns (not a real tender item)"""
    text_lower = text.lower().strip()

    # Too short
    if len(text_lower) < 8:
        return True

    # Check garbage patterns
    for pattern in GARBAGE_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True

    # Check if mostly numbers (likely dimensions, not items)
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count < 5:
        return True

    # All caps single words are usually headers
    if text.isupper() and len(text.split()) <= 2 and len(text) < 20:
        return True

    return False

def is_dimension_or_measurement(text: str, quantity: str) -> bool:
    """Check if quantity is a dimension/measurement, not a real quantity"""
    # If preceded by "QTY:" it's definitely a quantity
    if re.search(r'QTY[:\s]*' + re.escape(quantity), text, re.IGNORECASE):
        return False

    # Check if followed by measurement units
    measurement_pattern = re.compile(
        re.escape(quantity) + r'\s*(?:cm|mm|m|inch|ft|°|degrees?|kg|g|lb|%|x\s*\d)',
        re.IGNORECASE
    )
    if measurement_pattern.search(text):
        return True

    return False

def clean_description(desc: str) -> str:
    """Clean up item description"""
    # Remove leading item numbers
    desc = re.sub(r'^[\d\s.)\-:]+', '', desc)
    # Remove trailing punctuation
    desc = re.sub(r'[\s\-–:,.]+$', '', desc)
    # Normalize whitespace
    desc = re.sub(r'\s+', ' ', desc).strip()
    # Remove OCR artifacts
    desc = re.sub(r'[|_]{2,}', '', desc)
    return desc[:250]

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(CONFIG["log_file"], encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TenderItem:
    """Represents a single item in a tender"""
    item_number: str
    description: str
    quantity: str
    unit: str = ""
    specifications: str = ""
    
@dataclass
class TenderData:
    """Represents extracted tender data"""
    reference: str
    closing_date: str
    posting_date: str = ""
    department: str = ""
    title: str = ""
    items: List[TenderItem] = None
    specifications: str = ""
    raw_text: str = ""
    ocr_confidence: float = 0.0
    source_file: str = ""
    extraction_date: str = ""
    
    def __post_init__(self):
        if self.items is None:
            self.items = []
        if not self.extraction_date:
            self.extraction_date = datetime.now().isoformat()

# ============================================================================
# DEPENDENCY MANAGEMENT
# ============================================================================

def check_and_install_dependencies():
    """Check and install required dependencies"""
    required_packages = {
        'pytesseract': 'pytesseract',
        'pdf2image': 'pdf2image',
        'PIL': 'Pillow',
        'pandas': 'pandas',
        'openpyxl': 'openpyxl',
    }
    
    missing = []
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        logger.info(f"Installing missing packages: {missing}")
        import subprocess
        for package in missing:
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    package, "--break-system-packages"
                ])
            except subprocess.CalledProcessError:
                # Try without --break-system-packages
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Verify tesseract installation
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        logger.info("✅ Tesseract OCR is installed")
    except Exception as e:
        logger.error(f"❌ Tesseract not found. Install with: brew install tesseract tesseract-lang")
        logger.error(f"   Error: {e}")
        return False
    
    # Verify poppler installation (for pdf2image)
    try:
        from pdf2image import convert_from_path
        logger.info("✅ pdf2image is ready")
    except Exception as e:
        logger.error(f"❌ poppler not found. Install with: brew install poppler")
        logger.error(f"   Error: {e}")
        return False
    
    return True

# ============================================================================
# OCR ENGINE
# ============================================================================

class OCREngine:
    """Handles OCR operations on PDF files"""
    
    def __init__(self, language: str = "eng+ara", dpi: int = 300):
        self.language = language
        self.dpi = dpi
        self._setup()
    
    def _setup(self):
        """Initialize OCR engine"""
        import pytesseract
        from pdf2image import convert_from_path
        
        self.pytesseract = pytesseract
        self.convert_from_path = convert_from_path
        
        # Configure tesseract for better Arabic/English mixed text
        self.tesseract_config = r'--oem 3 --psm 6'
    
    def pdf_to_images(self, pdf_path: str, max_pages: int = 10) -> List:
        """Convert PDF pages to images"""
        try:
            images = self.convert_from_path(
                pdf_path,
                dpi=self.dpi,
                first_page=1,
                last_page=max_pages
            )
            logger.info(f"📄 Converted {len(images)} pages from {os.path.basename(pdf_path)}")
            return images
        except Exception as e:
            logger.error(f"❌ Error converting PDF to images: {e}")
            return []
    
    def extract_text_from_image(self, image) -> Tuple[str, float]:
        """Extract text from a single image with confidence score"""
        try:
            # Get detailed OCR data including confidence
            ocr_data = self.pytesseract.image_to_data(
                image, 
                lang=self.language,
                config=self.tesseract_config,
                output_type=self.pytesseract.Output.DICT
            )
            
            # Calculate average confidence
            confidences = [int(c) for c in ocr_data['conf'] if int(c) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Get full text
            text = self.pytesseract.image_to_string(
                image,
                lang=self.language,
                config=self.tesseract_config
            )
            
            return text, avg_confidence
        except Exception as e:
            logger.error(f"❌ OCR error: {e}")
            return "", 0.0
    
    def extract_text_from_pdf(self, pdf_path: str, max_pages: int = 10) -> Tuple[str, float]:
        """Extract all text from PDF using OCR"""
        images = self.pdf_to_images(pdf_path, max_pages)
        if not images:
            return "", 0.0
        
        all_text = []
        all_confidences = []
        
        for i, image in enumerate(images):
            logger.info(f"  📖 Processing page {i+1}/{len(images)}...")
            text, confidence = self.extract_text_from_image(image)
            all_text.append(f"--- PAGE {i+1} ---\n{text}")
            all_confidences.append(confidence)
        
        combined_text = "\n\n".join(all_text)
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        logger.info(f"✅ OCR complete. Confidence: {avg_confidence:.1f}%")
        return combined_text, avg_confidence

# ============================================================================
# DATA EXTRACTION PATTERNS
# ============================================================================

class TenderExtractor:
    """Extracts structured data from OCR text"""
    
    # Reference number patterns
    REFERENCE_PATTERNS = [
        r'(\d{1,2}[A-Z]{2,4}\d{2,4})',  # General: 5SSN17, 6BM123
        r'Reference\s*[:#]?\s*(\d+[A-Z]+\d+)',
        r'Tender\s*(?:No|Number|#)?\s*[:#]?\s*(\d+[A-Z]+\d+)',
        r'مناقصة\s*رقم\s*[:#]?\s*(\d+[A-Z]+\d+)',  # Arabic: Tender number
    ]
    
    # Date patterns (DD/MM/YYYY, YYYY-MM-DD, etc.)
    DATE_PATTERNS = [
        r'(\d{2}/\d{2}/\d{4})',
        r'(\d{4}-\d{2}-\d{2})',
        r'(\d{2}-\d{2}-\d{4})',
        r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',
    ]
    
    # Closing date indicators
    CLOSING_INDICATORS = [
        r'closing\s*date',
        r'deadline',
        r'last\s*date',
        r'submission\s*date',
        r'تاريخ\s*الإغلاق',  # Arabic: Closing date
        r'آخر\s*موعد',  # Arabic: Last date
        r'الموعد\s*النهائي',  # Arabic: Final deadline
    ]
    
    # Item patterns
    ITEM_PATTERNS = [
        # Pattern 1: Numbered items with quantity
        r'(\d+)\s*[.):-]\s*(.+?)\s+(\d+(?:,\d{3})*)\s*(PCS|pieces?|units?|each|nos?|qty|pcs|قطعة|وحدة)',
        # Pattern 2: Description then quantity
        r'([A-Za-z][^\n\r]{10,100}?)\s+(\d+(?:,\d{3})*)\s*(PCS|pieces?|units?|each|nos?|qty|pcs)',
        # Pattern 3: Table format with tabs/multiple spaces
        r'(\d+)\s{2,}(.{10,80}?)\s{2,}(\d+(?:,\d{3})*)',
    ]
    
    # Specification indicators
    SPEC_INDICATORS = [
        r'specification',
        r'technical\s*requirement',
        r'requirement',
        r'المواصفات',  # Arabic: Specifications
        r'المتطلبات\s*الفنية',  # Arabic: Technical requirements
    ]
    
    def __init__(self):
        self.ocr_engine = OCREngine(
            language=CONFIG["tesseract_lang"],
            dpi=CONFIG["dpi"]
        )
    
    def extract_reference(self, text: str, filename: str = "") -> str:
        """Extract tender reference number"""
        # Try filename first (usually contains reference)
        if filename:
            for pattern in self.REFERENCE_PATTERNS:
                match = re.search(pattern, filename, re.IGNORECASE)
                if match:
                    return match.group(1)
        
        # Try text content
        for pattern in self.REFERENCE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "UNKNOWN"
    
    def extract_closing_date(self, text: str) -> str:
        """Extract closing/deadline date"""
        text_lower = text.lower()
        
        # Find closing date section
        for indicator in self.CLOSING_INDICATORS:
            indicator_match = re.search(indicator, text_lower, re.IGNORECASE)
            if indicator_match:
                # Look for date near the indicator
                search_region = text[max(0, indicator_match.start()-50):indicator_match.end()+200]
                for date_pattern in self.DATE_PATTERNS:
                    date_match = re.search(date_pattern, search_region, re.IGNORECASE)
                    if date_match:
                        return date_match.group(1)
        
        # Fallback: find all dates and take the latest one
        all_dates = []
        for pattern in self.DATE_PATTERNS:
            dates = re.findall(pattern, text, re.IGNORECASE)
            all_dates.extend(dates)
        
        if all_dates:
            # Return the last date found (often the closing date)
            return all_dates[-1] if len(all_dates) > 1 else all_dates[0]
        
        return "UNKNOWN"
    
    def extract_items(self, text: str) -> List[TenderItem]:
        """Extract tender items with descriptions and quantities - IMPROVED WITH GARBAGE FILTERING"""
        items = []
        seen_descriptions = set()
        seen_item_numbers = set()

        # PATTERN 1: MOH standard format "1. ITEM DESCRIPTION QTY: N"
        # This is the most reliable pattern for MOH tenders
        qty_pattern = re.compile(
            r'(\d{1,3})\s*[.)]\s*([A-Z][^\n]{10,200}?)\s+(?:QTY|Qty|QUANTITY|Quantity)[:\s]*(\d+)',
            re.IGNORECASE
        )

        for match in qty_pattern.finditer(text):
            item_no = match.group(1)
            description = clean_description(match.group(2))
            quantity = match.group(3)

            # Skip garbage
            if is_garbage_text(description):
                continue
            if is_dimension_or_measurement(match.group(0), quantity):
                continue
            if item_no in seen_item_numbers:
                continue

            desc_key = description.lower()[:80]
            if desc_key in seen_descriptions or len(description) < 10:
                continue

            seen_item_numbers.add(item_no)
            seen_descriptions.add(desc_key)

            items.append(TenderItem(
                item_number=item_no,
                description=description,
                quantity=quantity,
                unit="PCS"
            ))

        # PATTERN 2: Items with explicit units like "10 PCS" or "5 units"
        if len(items) < 3:
            unit_pattern = re.compile(
                r'(\d{1,3})\s*[.)]\s*([A-Za-z][^\n]{10,150}?)\s+(\d+)\s*(pieces?|pcs?|units?|sets?|nos?|each|vial|amp|tab|cap|bot|kit)',
                re.IGNORECASE
            )

            for match in unit_pattern.finditer(text):
                item_no = match.group(1)
                description = clean_description(match.group(2))
                quantity = match.group(3)
                unit = match.group(4).upper()

                if is_garbage_text(description):
                    continue
                if is_dimension_or_measurement(match.group(0), quantity):
                    continue
                if item_no in seen_item_numbers:
                    continue

                desc_key = description.lower()[:80]
                if desc_key in seen_descriptions or len(description) < 10:
                    continue

                seen_item_numbers.add(item_no)
                seen_descriptions.add(desc_key)

                items.append(TenderItem(
                    item_number=item_no,
                    description=description,
                    quantity=quantity,
                    unit=unit
                ))

        # PATTERN 3: Medical Store tabular format "1 | ITEM_NAME UNIT QTY"
        if len(items) < 3:
            tabular_pattern = re.compile(
                r'^(\d{1,3})\s*\|?\s*([A-Z][A-Z0-9\s\-/.,()%#]+?)\s+(PFS|VIAL|AMP|TAB|CAP|BOT|BTL|PKT|TUBE|BAG|PCK|SET|EACH|EA|UNIT|BOX|PACK|PCS|KIT|PAIR|ROLL)\s+([\d,]+)$',
                re.IGNORECASE | re.MULTILINE
            )

            for match in tabular_pattern.finditer(text):
                item_no = match.group(1)
                description = match.group(2).strip()
                unit = match.group(3).upper()
                quantity = match.group(4).replace(',', '')

                if is_garbage_text(description):
                    continue
                if item_no in seen_item_numbers:
                    continue

                description = re.sub(r'\s+', ' ', description).strip()
                if len(description) < 5:
                    continue

                desc_key = description.lower()[:80]
                if desc_key in seen_descriptions:
                    continue

                seen_item_numbers.add(item_no)
                seen_descriptions.add(desc_key)

                items.append(TenderItem(
                    item_number=item_no,
                    description=description,
                    quantity=quantity,
                    unit=unit
                ))

        # PATTERN 4: More flexible - numbered items with any quantity at end of line
        # Catches: "1. DESCRIPTION TEXT HERE 100" or "2) Some item 50"
        if len(items) < 3:
            flexible_pattern = re.compile(
                r'^(\d{1,3})\s*[.)]\s*([A-Za-z][A-Za-z0-9\s\-/.,()%#&\'\"]+?)\s+(\d{1,5})$',
                re.MULTILINE
            )

            for match in flexible_pattern.finditer(text):
                item_no = match.group(1)
                description = clean_description(match.group(2))
                quantity = match.group(3)

                # Additional validation for flexible pattern
                if len(description) < 15:  # Require longer description
                    continue
                if is_garbage_text(description):
                    continue
                if item_no in seen_item_numbers:
                    continue
                # Skip if quantity looks like a year or phone number
                if int(quantity) > 50000:
                    continue

                desc_key = description.lower()[:80]
                if desc_key in seen_descriptions:
                    continue

                seen_item_numbers.add(item_no)
                seen_descriptions.add(desc_key)

                items.append(TenderItem(
                    item_number=item_no,
                    description=description,
                    quantity=quantity,
                    unit="PCS"
                ))

        # PATTERN 5: Medical equipment format - "Item: DESCRIPTION Qty: N"
        if len(items) < 3:
            item_qty_pattern = re.compile(
                r'(?:Item|SN|S\.N\.?|No\.?)\s*[:#]?\s*(\d{1,3})\s*[.):-]?\s*([A-Za-z][^\n]{15,200}?)\s+(?:Qty|QTY|Quantity|Q)[:\s]*(\d+)',
                re.IGNORECASE
            )

            for match in item_qty_pattern.finditer(text):
                item_no = match.group(1)
                description = clean_description(match.group(2))
                quantity = match.group(3)

                if is_garbage_text(description):
                    continue
                if item_no in seen_item_numbers:
                    continue

                desc_key = description.lower()[:80]
                if desc_key in seen_descriptions or len(description) < 10:
                    continue

                seen_item_numbers.add(item_no)
                seen_descriptions.add(desc_key)

                items.append(TenderItem(
                    item_number=item_no,
                    description=description,
                    quantity=quantity,
                    unit="PCS"
                ))

        # PATTERN 6: Arabic items with عدد (quantity)
        arabic_pattern = re.compile(
            r'(\d{1,3})\s*[.)]\s*([\u0600-\u06FF][^\n]{10,150}?)\s+(?:عدد|الكمية)[:\s]*(\d+)',
            re.IGNORECASE
        )

        for match in arabic_pattern.finditer(text):
            item_no = match.group(1)
            description = clean_description(match.group(2))
            quantity = match.group(3)

            if item_no in seen_item_numbers:
                continue

            desc_key = description.lower()[:80]
            if desc_key in seen_descriptions or len(description) < 10:
                continue

            seen_item_numbers.add(item_no)
            seen_descriptions.add(desc_key)

            items.append(TenderItem(
                item_number=item_no,
                description=description,
                quantity=quantity,
                unit="وحدة"
            ))

        # Sort by item number
        try:
            items.sort(key=lambda x: int(x.item_number) if x.item_number.isdigit() else 999)
        except:
            pass

        logger.info(f"📦 Extracted {len(items)} items (filtered garbage)")
        return items[:50]  # Limit to 50 items
    
    def extract_specifications(self, text: str) -> str:
        """Extract technical specifications section"""
        specs_text = []
        
        # Find specifications section
        for indicator in self.SPEC_INDICATORS:
            match = re.search(indicator, text, re.IGNORECASE)
            if match:
                # Extract text after the indicator
                start_pos = match.end()
                # Find the next section or end
                end_patterns = [r'\n\s*\d+\.', r'\n\s*Article', r'\n\s*Section', r'\n{3,}']
                end_pos = len(text)
                
                for end_pattern in end_patterns:
                    end_match = re.search(end_pattern, text[start_pos:])
                    if end_match:
                        end_pos = min(end_pos, start_pos + end_match.start())
                
                spec_section = text[start_pos:end_pos].strip()
                if len(spec_section) > 20:
                    specs_text.append(spec_section)
        
        return "\n\n".join(specs_text) if specs_text else ""
    
    def process_pdf(self, pdf_path: str) -> TenderData:
        """Process a single PDF and extract all data"""
        filename = os.path.basename(pdf_path)
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 Processing: {filename}")
        logger.info(f"{'='*60}")
        
        # Perform OCR
        text, confidence = self.ocr_engine.extract_text_from_pdf(
            pdf_path, 
            max_pages=CONFIG["max_pages"]
        )
        
        if not text:
            logger.warning(f"⚠️ No text extracted from {filename}")
            return TenderData(
                reference=self.extract_reference("", filename),
                closing_date="UNKNOWN",
                source_file=pdf_path,
                ocr_confidence=0.0
            )
        
        # Extract structured data
        tender_data = TenderData(
            reference=self.extract_reference(text, filename),
            closing_date=self.extract_closing_date(text),
            items=self.extract_items(text),
            specifications=self.extract_specifications(text),
            raw_text=text,
            ocr_confidence=confidence,
            source_file=pdf_path
        )
        
        logger.info(f"📋 Reference: {tender_data.reference}")
        logger.info(f"📅 Closing Date: {tender_data.closing_date}")
        logger.info(f"📦 Items: {len(tender_data.items)}")
        logger.info(f"📊 Confidence: {tender_data.ocr_confidence:.1f}%")
        
        return tender_data
    
    def process_biomedical_tender(self, v1_path: str, v2_path: str = None) -> TenderData:
        """Process Biomedical Engineering tender with V1 (main) and V2 (specs)"""
        filename = os.path.basename(v1_path)
        logger.info(f"\n{'='*60}")
        logger.info(f"🔬 Processing Biomedical Tender: {filename}")
        logger.info(f"{'='*60}")
        
        # Process main tender (V1)
        tender_data = self.process_pdf(v1_path)
        tender_data.department = "Biomedical Engineering"
        
        # Process specifications (V2) if available
        if v2_path and os.path.exists(v2_path):
            logger.info(f"📋 Processing V2 specifications: {os.path.basename(v2_path)}")
            v2_text, v2_confidence = self.ocr_engine.extract_text_from_pdf(v2_path, max_pages=5)
            
            if v2_text:
                # Extract specifications from V2
                v2_specs = self.extract_specifications(v2_text)
                if v2_specs:
                    tender_data.specifications = v2_specs
                else:
                    # Use full V2 text as specifications
                    tender_data.specifications = v2_text
                
                logger.info(f"✅ Specifications extracted from V2 ({len(tender_data.specifications)} chars)")
        
        return tender_data

# ============================================================================
# BATCH PROCESSOR
# ============================================================================

class BatchProcessor:
    """Process multiple PDFs and generate reports"""
    
    def __init__(self):
        self.extractor = TenderExtractor()
        self.results: List[TenderData] = []
    
    def process_medical_store(self, folder: str = None) -> List[TenderData]:
        """Process all Medical Store PDFs"""
        if folder is None:
            folder = os.path.join(CONFIG["base_folder"], "Medical_Store")
        
        logger.info(f"\n{'#'*60}")
        logger.info(f"🏥 Processing Medical Store Tenders")
        logger.info(f"📁 Folder: {folder}")
        logger.info(f"{'#'*60}")
        
        if not os.path.exists(folder):
            logger.error(f"❌ Folder not found: {folder}")
            return []
        
        pdf_files = [f for f in os.listdir(folder) if f.endswith('.pdf')]
        logger.info(f"📄 Found {len(pdf_files)} PDF files")
        
        results = []
        for i, pdf_file in enumerate(pdf_files):
            pdf_path = os.path.join(folder, pdf_file)
            logger.info(f"\n[{i+1}/{len(pdf_files)}] Processing {pdf_file}")
            
            try:
                tender_data = self.extractor.process_pdf(pdf_path)
                tender_data.department = "Medical Store"
                results.append(tender_data)
            except Exception as e:
                logger.error(f"❌ Error processing {pdf_file}: {e}")
        
        self.results.extend(results)
        return results
    
    def process_biomedical_engineering(self, folder: str = None) -> List[TenderData]:
        """Process all Biomedical Engineering PDFs (V1 + V2 specs)"""
        if folder is None:
            folder = os.path.join(CONFIG["base_folder"], "Biomedical_Engineering")
        
        logger.info(f"\n{'#'*60}")
        logger.info(f"🔬 Processing Biomedical Engineering Tenders")
        logger.info(f"📁 Folder: {folder}")
        logger.info(f"{'#'*60}")
        
        if not os.path.exists(folder):
            logger.error(f"❌ Folder not found: {folder}")
            return []
        
        pdf_files = [f for f in os.listdir(folder) if f.endswith('.pdf')]
        
        # Group by tender reference (V1, V2, V3)
        tender_groups = {}
        for pdf_file in pdf_files:
            # Extract reference from filename
            ref_match = re.search(r'(\d{1,2}[A-Z]{2,4}\d{2,4})', pdf_file)
            if ref_match:
                ref = ref_match.group(1)
                if ref not in tender_groups:
                    tender_groups[ref] = {'v1': None, 'v2': None, 'v3': None}
                
                if '_V1' in pdf_file or '_V1.' in pdf_file:
                    tender_groups[ref]['v1'] = os.path.join(folder, pdf_file)
                elif '_V2' in pdf_file or '_V2.' in pdf_file:
                    tender_groups[ref]['v2'] = os.path.join(folder, pdf_file)
                elif '_V3' in pdf_file or '_V3.' in pdf_file:
                    tender_groups[ref]['v3'] = os.path.join(folder, pdf_file)
                elif tender_groups[ref]['v1'] is None:
                    tender_groups[ref]['v1'] = os.path.join(folder, pdf_file)
        
        logger.info(f"📄 Found {len(tender_groups)} tender groups")
        
        results = []
        for i, (ref, files) in enumerate(tender_groups.items()):
            logger.info(f"\n[{i+1}/{len(tender_groups)}] Processing tender {ref}")
            
            try:
                if files['v1']:
                    tender_data = self.extractor.process_biomedical_tender(
                        v1_path=files['v1'],
                        v2_path=files['v2']
                    )
                    results.append(tender_data)
            except Exception as e:
                logger.error(f"❌ Error processing {ref}: {e}")
        
        self.results.extend(results)
        return results
    
    def export_to_json(self, output_path: str = None) -> str:
        """Export results to JSON"""
        if output_path is None:
            os.makedirs(CONFIG["output_folder"], exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(CONFIG["output_folder"], f"ocr_results_{timestamp}.json")
        
        # Convert to serializable format
        export_data = {
            "extraction_date": datetime.now().isoformat(),
            "total_tenders": len(self.results),
            "tenders": []
        }
        
        for tender in self.results:
            tender_dict = asdict(tender)
            tender_dict['items'] = [asdict(item) for item in tender.items]
            # Remove raw_text for cleaner export (it's huge)
            tender_dict.pop('raw_text', None)
            export_data['tenders'].append(tender_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 JSON exported to: {output_path}")
        return output_path
    
    def export_to_excel(self, output_path: str = None) -> str:
        """Export results to Excel with multiple sheets"""
        import pandas as pd
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment
        
        if output_path is None:
            os.makedirs(CONFIG["output_folder"], exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(CONFIG["output_folder"], f"OCR_Tender_Analysis_{timestamp}.xlsx")
        
        # Create summary data
        summary_data = []
        items_data = []
        specs_data = []
        
        for tender in self.results:
            summary_data.append({
                'Reference': tender.reference,
                'Department': tender.department,
                'Closing Date': tender.closing_date,
                'Items Count': len(tender.items),
                'Has Specs': 'Yes' if tender.specifications else 'No',
                'OCR Confidence': f"{tender.ocr_confidence:.1f}%",
                'Source File': os.path.basename(tender.source_file)
            })
            
            for item in tender.items:
                items_data.append({
                    'Reference': tender.reference,
                    'Item #': item.item_number,
                    'Description': item.description,
                    'Quantity': item.quantity,
                    'Unit': item.unit,
                    'Specifications': item.specifications
                })
            
            if tender.specifications:
                specs_data.append({
                    'Reference': tender.reference,
                    'Department': tender.department,
                    'Specifications': tender.specifications[:5000]  # Limit length
                })
        
        # Create Excel file
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary sheet
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='📋 Summary', index=False)
            
            # Items sheet
            df_items = pd.DataFrame(items_data)
            df_items.to_excel(writer, sheet_name='📦 Items', index=False)
            
            # Specifications sheet
            if specs_data:
                df_specs = pd.DataFrame(specs_data)
                df_specs.to_excel(writer, sheet_name='🔧 Specifications', index=False)
        
        logger.info(f"📊 Excel exported to: {output_path}")
        return output_path
    
    def print_summary(self):
        """Print extraction summary"""
        print(f"\n{'='*60}")
        print(f"📊 OCR EXTRACTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total tenders processed: {len(self.results)}")
        
        total_items = sum(len(t.items) for t in self.results)
        print(f"Total items extracted: {total_items}")
        
        avg_confidence = sum(t.ocr_confidence for t in self.results) / len(self.results) if self.results else 0
        print(f"Average OCR confidence: {avg_confidence:.1f}%")
        
        by_dept = {}
        for t in self.results:
            dept = t.department or "Unknown"
            by_dept[dept] = by_dept.get(dept, 0) + 1
        
        print(f"\nBy Department:")
        for dept, count in by_dept.items():
            print(f"  - {dept}: {count} tenders")
        
        print(f"{'='*60}\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║         MOH TENDER OCR EXTRACTOR - Enhanced               ║
    ║    Extracts: Reference, Dates, Items, Specifications      ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Check dependencies
    logger.info("🔧 Checking dependencies...")
    if not check_and_install_dependencies():
        logger.error("❌ Failed to setup dependencies. Exiting.")
        return None
    
    # Create processor
    processor = BatchProcessor()
    
    # Process Medical Store tenders
    processor.process_medical_store()
    
    # Process Biomedical Engineering tenders (with V2 specs)
    processor.process_biomedical_engineering()
    
    # Export results
    json_path = processor.export_to_json()
    excel_path = processor.export_to_excel()
    
    # Print summary
    processor.print_summary()
    
    print(f"📁 Results saved to:")
    print(f"   JSON: {json_path}")
    print(f"   Excel: {excel_path}")
    
    return processor.results

def process_single_pdf(pdf_path: str) -> TenderData:
    """Process a single PDF file"""
    if not check_and_install_dependencies():
        return None
    
    extractor = TenderExtractor()
    return extractor.process_pdf(pdf_path)

def test_ocr():
    """Test OCR on a sample file"""
    logger.info("🧪 Testing OCR...")
    
    if not check_and_install_dependencies():
        return False
    
    # Find a sample PDF
    medical_store = os.path.join(CONFIG["base_folder"], "Medical_Store")
    if os.path.exists(medical_store):
        pdfs = [f for f in os.listdir(medical_store) if f.endswith('.pdf')]
        if pdfs:
            test_pdf = os.path.join(medical_store, pdfs[0])
            logger.info(f"Testing with: {test_pdf}")
            
            result = process_single_pdf(test_pdf)
            if result:
                print(f"\n📋 Reference: {result.reference}")
                print(f"📅 Closing Date: {result.closing_date}")
                print(f"📦 Items found: {len(result.items)}")
                for item in result.items[:5]:
                    print(f"   - {item.item_number}: {item.description[:50]}... ({item.quantity} {item.unit})")
                return True
    
    logger.error("❌ No test files found")
    return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            test_ocr()
        elif sys.argv[1] == "--single" and len(sys.argv) > 2:
            result = process_single_pdf(sys.argv[2])
            if result:
                print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))
        else:
            print("Usage:")
            print("  python moh_ocr_extractor.py          # Process all PDFs")
            print("  python moh_ocr_extractor.py --test   # Test OCR on sample")
            print("  python moh_ocr_extractor.py --single <pdf_path>  # Process single PDF")
    else:
        main()
