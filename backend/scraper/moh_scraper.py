"""
MOH Tender Scraper Module
Scrapes tender documents from the Kuwait Ministry of Health website
"""
import os
import re
import time
import random
import logging
import requests
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from urllib.parse import urljoin, urlparse
from pathlib import Path
from enum import Enum

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.action_chains import ActionChains
    from bs4 import BeautifulSoup
    import chromedriver_autoinstaller
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Warning: Selenium not available. Install with: pip install selenium beautifulsoup4 chromedriver-autoinstaller")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScraperStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ScraperConfig:
    """Configuration for the MOH scraper"""
    home_url: str = "https://www.moh.gov.kw/en/Pages/default.aspx"
    headless: bool = True
    fast_mode: bool = True
    max_tenders: int = 50
    download_timeout: int = 60
    departments: List[str] = field(default_factory=lambda: ['Medical Store', 'Biomedical Engineering'])


@dataclass
class TenderLink:
    """Represents a tender link found during scraping"""
    url: str
    title: str
    link_type: str  # 'pdf' or 'page'
    tender_ref: str = ""


@dataclass
class ScrapedTender:
    """Represents a scraped tender"""
    reference: str
    url: str
    title: str
    department: str
    pdf_path: str = ""
    downloaded: bool = False
    error: str = ""


class MOHScraper:
    """
    Web scraper for Kuwait Ministry of Health tender documents
    """

    def __init__(self, config: ScraperConfig = None, download_dir: str = None):
        self.config = config or ScraperConfig()
        self.download_dir = download_dir or os.path.expanduser("~/Documents/MOH Tenders/downloads")
        self.driver = None
        self.status = ScraperStatus.IDLE
        self.progress = 0.0
        self.message = ""
        self.scraped_tenders: List[ScrapedTender] = []
        self.current_department = ""

        # Create download directory
        os.makedirs(self.download_dir, exist_ok=True)

    def _human_delay(self, min_time=1.0, max_time=3.0):
        """Add random human-like delay"""
        if self.config.fast_mode:
            time.sleep(random.uniform(0.3, 0.8))
        else:
            time.sleep(random.uniform(min_time, max_time))

    def _setup_driver(self):
        """Setup Chrome WebDriver with anti-detection measures"""
        if not SELENIUM_AVAILABLE:
            raise RuntimeError("Selenium is not installed. Run: pip install selenium beautifulsoup4 chromedriver-autoinstaller")

        logger.info("Setting up Chrome WebDriver...")
        chromedriver_autoinstaller.install()

        options = webdriver.ChromeOptions()

        if self.config.headless:
            options.add_argument('--headless=new')

        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')

        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)

        prefs = {
            "download.default_directory": self.download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "plugins.always_open_pdf_externally": True
        }
        options.add_experimental_option("prefs", prefs)

        self.driver = webdriver.Chrome(options=options)
        self.driver.execute_cdp_cmd('Network.setUserAgentOverride', {
            "userAgent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        logger.info("WebDriver ready")

    def _safe_click(self, element, wait_time=0.5):
        """Safely click an element with retry logic"""
        try:
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
            time.sleep(wait_time)
            self.driver.execute_script("arguments[0].click();", element)
            return True
        except Exception:
            try:
                ActionChains(self.driver).move_to_element(element).click().perform()
                return True
            except Exception as e:
                logger.warning(f"Click failed: {e}")
                return False

    def _navigate_to_tenders(self, department: str) -> bool:
        """Navigate to the tenders page for a specific department"""
        try:
            logger.info(f"Navigating to MOH website for {department}...")
            self.driver.get(self.config.home_url)
            self._human_delay(2, 4)

            # Look for tenders/bids links
            tender_keywords = ['tender', 'bid', 'procurement', 'مناقصة', 'عطاء']
            links = self.driver.find_elements(By.TAG_NAME, 'a')

            for link in links:
                href = link.get_attribute('href') or ''
                text = link.text.lower()

                for keyword in tender_keywords:
                    if keyword in href.lower() or keyword in text:
                        logger.info(f"Found tender link: {link.text[:50]}")
                        self._safe_click(link)
                        self._human_delay(2, 3)
                        return True

            return True

        except Exception as e:
            logger.error(f"Navigation error: {e}")
            return False

    def _extract_tender_links(self) -> List[TenderLink]:
        """Extract all tender links from current page"""
        tender_links = []

        try:
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')

            # Find PDF links
            pdf_links = soup.find_all('a', href=re.compile(r'\.pdf', re.IGNORECASE))
            for link in pdf_links:
                href = link.get('href', '')
                if href:
                    full_url = urljoin(self.driver.current_url, href)
                    text = link.get_text(strip=True) or os.path.basename(href)
                    tender_ref = self._extract_tender_ref(full_url)
                    tender_links.append(TenderLink(
                        url=full_url,
                        title=text,
                        link_type='pdf',
                        tender_ref=tender_ref
                    ))

            # Find tender detail pages
            detail_patterns = [
                r'tender.*detail', r'view.*tender', r'tender.*info',
                r'bid.*detail', r'\?id=\d+', r'itemid=\d+'
            ]

            all_links = soup.find_all('a', href=True)
            for link in all_links:
                href = link.get('href', '')
                for pattern in detail_patterns:
                    if re.search(pattern, href, re.IGNORECASE):
                        full_url = urljoin(self.driver.current_url, href)
                        text = link.get_text(strip=True)
                        tender_ref = self._extract_tender_ref(full_url)
                        tender_links.append(TenderLink(
                            url=full_url,
                            title=text,
                            link_type='page',
                            tender_ref=tender_ref
                        ))
                        break

            logger.info(f"Found {len(tender_links)} tender links")
            return tender_links

        except Exception as e:
            logger.error(f"Error extracting links: {e}")
            return []

    def _extract_tender_ref(self, url: str) -> str:
        """Extract tender reference from URL"""
        patterns = [
            r'(\d{1,2}[A-Z]{2,3}\d{2,4})',
            r'tender[_-]?(\d+)',
            r'id[_=](\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1)

        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        name = os.path.splitext(filename)[0]
        return re.sub(r'[^\w\-]', '_', name)[:50]

    def start(self) -> Dict[str, Any]:
        """Start the scraping process"""
        if self.status == ScraperStatus.RUNNING:
            return {"error": "Scraper is already running"}

        self.status = ScraperStatus.RUNNING
        self.progress = 0.0
        self.message = "Initializing scraper..."
        self.scraped_tenders = []

        try:
            self._setup_driver()

            total_departments = len(self.config.departments)

            for idx, department in enumerate(self.config.departments):
                self.current_department = department
                self.message = f"Scraping {department}..."
                self.progress = (idx / total_departments) * 100

                if not self._navigate_to_tenders(department):
                    logger.error(f"Failed to navigate to {department}")
                    continue

                self._human_delay(2, 4)

                tender_links = self._extract_tender_links()

                for i, tender_link in enumerate(tender_links[:self.config.max_tenders]):
                    self.message = f"Processing {department}: {i+1}/{len(tender_links)}"

                    scraped = ScrapedTender(
                        reference=tender_link.tender_ref,
                        url=tender_link.url,
                        title=tender_link.title,
                        department=department
                    )

                    if tender_link.link_type == 'pdf':
                        # Download PDF
                        dept_dir = os.path.join(self.download_dir, department.replace(' ', '_'))
                        os.makedirs(dept_dir, exist_ok=True)

                        pdf_filename = f"{tender_link.tender_ref}.pdf"
                        pdf_path = os.path.join(dept_dir, pdf_filename)

                        if os.path.exists(pdf_path):
                            scraped.pdf_path = pdf_path
                            scraped.downloaded = True
                            logger.info(f"File exists: {pdf_path}")
                        else:
                            success = download_pdf(tender_link.url, pdf_path)
                            if success:
                                scraped.pdf_path = pdf_path
                                scraped.downloaded = True
                            else:
                                scraped.error = "Download failed"

                    self.scraped_tenders.append(scraped)
                    self._human_delay(1, 2)

            self.status = ScraperStatus.COMPLETED
            self.progress = 100.0
            self.message = f"Completed! Found {len(self.scraped_tenders)} tenders"

            return {
                "status": "completed",
                "total_found": len(self.scraped_tenders),
                "downloaded": sum(1 for t in self.scraped_tenders if t.downloaded),
                "tenders": [
                    {
                        "reference": t.reference,
                        "url": t.url,
                        "title": t.title,
                        "department": t.department,
                        "pdf_path": t.pdf_path,
                        "downloaded": t.downloaded,
                        "error": t.error
                    }
                    for t in self.scraped_tenders
                ]
            }

        except Exception as e:
            logger.error(f"Scraper error: {e}")
            self.status = ScraperStatus.FAILED
            self.message = f"Error: {str(e)}"
            return {"error": str(e)}

        finally:
            self.stop()

    def stop(self):
        """Stop the scraper and cleanup"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None
        self.status = ScraperStatus.IDLE
        logger.info("Scraper stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current scraper status"""
        return {
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "current_department": self.current_department,
            "tenders_found": len(self.scraped_tenders)
        }


def download_pdf(url: str, save_path: str, timeout: int = 60) -> bool:
    """Download PDF file with progress tracking"""
    try:
        logger.info(f"Downloading: {os.path.basename(save_path)}")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/pdf,*/*',
        }

        response = requests.get(url, headers=headers, stream=True, timeout=timeout)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '')
        if 'pdf' not in content_type.lower() and not url.lower().endswith('.pdf'):
            logger.warning(f"Not a PDF: {content_type}")
            return False

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        file_size = os.path.getsize(save_path)
        if file_size < 1000:
            logger.warning(f"File too small ({file_size} bytes)")
            os.remove(save_path)
            return False

        logger.info(f"Downloaded: {os.path.basename(save_path)} ({file_size/1024:.1f} KB)")
        return True

    except Exception as e:
        logger.error(f"Download failed: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False


def extract_tender_links(html_content: str, base_url: str) -> List[Dict]:
    """Extract tender links from HTML content (for use without Selenium)"""
    tender_links = []

    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find PDF links
        pdf_links = soup.find_all('a', href=re.compile(r'\.pdf', re.IGNORECASE))
        for link in pdf_links:
            href = link.get('href', '')
            if href:
                full_url = urljoin(base_url, href)
                text = link.get_text(strip=True) or os.path.basename(href)
                tender_links.append({
                    'url': full_url,
                    'title': text,
                    'type': 'pdf'
                })

        return tender_links

    except Exception as e:
        logger.error(f"Error extracting links: {e}")
        return []
