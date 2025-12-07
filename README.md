# MOH Tender OCR Web Application

A web application for extracting structured data from Ministry of Health tender PDFs using OCR (Optical Character Recognition).

## Features

- **PDF Upload**: Drag-and-drop or click to upload tender PDFs
- **OCR Processing**: Extract text using Tesseract with Arabic and English support
- **Data Extraction**: Automatically parse reference numbers, dates, items, and specifications
- **Export**: Download results as JSON or CSV
- **Real-time Progress**: Track extraction status with progress indicators
- **Configurable**: Adjust OCR settings (DPI, languages, max pages)

## Quick Start with Docker

### Prerequisites
- Docker
- Docker Compose

### Run the Application

```bash
# Clone or navigate to the project directory
cd moh-tender-ocr-web

# Build and start containers
docker-compose up --build

# Access the application
open http://localhost:3000
```

### Stop the Application

```bash
docker-compose down
```

## Local Development

### Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install system dependencies (macOS)
brew install tesseract tesseract-lang poppler

# Run development server
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/upload` | Upload PDF files |
| POST | `/api/extract/{job_id}` | Start extraction |
| GET | `/api/job/{job_id}` | Get job status |
| GET | `/api/export/{job_id}` | Export results |

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| Languages | eng, ara | OCR languages |
| DPI | 300 | Image resolution |
| Max Pages | 10 | Pages to process |
| Department | Biomedical Engineering | Default department |

## Project Structure

```
moh-tender-ocr-web/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI application
│   │   ├── models.py        # Pydantic schemas
│   │   └── ocr_engine.py    # OCR processing logic
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── hooks/           # API hooks
│   │   ├── types/           # TypeScript types
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── Dockerfile
│   ├── nginx.conf
│   └── package.json
├── docker-compose.yml
└── README.md
```

## Extracted Data Fields

- **Reference Number**: Tender ID (e.g., 5TN2024)
- **Closing Date**: Submission deadline
- **Posting Date**: Publication date
- **Items**: List of tender items with quantities
- **Specifications**: Technical requirements
- **Language**: Detected language (English/Arabic/Mixed)
- **OCR Confidence**: Extraction accuracy percentage

## Troubleshooting

### OCR not working
- Ensure Tesseract is installed with Arabic language pack
- Check that Poppler is installed for PDF conversion

### Low confidence scores
- Try increasing DPI setting (300-400 recommended)
- Ensure PDFs are not password-protected
- Check image quality in source documents

### Upload failures
- Maximum file size is 50MB per upload
- Only PDF files are accepted
