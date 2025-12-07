export interface TenderItem {
  item_number: string;
  description: string;
  quantity: string;
  unit: string;
  specifications: string;
  language: string;
  has_arabic: boolean;
}

export interface TenderData {
  id?: string;
  reference_number: string;
  title: string;
  closing_date: string;
  posting_date: string;
  department: string;
  items: TenderItem[];
  specifications_text: string;
  items_count: number;
  ocr_confidence: number;
  source_files: string[];
  extraction_timestamp: string;
  extraction_method: string;
  language: string;
  has_arabic_content: boolean;
  errors: string[];
}

export type ExtractionStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface ExtractionJob {
  job_id: string;
  status: ExtractionStatus;
  progress: number;
  total_files: number;
  processed_files: number;
  results: TenderData[];
  errors: string[];
  created_at: string;
  completed_at?: string;
}

export interface FileUploadResponse {
  job_id: string;
  message: string;
  files_received: number;
  status: ExtractionStatus;
}

export interface ExtractionSummary {
  total_tenders: number;
  total_items: number;
  arabic_tenders: number;
  tenders_with_specs: number;
  average_confidence: number;
  extraction_date: string;
}

export interface HealthResponse {
  status: string;
  tesseract_available: boolean;
  poppler_available: boolean;
  version: string;
}

export interface ExtractionConfig {
  languages: string[];
  dpi: number;
  max_pages: number;
  department: string;
}
