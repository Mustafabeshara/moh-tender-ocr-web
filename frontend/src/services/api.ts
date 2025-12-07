import axios from 'axios';

const API_BASE = '/api';

export interface TenderItem {
  item_number: string;
  description: string;
  quantity: string;
  unit: string;
  language: string;
  has_arabic: boolean;
}

export interface Tender {
  id: string;
  reference: string;
  department: string;
  title: string;
  closing_date: string;
  items: TenderItem[];
  ocr_confidence: number;
  has_arabic_content: boolean;
  specifications: string;
  source_file: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  created_at: string;
  updated_at: string;
}

export interface Stats {
  total_tenders: number;
  pending: number;
  processing: number;
  completed: number;
  failed: number;
  total_items: number;
  avg_confidence: number;
}

export interface ProcessingStatus {
  tender_id: string;
  status: string;
  progress: number;
  message: string;
}

export interface TendersResponse {
  total: number;
  limit: number;
  offset: number;
  tenders: Tender[];
}

// Fetch all tenders
export async function fetchTenders(params?: {
  status?: string;
  department?: string;
  search?: string;
  limit?: number;
  offset?: number;
}): Promise<TendersResponse> {
  const response = await axios.get(`${API_BASE}/tenders`, { params });
  return response.data;
}

// Fetch single tender
export async function fetchTender(tenderId: string): Promise<Tender> {
  const response = await axios.get(`${API_BASE}/tenders/${tenderId}`);
  return response.data;
}

// Fetch stats
export async function fetchStats(): Promise<Stats> {
  const response = await axios.get(`${API_BASE}/stats`);
  return response.data;
}

// Upload PDF
export async function uploadPDF(file: File, department?: string): Promise<{ tender_id: string; filename: string }> {
  const formData = new FormData();
  formData.append('file', file);
  if (department) {
    formData.append('department', department);
  }
  
  const response = await axios.post(`${API_BASE}/upload`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

// Start OCR processing
export async function processTender(tenderId: string): Promise<{ message: string }> {
  const response = await axios.post(`${API_BASE}/tenders/${tenderId}/process`);
  return response.data;
}

// Get processing status
export async function getProcessingStatus(tenderId: string): Promise<ProcessingStatus> {
  const response = await axios.get(`${API_BASE}/tenders/${tenderId}/status`);
  return response.data;
}

// Update tender
export async function updateTender(tenderId: string, data: Partial<Tender>): Promise<Tender> {
  const response = await axios.put(`${API_BASE}/tenders/${tenderId}`, data);
  return response.data;
}

// Delete tender
export async function deleteTender(tenderId: string): Promise<void> {
  await axios.delete(`${API_BASE}/tenders/${tenderId}`);
}

// Add item to tender
export async function addItem(tenderId: string, item: TenderItem): Promise<Tender> {
  const response = await axios.post(`${API_BASE}/tenders/${tenderId}/items`, item);
  return response.data;
}

// Delete item from tender
export async function deleteItem(tenderId: string, itemIndex: number): Promise<Tender> {
  const response = await axios.delete(`${API_BASE}/tenders/${tenderId}/items/${itemIndex}`);
  return response.data;
}

// Export tenders
export async function exportTenders(format: 'excel' | 'json', params?: {
  status?: string;
  tender_ids?: string;
}): Promise<Blob> {
  const response = await axios.get(`${API_BASE}/export/${format}`, {
    params,
    responseType: 'blob',
  });
  return response.data;
}

// Batch process
export async function batchProcess(status: string = 'pending'): Promise<{ message: string; count: number }> {
  const response = await axios.post(`${API_BASE}/batch-process`, null, { params: { status } });
  return response.data;
}

// Get PDF URL
export function getPDFUrl(tenderId: string): string {
  return `${API_BASE}/pdf/${tenderId}`;
}
