import axios from 'axios';
import {
  TenderData,
  ExtractionJob,
  FileUploadResponse,
  ExtractionSummary,
  HealthResponse,
  ExtractionConfig
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const apiService = {
  // Health check
  async checkHealth(): Promise<HealthResponse> {
    const { data } = await api.get('/health');
    return data;
  },

  // Upload files
  async uploadFiles(files: File[]): Promise<FileUploadResponse> {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    const { data } = await api.post('/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return data;
  },

  // Start extraction
  async startExtraction(jobId: string, config: ExtractionConfig): Promise<ExtractionJob> {
    const { data } = await api.post(`/extract/${jobId}`, config);
    return data;
  },

  // Get job status
  async getJobStatus(jobId: string): Promise<ExtractionJob> {
    const { data } = await api.get(`/jobs/${jobId}`);
    return data;
  },

  // Get all jobs
  async getJobs(status?: string): Promise<ExtractionJob[]> {
    const params = status ? { status } : {};
    const { data } = await api.get('/jobs', { params });
    return data;
  },

  // Delete job
  async deleteJob(jobId: string): Promise<void> {
    await api.delete(`/jobs/${jobId}`);
  },

  // Get results
  async getResults(jobId: string): Promise<TenderData[]> {
    const { data } = await api.get(`/results/${jobId}`);
    return data;
  },

  // Get summary
  async getSummary(jobId: string): Promise<ExtractionSummary> {
    const { data } = await api.get(`/results/${jobId}/summary`);
    return data;
  },

  // Export results
  async exportResults(jobId: string, format: 'json' | 'csv'): Promise<Blob> {
    const { data } = await api.get(`/results/${jobId}/export`, {
      params: { format },
      responseType: 'blob',
    });
    return data;
  },

  // Quick single file extraction
  async extractSingleFile(
    file: File,
    config: Partial<ExtractionConfig>
  ): Promise<TenderData> {
    const formData = new FormData();
    formData.append('file', file);

    const params = new URLSearchParams();
    if (config.languages) params.append('languages', config.languages.join('+'));
    if (config.dpi) params.append('dpi', config.dpi.toString());
    if (config.max_pages) params.append('max_pages', config.max_pages.toString());
    if (config.department) params.append('department', config.department);

    const { data } = await api.post(`/extract-single?${params}`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return data;
  },
};

export default apiService;
