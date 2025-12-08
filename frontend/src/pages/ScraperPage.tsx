import React, { useState, useEffect } from 'react';

interface ScraperStatus {
  running: boolean;
  progress: number;
  message: string;
  pid: number | null;
  last_run: string | null;
  tenders_found: number;
  script_exists: boolean;
  python_exists: boolean;
  tenders_dir: string;
  results_dir: string;
  result_files: number;
}

interface ResultFile {
  name: string;
  path: string;
  size: number;
  modified: string;
}

interface PdfFolder {
  name: string;
  path: string;
  pdf_count: number;
}

const API_BASE = 'http://localhost:8000/api';

export default function ScraperPage() {
  const [status, setStatus] = useState<ScraperStatus | null>(null);
  const [resultFiles, setResultFiles] = useState<ResultFile[]>([]);
  const [pdfFolders, setPdfFolders] = useState<PdfFolder[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pages, setPages] = useState(5);
  const [headless, setHeadless] = useState(true);
  const [selectedFolder, setSelectedFolder] = useState<string>('');

  // Fetch status on mount and every 3 seconds when running
  useEffect(() => {
    fetchStatus();
    fetchResults();
    fetchPdfFolders();

    const interval = setInterval(() => {
      fetchStatus();
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/scraper/status`);
      const data = await res.json();
      setStatus(data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch scraper status');
    }
  };

  const fetchResults = async () => {
    try {
      const res = await fetch(`${API_BASE}/scraper/results`);
      const data = await res.json();
      // Map backend response to frontend format
      const files = (data.results || []).map((r: any) => ({
        name: r.filename,
        path: r.filename,
        size: r.size,
        modified: r.modified
      }));
      setResultFiles(files);
    } catch (err) {
      console.error('Failed to fetch results:', err);
    }
  };

  const fetchPdfFolders = async () => {
    try {
      const res = await fetch(`${API_BASE}/scraper/pdfs`);
      const data = await res.json();
      setPdfFolders(data.folders || []);
    } catch (err) {
      console.error('Failed to fetch PDF folders:', err);
    }
  };

  const startScraper = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/scraper/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pages, headless })
      });
      const data = await res.json();
      if (data.success) {
        setError(null);
        fetchStatus();
      } else {
        setError(data.message || 'Failed to start scraper');
      }
    } catch (err) {
      setError('Failed to start scraper');
    }
    setLoading(false);
  };

  const startBatchOcr = async () => {
    if (!selectedFolder) {
      setError('Please select a PDF folder first');
      return;
    }
    setLoading(true);
    try {
      const params = new URLSearchParams({ folder: selectedFolder });
      const res = await fetch(`${API_BASE}/scraper/batch-ocr?${params}`, {
        method: 'POST'
      });
      const data = await res.json();
      if (data.message) {
        setError(null);
        fetchStatus();
      } else if (data.detail) {
        setError(data.detail);
      } else {
        setError('Failed to start batch OCR');
      }
    } catch (err) {
      setError('Failed to start batch OCR');
    }
    setLoading(false);
  };

  const stopScraper = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/scraper/stop`, { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        setError(null);
        fetchStatus();
      } else {
        setError(data.message || 'Failed to stop scraper');
      }
    } catch (err) {
      setError('Failed to stop scraper');
    }
    setLoading(false);
  };

  const downloadFile = (filename: string) => {
    window.open(`${API_BASE}/scraper/download/${filename}`, '_blank');
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleString();
  };

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold text-gray-900 mb-6">MOH Tender Scraper</h1>

      {/* Status Card */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">Scraper Status</h2>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}

        {status && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-50 p-4 rounded">
              <p className="text-sm text-gray-500">Status</p>
              <p className={`text-lg font-semibold ${status.running ? 'text-green-600' : 'text-gray-600'}`}>
                {status.running ? 'Running' : 'Idle'}
              </p>
            </div>
            <div className="bg-gray-50 p-4 rounded">
              <p className="text-sm text-gray-500">Progress</p>
              <p className="text-lg font-semibold">{status.progress.toFixed(1)}%</p>
            </div>
            <div className="bg-gray-50 p-4 rounded">
              <p className="text-sm text-gray-500">Message</p>
              <p className="text-lg font-semibold truncate" title={status.message}>
                {status.message}
              </p>
            </div>
            <div className="bg-gray-50 p-4 rounded">
              <p className="text-sm text-gray-500">Result Files</p>
              <p className="text-lg font-semibold">{status.result_files}</p>
            </div>
          </div>
        )}

        {status?.running && (
          <div className="mt-4">
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div
                className="bg-blue-600 h-2.5 rounded-full transition-all duration-500"
                style={{ width: `${status.progress}%` }}
              ></div>
            </div>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">Controls</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Scrape from MOH Website */}
          <div className="border rounded-lg p-4">
            <h3 className="font-semibold mb-3">Scrape from MOH Website</h3>
            <div className="space-y-3">
              <div>
                <label className="block text-sm text-gray-600 mb-1">Number of Pages</label>
                <input
                  type="number"
                  value={pages}
                  onChange={(e) => setPages(parseInt(e.target.value) || 1)}
                  min={1}
                  max={50}
                  className="w-full border rounded px-3 py-2"
                  disabled={status?.running}
                />
              </div>
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="headless"
                  checked={headless}
                  onChange={(e) => setHeadless(e.target.checked)}
                  className="mr-2"
                  disabled={status?.running}
                />
                <label htmlFor="headless" className="text-sm text-gray-600">
                  Headless mode (no browser window)
                </label>
              </div>
              <button
                onClick={startScraper}
                disabled={loading || status?.running}
                className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                {status?.running ? 'Scraper Running...' : 'Start Scraper'}
              </button>
            </div>
          </div>

          {/* Batch OCR Processing */}
          <div className="border rounded-lg p-4">
            <h3 className="font-semibold mb-3">Batch OCR Processing</h3>
            <div className="space-y-3">
              <div>
                <label className="block text-sm text-gray-600 mb-1">PDF Folder</label>
                <select
                  value={selectedFolder}
                  onChange={(e) => setSelectedFolder(e.target.value)}
                  className="w-full border rounded px-3 py-2"
                  disabled={status?.running}
                >
                  <option value="">Select a folder...</option>
                  {pdfFolders.map((folder) => (
                    <option key={folder.path} value={folder.path}>
                      {folder.name} ({folder.pdf_count} PDFs)
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm text-gray-600 mb-1">Pages per PDF</label>
                <input
                  type="number"
                  value={pages}
                  onChange={(e) => setPages(parseInt(e.target.value) || 1)}
                  min={1}
                  max={20}
                  className="w-full border rounded px-3 py-2"
                  disabled={status?.running}
                />
              </div>
              <button
                onClick={startBatchOcr}
                disabled={loading || status?.running || !selectedFolder}
                className="w-full bg-green-600 text-white py-2 px-4 rounded hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                {status?.running ? 'Processing...' : 'Start Batch OCR'}
              </button>
            </div>
          </div>
        </div>

        {status?.running && (
          <div className="mt-4">
            <button
              onClick={stopScraper}
              disabled={loading}
              className="bg-red-600 text-white py-2 px-6 rounded hover:bg-red-700"
            >
              Stop Scraper
            </button>
          </div>
        )}
      </div>

      {/* Results */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">OCR Results</h2>
          <button
            onClick={fetchResults}
            className="text-blue-600 hover:text-blue-800 text-sm"
          >
            Refresh
          </button>
        </div>

        {resultFiles.length === 0 ? (
          <p className="text-gray-500">No result files found</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    File Name
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Size
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Modified
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {resultFiles.slice(0, 20).map((file) => (
                  <tr key={file.name} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="text-sm text-gray-900">{file.name}</span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatBytes(file.size)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatDate(file.modified)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <button
                        onClick={() => downloadFile(file.name)}
                        className="text-blue-600 hover:text-blue-800 text-sm"
                      >
                        Download
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {resultFiles.length > 20 && (
              <p className="text-sm text-gray-500 mt-2 text-center">
                Showing 20 of {resultFiles.length} files
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
