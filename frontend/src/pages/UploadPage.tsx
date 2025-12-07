import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { uploadPDF } from '../services/api';

interface UploadedFile {
  file: File;
  status: 'pending' | 'uploading' | 'success' | 'error';
  progress: number;
  tenderId?: string;
  error?: string;
}

export default function UploadPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [processAfterUpload, setProcessAfterUpload] = useState(true);

  const uploadMutation = useMutation({
    mutationFn: async (uploadedFile: UploadedFile) => {
      return uploadPDF(uploadedFile.file, processAfterUpload);
    },
    onSuccess: (data, variables) => {
      setFiles(prev => prev.map(f => 
        f.file === variables.file 
          ? { ...f, status: 'success' as const, progress: 100, tenderId: data.tender_id }
          : f
      ));
      queryClient.invalidateQueries({ queryKey: ['tenders'] });
      queryClient.invalidateQueries({ queryKey: ['stats'] });
    },
    onError: (error: Error, variables) => {
      setFiles(prev => prev.map(f => 
        f.file === variables.file 
          ? { ...f, status: 'error' as const, error: error.message }
          : f
      ));
    }
  });

  const handleFiles = useCallback((newFiles: FileList | File[]) => {
    const pdfFiles = Array.from(newFiles).filter(file => 
      file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')
    );

    const uploadedFiles: UploadedFile[] = pdfFiles.map(file => ({
      file,
      status: 'pending' as const,
      progress: 0
    }));

    setFiles(prev => [...prev, ...uploadedFiles]);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  }, [handleFiles]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      handleFiles(e.target.files);
    }
  }, [handleFiles]);

  const uploadAll = async () => {
    const pendingFiles = files.filter(f => f.status === 'pending');
    
    for (const file of pendingFiles) {
      setFiles(prev => prev.map(f => 
        f.file === file.file ? { ...f, status: 'uploading' as const } : f
      ));
      
      await uploadMutation.mutateAsync(file);
    }
  };

  const removeFile = (fileToRemove: File) => {
    setFiles(prev => prev.filter(f => f.file !== fileToRemove));
  };

  const clearCompleted = () => {
    setFiles(prev => prev.filter(f => f.status !== 'success'));
  };

  const pendingCount = files.filter(f => f.status === 'pending').length;
  const uploadingCount = files.filter(f => f.status === 'uploading').length;
  const successCount = files.filter(f => f.status === 'success').length;
  const errorCount = files.filter(f => f.status === 'error').length;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900">Upload Tender PDFs</h1>
        <button
          onClick={() => navigate('/tenders')}
          className="text-gray-600 hover:text-gray-900"
        >
          View All Tenders →
        </button>
      </div>

      {/* Upload Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-xl p-12 text-center transition-colors ${
          isDragging 
            ? 'border-blue-500 bg-blue-50' 
            : 'border-gray-300 hover:border-gray-400'
        }`}
      >
        <div className="space-y-4">
          <div className="text-6xl">📄</div>
          <div>
            <p className="text-xl font-medium text-gray-700">
              Drag & drop PDF files here
            </p>
            <p className="text-gray-500 mt-1">or click to browse</p>
          </div>
          <input
            type="file"
            accept=".pdf,application/pdf"
            multiple
            onChange={handleFileInput}
            className="hidden"
            id="file-input"
          />
          <label
            htmlFor="file-input"
            className="inline-block px-6 py-3 bg-blue-600 text-white rounded-lg cursor-pointer hover:bg-blue-700 transition-colors"
          >
            Select Files
          </label>
        </div>
      </div>

      {/* Options */}
      <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-lg">
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={processAfterUpload}
            onChange={(e) => setProcessAfterUpload(e.target.checked)}
            className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
          />
          <span className="text-gray-700">Process with OCR after upload</span>
        </label>
      </div>

      {/* File List */}
      {files.length > 0 && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200">
          <div className="p-4 border-b border-gray-200 flex justify-between items-center">
            <div className="flex gap-4 text-sm">
              <span className="text-gray-600">
                Total: <strong>{files.length}</strong>
              </span>
              {pendingCount > 0 && (
                <span className="text-yellow-600">
                  Pending: <strong>{pendingCount}</strong>
                </span>
              )}
              {uploadingCount > 0 && (
                <span className="text-blue-600">
                  Uploading: <strong>{uploadingCount}</strong>
                </span>
              )}
              {successCount > 0 && (
                <span className="text-green-600">
                  Completed: <strong>{successCount}</strong>
                </span>
              )}
              {errorCount > 0 && (
                <span className="text-red-600">
                  Failed: <strong>{errorCount}</strong>
                </span>
              )}
            </div>
            <div className="flex gap-2">
              {successCount > 0 && (
                <button
                  onClick={clearCompleted}
                  className="px-3 py-1 text-sm text-gray-600 hover:text-gray-900"
                >
                  Clear Completed
                </button>
              )}
              {pendingCount > 0 && (
                <button
                  onClick={uploadAll}
                  disabled={uploadingCount > 0}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {uploadingCount > 0 ? 'Uploading...' : `Upload All (${pendingCount})`}
                </button>
              )}
            </div>
          </div>

          <ul className="divide-y divide-gray-200 max-h-96 overflow-y-auto">
            {files.map((uploadedFile, index) => (
              <li key={index} className="p-4 flex items-center gap-4">
                <div className="text-2xl">
                  {uploadedFile.status === 'pending' && '📄'}
                  {uploadedFile.status === 'uploading' && '⏳'}
                  {uploadedFile.status === 'success' && '✅'}
                  {uploadedFile.status === 'error' && '❌'}
                </div>
                
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-gray-900 truncate">
                    {uploadedFile.file.name}
                  </p>
                  <p className="text-sm text-gray-500">
                    {(uploadedFile.file.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                  
                  {uploadedFile.status === 'uploading' && (
                    <div className="mt-2 h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-blue-600 transition-all duration-300"
                        style={{ width: '50%' }}
                      />
                    </div>
                  )}
                  
                  {uploadedFile.status === 'error' && (
                    <p className="text-sm text-red-600 mt-1">
                      {uploadedFile.error || 'Upload failed'}
                    </p>
                  )}
                  
                  {uploadedFile.status === 'success' && uploadedFile.tenderId && (
                    <button
                      onClick={() => navigate(`/tenders/${uploadedFile.tenderId}`)}
                      className="text-sm text-blue-600 hover:underline mt-1"
                    >
                      View Tender →
                    </button>
                  )}
                </div>

                {uploadedFile.status === 'pending' && (
                  <button
                    onClick={() => removeFile(uploadedFile.file)}
                    className="text-gray-400 hover:text-red-600"
                  >
                    ✕
                  </button>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Instructions */}
      <div className="bg-blue-50 rounded-xl p-6">
        <h3 className="font-semibold text-blue-900 mb-2">Instructions</h3>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• Upload PDF files containing tender documents</li>
          <li>• OCR processing will extract tender details automatically</li>
          <li>• Supported format: PDF files only</li>
          <li>• You can upload multiple files at once</li>
          <li>• Processing may take a few minutes per file</li>
        </ul>
      </div>
    </div>
  );
}
