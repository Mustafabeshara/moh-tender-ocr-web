import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, X, Loader2 } from 'lucide-react';

interface FileUploadProps {
  onFilesSelected: (files: File[]) => void;
  isLoading?: boolean;
  maxFiles?: number;
}

export function FileUpload({ onFilesSelected, isLoading, maxFiles = 20 }: FileUploadProps) {
  const [files, setFiles] = useState<File[]>([]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const pdfFiles = acceptedFiles.filter(f => f.type === 'application/pdf');
    setFiles(prev => [...prev, ...pdfFiles].slice(0, maxFiles));
  }, [maxFiles]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    maxFiles,
    disabled: isLoading,
  });

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleUpload = () => {
    if (files.length > 0) {
      onFilesSelected(files);
    }
  };

  const clearAll = () => {
    setFiles([]);
  };

  return (
    <div className="space-y-4">
      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-xl p-8 text-center cursor-pointer
          transition-all duration-200
          ${isDragActive ? 'border-primary-500 bg-primary-50' : 'border-gray-300 hover:border-primary-400'}
          ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />
        <Upload className={`w-12 h-12 mx-auto mb-4 ${isDragActive ? 'text-primary-500' : 'text-gray-400'}`} />
        {isDragActive ? (
          <p className="text-primary-600 font-medium">Drop the PDF files here...</p>
        ) : (
          <>
            <p className="text-gray-600 font-medium">
              Drag & drop PDF files here, or click to select
            </p>
            <p className="text-gray-400 text-sm mt-2">
              Supports MOH tender PDFs (V1, V2, V3 files)
            </p>
          </>
        )}
      </div>

      {/* File List */}
      {files.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex justify-between items-center mb-3">
            <span className="text-sm font-medium text-gray-700">
              {files.length} file{files.length !== 1 ? 's' : ''} selected
            </span>
            <button
              onClick={clearAll}
              className="text-sm text-red-500 hover:text-red-700"
              disabled={isLoading}
            >
              Clear all
            </button>
          </div>

          <div className="space-y-2 max-h-48 overflow-y-auto">
            {files.map((file, index) => (
              <div
                key={`${file.name}-${index}`}
                className="flex items-center justify-between py-2 px-3 bg-gray-50 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <FileText className="w-5 h-5 text-red-500" />
                  <div>
                    <p className="text-sm font-medium text-gray-700 truncate max-w-xs">
                      {file.name}
                    </p>
                    <p className="text-xs text-gray-400">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => removeFile(index)}
                  className="p-1 hover:bg-gray-200 rounded"
                  disabled={isLoading}
                >
                  <X className="w-4 h-4 text-gray-500" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Upload Button */}
      {files.length > 0 && (
        <button
          onClick={handleUpload}
          disabled={isLoading}
          className={`
            w-full py-3 px-4 rounded-lg font-medium
            flex items-center justify-center gap-2
            transition-colors duration-200
            ${isLoading
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-primary-500 text-white hover:bg-primary-600'
            }
          `}
        >
          {isLoading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Processing...
            </>
          ) : (
            <>
              <Upload className="w-5 h-5" />
              Start OCR Extraction
            </>
          )}
        </button>
      )}
    </div>
  );
}
