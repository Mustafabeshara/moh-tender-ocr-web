import { Loader2, CheckCircle2, XCircle, Clock } from 'lucide-react';
import { ExtractionJob, ExtractionStatus } from '../types';

interface ExtractionProgressProps {
  job: ExtractionJob;
}

export function ExtractionProgress({ job }: ExtractionProgressProps) {
  const getStatusIcon = (status: ExtractionStatus) => {
    switch (status) {
      case 'pending':
        return <Clock className="w-6 h-6 text-yellow-500" />;
      case 'processing':
        return <Loader2 className="w-6 h-6 text-primary-500 animate-spin" />;
      case 'completed':
        return <CheckCircle2 className="w-6 h-6 text-green-500" />;
      case 'failed':
        return <XCircle className="w-6 h-6 text-red-500" />;
    }
  };

  const getStatusText = (status: ExtractionStatus) => {
    switch (status) {
      case 'pending':
        return 'Waiting to start...';
      case 'processing':
        return `Processing ${job.processed_files} of ${job.total_files} files...`;
      case 'completed':
        return 'Extraction complete!';
      case 'failed':
        return 'Extraction failed';
    }
  };

  const getStatusColor = (status: ExtractionStatus) => {
    switch (status) {
      case 'pending':
        return 'bg-yellow-500';
      case 'processing':
        return 'bg-primary-500';
      case 'completed':
        return 'bg-green-500';
      case 'failed':
        return 'bg-red-500';
    }
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6">
      {/* Status Header */}
      <div className="flex items-center gap-3 mb-4">
        {getStatusIcon(job.status)}
        <div>
          <h3 className="font-semibold text-gray-800">
            {getStatusText(job.status)}
          </h3>
          <p className="text-sm text-gray-500">
            Job ID: {job.job_id.slice(0, 8)}...
          </p>
        </div>
      </div>

      {/* Progress Bar */}
      {job.status === 'processing' && (
        <div className="mb-4">
          <div className="flex justify-between text-sm mb-1">
            <span className="text-gray-600">Progress</span>
            <span className="text-gray-800 font-medium">{job.progress}%</span>
          </div>
          <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-500 ${getStatusColor(job.status)}`}
              style={{ width: `${job.progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Stats */}
      {job.status === 'completed' && (
        <div className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t border-gray-100">
          <div className="text-center">
            <p className="text-2xl font-bold text-primary-600">{job.results.length}</p>
            <p className="text-xs text-gray-500">Tenders</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-primary-600">
              {job.results.reduce((sum, t) => sum + t.items_count, 0)}
            </p>
            <p className="text-xs text-gray-500">Items</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-primary-600">
              {job.results.length > 0
                ? (job.results.reduce((sum, t) => sum + t.ocr_confidence, 0) / job.results.length).toFixed(0)
                : 0}%
            </p>
            <p className="text-xs text-gray-500">Confidence</p>
          </div>
        </div>
      )}

      {/* Errors */}
      {job.errors.length > 0 && (
        <div className="mt-4 p-3 bg-red-50 rounded-lg">
          <p className="text-sm font-medium text-red-700 mb-2">Errors:</p>
          <ul className="text-sm text-red-600 space-y-1">
            {job.errors.map((error, i) => (
              <li key={i}>• {error}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
