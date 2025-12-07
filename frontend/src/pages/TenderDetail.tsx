import { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { ArrowLeft, Play, Save, Trash2, Plus, FileText, Download, Edit2 } from 'lucide-react';
import { fetchTender, updateTender, processTender, getProcessingStatus, deleteTender, deleteItem, getPDFUrl } from '../services/api';
import type { Tender, TenderItem } from '../services/api';
import toast from 'react-hot-toast';

const statusColors: Record<string, string> = {
  pending: 'badge-pending',
  processing: 'badge-processing',
  completed: 'badge-completed',
  failed: 'badge-failed',
};

export default function TenderDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  
  const [isEditing, setIsEditing] = useState(false);
  const [editData, setEditData] = useState<Partial<Tender>>({});
  const [processingProgress, setProcessingProgress] = useState<{ progress: number; message: string } | null>(null);

  const { data: tender, isLoading, refetch } = useQuery({
    queryKey: ['tender', id],
    queryFn: () => fetchTender(id!),
    enabled: !!id,
  });

  // Poll for processing status
  useEffect(() => {
    if (tender?.status === 'processing') {
      const interval = setInterval(async () => {
        try {
          const status = await getProcessingStatus(id!);
          setProcessingProgress({ progress: status.progress, message: status.message });
          if (status.status !== 'processing') {
            clearInterval(interval);
            setProcessingProgress(null);
            refetch();
          }
        } catch {
          clearInterval(interval);
        }
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [tender?.status, id, refetch]);

  const updateM = useMutation({
    mutationFn: (data: Partial<Tender>) => updateTender(id!, data),
    onSuccess: () => {
      toast.success('Tender updated');
      setIsEditing(false);
      queryClient.invalidateQueries({ queryKey: ['tender', id] });
    },
    onError: () => toast.error('Failed to update'),
  });

  const processM = useMutation({
    mutationFn: () => processTender(id!),
    onSuccess: () => {
      toast.success('OCR processing started');
      refetch();
    },
    onError: () => toast.error('Failed to start processing'),
  });

  const deleteM = useMutation({
    mutationFn: () => deleteTender(id!),
    onSuccess: () => {
      toast.success('Tender deleted');
      navigate('/tenders');
    },
    onError: () => toast.error('Failed to delete'),
  });

  const deleteItemM = useMutation({
    mutationFn: (index: number) => deleteItem(id!, index),
    onSuccess: () => {
      toast.success('Item deleted');
      queryClient.invalidateQueries({ queryKey: ['tender', id] });
    },
    onError: () => toast.error('Failed to delete item'),
  });

  const handleSave = () => {
    updateM.mutate(editData);
  };

  const startEditing = () => {
    setEditData({
      reference: tender?.reference,
      department: tender?.department,
      title: tender?.title,
      closing_date: tender?.closing_date,
      specifications: tender?.specifications,
    });
    setIsEditing(true);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!tender) {
    return (
      <div className="card text-center py-12">
        <p className="text-gray-500">Tender not found</p>
        <Link to="/tenders" className="btn-primary inline-block mt-4">
          Back to List
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link to="/tenders" className="p-2 hover:bg-gray-100 rounded-lg">
            <ArrowLeft className="w-5 h-5" />
          </Link>
          <div>
            <h2 className="text-2xl font-bold text-gray-900">{tender.reference || 'Tender Details'}</h2>
            <p className="text-sm text-gray-500">ID: {tender.id}</p>
          </div>
          <span className={`badge ${statusColors[tender.status]}`}>{tender.status}</span>
        </div>
        
        <div className="flex gap-2">
          {tender.source_file && (
            <a
              href={getPDFUrl(tender.id)}
              target="_blank"
              rel="noopener noreferrer"
              className="btn-secondary flex items-center gap-2"
            >
              <FileText className="w-4 h-4" />
              View PDF
            </a>
          )}
          {tender.status === 'pending' && (
            <button onClick={() => processM.mutate()} className="btn-primary flex items-center gap-2">
              <Play className="w-4 h-4" />
              Run OCR
            </button>
          )}
          {!isEditing ? (
            <button onClick={startEditing} className="btn-secondary flex items-center gap-2">
              <Edit2 className="w-4 h-4" />
              Edit
            </button>
          ) : (
            <button onClick={handleSave} disabled={updateM.isPending} className="btn-success flex items-center gap-2">
              <Save className="w-4 h-4" />
              Save
            </button>
          )}
          <button
            onClick={() => {
              if (confirm('Delete this tender?')) deleteM.mutate();
            }}
            className="btn-danger flex items-center gap-2"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Processing Progress */}
      {tender.status === 'processing' && processingProgress && (
        <div className="card bg-blue-50 border-blue-200">
          <div className="flex items-center gap-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <div className="flex-1">
              <p className="font-medium text-blue-900">Processing OCR...</p>
              <p className="text-sm text-blue-700">{processingProgress.message}</p>
              <div className="mt-2 w-full bg-blue-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all"
                  style={{ width: `${processingProgress.progress}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tender Info */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Tender Information</h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-500 mb-1">Reference</label>
              {isEditing ? (
                <input
                  type="text"
                  value={editData.reference || ''}
                  onChange={(e) => setEditData({ ...editData, reference: e.target.value })}
                  className="input"
                />
              ) : (
                <p className="text-gray-900">{tender.reference || '-'}</p>
              )}
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-500 mb-1">Department</label>
              {isEditing ? (
                <select
                  value={editData.department || ''}
                  onChange={(e) => setEditData({ ...editData, department: e.target.value })}
                  className="input"
                >
                  <option value="">Select Department</option>
                  <option value="Medical Store">Medical Store</option>
                  <option value="Biomedical Engineering">Biomedical Engineering</option>
                </select>
              ) : (
                <p className="text-gray-900">{tender.department || '-'}</p>
              )}
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-500 mb-1">Title</label>
              {isEditing ? (
                <input
                  type="text"
                  value={editData.title || ''}
                  onChange={(e) => setEditData({ ...editData, title: e.target.value })}
                  className="input"
                />
              ) : (
                <p className="text-gray-900">{tender.title || '-'}</p>
              )}
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-500 mb-1">Closing Date</label>
              {isEditing ? (
                <input
                  type="text"
                  value={editData.closing_date || ''}
                  onChange={(e) => setEditData({ ...editData, closing_date: e.target.value })}
                  className="input"
                  placeholder="e.g., 2024-12-31"
                />
              ) : (
                <p className="text-gray-900">{tender.closing_date || '-'}</p>
              )}
            </div>
          </div>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">OCR Results</h3>
          
          <div className="space-y-4">
            <div className="flex justify-between">
              <span className="text-gray-500">OCR Confidence</span>
              <span className={`font-medium ${tender.ocr_confidence >= 80 ? 'text-green-600' : 'text-yellow-600'}`}>
                {tender.ocr_confidence > 0 ? `${tender.ocr_confidence.toFixed(1)}%` : '-'}
              </span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-gray-500">Items Extracted</span>
              <span className="font-medium">{tender.items?.length || 0}</span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-gray-500">Has Arabic Content</span>
              <span className="font-medium">{tender.has_arabic_content ? 'Yes' : 'No'}</span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-gray-500">Created</span>
              <span className="font-medium">{new Date(tender.created_at).toLocaleString()}</span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-gray-500">Updated</span>
              <span className="font-medium">{new Date(tender.updated_at).toLocaleString()}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Specifications */}
      {(tender.specifications || isEditing) && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Specifications</h3>
          {isEditing ? (
            <textarea
              value={editData.specifications || ''}
              onChange={(e) => setEditData({ ...editData, specifications: e.target.value })}
              className="input h-32"
              placeholder="Tender specifications..."
            />
          ) : (
            <p className="text-gray-700 whitespace-pre-wrap">{tender.specifications || 'No specifications extracted'}</p>
          )}
        </div>
      )}

      {/* Items */}
      <div className="card">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Items ({tender.items?.length || 0})</h3>
        </div>

        {tender.items && tender.items.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">#</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Description</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Quantity</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Unit</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Language</th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {tender.items.map((item, index) => (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="px-4 py-3 text-sm text-gray-500">{item.item_number || index + 1}</td>
                    <td className="px-4 py-3 text-sm text-gray-900 max-w-md">
                      <div className="truncate" title={item.description}>
                        {item.description}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-500">{item.quantity || '-'}</td>
                    <td className="px-4 py-3 text-sm text-gray-500">{item.unit || '-'}</td>
                    <td className="px-4 py-3 text-sm">
                      {item.has_arabic && <span className="badge bg-purple-100 text-purple-800">Arabic</span>}
                      {!item.has_arabic && item.language && <span className="text-gray-500">{item.language}</span>}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <button
                        onClick={() => {
                          if (confirm('Delete this item?')) deleteItemM.mutate(index);
                        }}
                        className="p-1 text-gray-400 hover:text-red-600"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <p>No items extracted yet</p>
            {tender.status === 'pending' && (
              <button onClick={() => processM.mutate()} className="btn-primary mt-4">
                Run OCR to Extract Items
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
