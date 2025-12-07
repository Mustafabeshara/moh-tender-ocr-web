import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useSearchParams, Link } from 'react-router-dom';
import { Search, Filter, Play, Trash2, Eye, RefreshCw } from 'lucide-react';
import { fetchTenders, processTender, deleteTender } from '../services/api';
import toast from 'react-hot-toast';

const statusColors: Record<string, string> = {
  pending: 'badge-pending',
  processing: 'badge-processing',
  completed: 'badge-completed',
  failed: 'badge-failed',
};

export default function TenderList() {
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const [search, setSearch] = useState(searchParams.get('search') || '');
  
  const status = searchParams.get('status') || '';
  const department = searchParams.get('department') || '';

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['tenders', status, department, search],
    queryFn: () => fetchTenders({ status: status || undefined, department: department || undefined, search: search || undefined }),
    refetchInterval: 5000,
  });

  const processM = useMutation({
    mutationFn: processTender,
    onSuccess: () => {
      toast.success('OCR processing started');
      queryClient.invalidateQueries({ queryKey: ['tenders'] });
    },
    onError: () => toast.error('Failed to start processing'),
  });

  const deleteM = useMutation({
    mutationFn: deleteTender,
    onSuccess: () => {
      toast.success('Tender deleted');
      queryClient.invalidateQueries({ queryKey: ['tenders'] });
      queryClient.invalidateQueries({ queryKey: ['stats'] });
    },
    onError: () => toast.error('Failed to delete tender'),
  });

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setSearchParams(prev => {
      if (search) prev.set('search', search);
      else prev.delete('search');
      return prev;
    });
  };

  const handleStatusFilter = (newStatus: string) => {
    setSearchParams(prev => {
      if (newStatus) prev.set('status', newStatus);
      else prev.delete('status');
      return prev;
    });
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-900">Tenders</h2>
        <button onClick={() => refetch()} className="btn-secondary flex items-center gap-2">
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      {/* Filters */}
      <div className="card">
        <div className="flex flex-wrap gap-4">
          <form onSubmit={handleSearch} className="flex-1 min-w-[300px]">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search by reference or title..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="input pl-10"
              />
            </div>
          </form>
          
          <div className="flex gap-2">
            <button
              onClick={() => handleStatusFilter('')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                !status ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              All
            </button>
            {['pending', 'processing', 'completed', 'failed'].map((s) => (
              <button
                key={s}
                onClick={() => handleStatusFilter(s)}
                className={`px-4 py-2 rounded-lg font-medium capitalize transition-colors ${
                  status === s ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Results */}
      {isLoading ? (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        </div>
      ) : data?.tenders.length === 0 ? (
        <div className="card text-center py-12">
          <p className="text-gray-500">No tenders found</p>
          <Link to="/upload" className="btn-primary inline-block mt-4">
            Upload PDF
          </Link>
        </div>
      ) : (
        <div className="card overflow-hidden p-0">
          <table className="w-full">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Reference
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Department
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Items
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Confidence
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {data?.tenders.map((tender) => (
                <tr key={tender.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <Link
                      to={`/tenders/${tender.id}`}
                      className="text-blue-600 hover:text-blue-800 font-medium"
                    >
                      {tender.reference || tender.id.slice(0, 12)}
                    </Link>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {tender.department || '-'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {tender.items?.length || 0}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">
                    {tender.ocr_confidence > 0 ? (
                      <span className={tender.ocr_confidence >= 80 ? 'text-green-600' : 'text-yellow-600'}>
                        {tender.ocr_confidence.toFixed(1)}%
                      </span>
                    ) : (
                      <span className="text-gray-400">-</span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`badge ${statusColors[tender.status]}`}>
                      {tender.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right">
                    <div className="flex items-center justify-end gap-2">
                      <Link
                        to={`/tenders/${tender.id}`}
                        className="p-2 text-gray-500 hover:text-blue-600 hover:bg-blue-50 rounded-lg"
                        title="View Details"
                      >
                        <Eye className="w-4 h-4" />
                      </Link>
                      {tender.status === 'pending' && (
                        <button
                          onClick={() => processM.mutate(tender.id)}
                          disabled={processM.isPending}
                          className="p-2 text-gray-500 hover:text-green-600 hover:bg-green-50 rounded-lg"
                          title="Start OCR"
                        >
                          <Play className="w-4 h-4" />
                        </button>
                      )}
                      <button
                        onClick={() => {
                          if (confirm('Delete this tender?')) {
                            deleteM.mutate(tender.id);
                          }
                        }}
                        className="p-2 text-gray-500 hover:text-red-600 hover:bg-red-50 rounded-lg"
                        title="Delete"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          
          {data && data.total > data.limit && (
            <div className="px-6 py-4 bg-gray-50 border-t border-gray-200">
              <p className="text-sm text-gray-500">
                Showing {data.tenders.length} of {data.total} tenders
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
