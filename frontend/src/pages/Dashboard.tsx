import { useQuery } from '@tanstack/react-query';
import { FileText, Clock, CheckCircle, XCircle, Package, TrendingUp, Download, Play } from 'lucide-react';
import { fetchStats, batchProcess, exportTenders } from '../services/api';
import toast from 'react-hot-toast';

export default function Dashboard() {
  const { data: stats, isLoading, refetch } = useQuery({
    queryKey: ['stats'],
    queryFn: fetchStats,
    refetchInterval: 5000,
  });

  const handleBatchProcess = async () => {
    try {
      const result = await batchProcess('pending');
      toast.success(result.message);
      refetch();
    } catch (error) {
      toast.error('Failed to start batch processing');
    }
  };

  const handleExport = async () => {
    try {
      const blob = await exportTenders({ status: 'completed' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'tenders_export.xlsx';
      a.click();
      URL.revokeObjectURL(url);
      toast.success('Exported to Excel');
    } catch (error) {
      toast.error('Export failed');
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  const statCards = [
    {
      label: 'Total Tenders',
      value: stats?.total_tenders || 0,
      icon: FileText,
      color: 'bg-blue-500',
    },
    {
      label: 'Pending',
      value: stats?.pending || 0,
      icon: Clock,
      color: 'bg-yellow-500',
    },
    {
      label: 'Processing',
      value: stats?.processing || 0,
      icon: TrendingUp,
      color: 'bg-purple-500',
    },
    {
      label: 'Completed',
      value: stats?.completed || 0,
      icon: CheckCircle,
      color: 'bg-green-500',
    },
    {
      label: 'Failed',
      value: stats?.failed || 0,
      icon: XCircle,
      color: 'bg-red-500',
    },
    {
      label: 'Total Items',
      value: stats?.total_items || 0,
      icon: Package,
      color: 'bg-indigo-500',
    },
  ];

  return (
    <div className="space-y-8">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-900">Dashboard</h2>
        <div className="flex gap-3">
          <button
            onClick={handleBatchProcess}
            disabled={!stats?.pending}
            className="btn-primary flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Play className="w-4 h-4" />
            Process All Pending ({stats?.pending || 0})
          </button>
          <button
            onClick={handleExport}
            disabled={!stats?.completed}
            className="btn-secondary flex items-center gap-2 disabled:opacity-50"
          >
            <Download className="w-4 h-4" />
            Export Excel
          </button>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {statCards.map((stat) => (
          <div key={stat.label} className="card flex items-center gap-4">
            <div className={`${stat.color} p-3 rounded-lg`}>
              <stat.icon className="w-6 h-6 text-white" />
            </div>
            <div>
              <p className="text-sm text-gray-500">{stat.label}</p>
              <p className="text-2xl font-bold text-gray-900">{stat.value}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Average Confidence */}
      {stats && stats.avg_confidence > 0 && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">OCR Quality</h3>
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-600">Average OCR Confidence</span>
                <span className="font-medium">{stats.avg_confidence.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className={`h-3 rounded-full ${
                    stats.avg_confidence >= 80
                      ? 'bg-green-500'
                      : stats.avg_confidence >= 60
                      ? 'bg-yellow-500'
                      : 'bg-red-500'
                  }`}
                  style={{ width: `${stats.avg_confidence}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Quick Actions */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <a
            href="/upload"
            className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <div className="bg-blue-100 p-2 rounded-lg">
              <FileText className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <p className="font-medium text-gray-900">Upload PDFs</p>
              <p className="text-sm text-gray-500">Add new tender documents</p>
            </div>
          </a>
          
          <a
            href="/tenders?status=pending"
            className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <div className="bg-yellow-100 p-2 rounded-lg">
              <Clock className="w-5 h-5 text-yellow-600" />
            </div>
            <div>
              <p className="font-medium text-gray-900">View Pending</p>
              <p className="text-sm text-gray-500">{stats?.pending || 0} tenders waiting</p>
            </div>
          </a>
          
          <a
            href="/tenders?status=completed"
            className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <div className="bg-green-100 p-2 rounded-lg">
              <CheckCircle className="w-5 h-5 text-green-600" />
            </div>
            <div>
              <p className="font-medium text-gray-900">View Completed</p>
              <p className="text-sm text-gray-500">{stats?.completed || 0} processed</p>
            </div>
          </a>
        </div>
      </div>
    </div>
  );
}
