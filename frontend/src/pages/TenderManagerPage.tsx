import React, { useState, useEffect } from 'react';
import { Search, RefreshCw, Sparkles, Download, ChevronLeft, ChevronRight, Eye } from 'lucide-react';

interface DashboardStats {
  total_tenders: number;
  total_items: number;
  unique_departments: number;
  ai_enhanced_count: number;
  avg_confidence: number;
  items_per_tender: number;
  department_breakdown: Record<string, number>;
  recent_tenders: Array<{ reference: string; title: string; items: number }>;
  top_items: Array<{ description: string; count: number }>;
  ai_stats: {
    workers: Array<{ name: string; rpm: number; requests: number; errors: number }>;
    total_rpm: number;
    available: boolean;
  };
}

interface TenderItem {
  item_number: string;
  description: string;
  quantity: string;
  unit: string;
  ai_cleaned: boolean;
}

interface Tender {
  reference: string;
  department: string;
  title: string;
  closing_date: string;
  items: TenderItem[];
  source_file: string;
  ocr_confidence: number;
  has_arabic: boolean;
  ai_enhanced: boolean;
}

interface TendersResponse {
  tenders: Tender[];
  total: number;
  page: number;
  limit: number;
  pages: number;
}

const API_BASE = 'http://localhost:8000/api/manager';

export default function TenderManagerPage() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [tenders, setTenders] = useState<Tender[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [department, setDepartment] = useState('');
  const [departments, setDepartments] = useState<string[]>([]);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalTenders, setTotalTenders] = useState(0);
  const [selectedTender, setSelectedTender] = useState<Tender | null>(null);
  const [enhancing, setEnhancing] = useState(false);
  const [activeTab, setActiveTab] = useState<'dashboard' | 'tenders' | 'search'>('dashboard');

  useEffect(() => {
    fetchDashboard();
    fetchDepartments();
  }, []);

  useEffect(() => {
    if (activeTab === 'tenders') {
      fetchTenders();
    }
  }, [page, search, department, activeTab]);

  const fetchDashboard = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/dashboard`);
      if (!res.ok) throw new Error('Failed to fetch dashboard');
      const data = await res.json();
      setStats(data);
      setError(null);
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  const fetchTenders = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        page: page.toString(),
        limit: '20'
      });
      if (search) params.append('search', search);
      if (department) params.append('department', department);

      const res = await fetch(`${API_BASE}/tenders?${params}`);
      if (!res.ok) throw new Error('Failed to fetch tenders');
      const data: TendersResponse = await res.json();
      setTenders(data.tenders);
      setTotalPages(data.pages);
      setTotalTenders(data.total);
      setError(null);
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  const fetchDepartments = async () => {
    try {
      const res = await fetch(`${API_BASE}/departments`);
      if (res.ok) {
        const data = await res.json();
        setDepartments(data.departments || []);
      }
    } catch (err) {
      console.error('Failed to fetch departments:', err);
    }
  };

  const enhanceTender = async (reference: string) => {
    setEnhancing(true);
    try {
      const res = await fetch(`${API_BASE}/enhance/${reference}`, { method: 'POST' });
      if (!res.ok) throw new Error('Enhancement failed');
      const data = await res.json();
      // Update the tender in the list
      setTenders(prev => prev.map(t => t.reference === reference ? data : t));
      if (selectedTender?.reference === reference) {
        setSelectedTender(data);
      }
      setError(null);
    } catch (err: any) {
      setError(err.message);
    }
    setEnhancing(false);
  };

  const enhanceBatch = async (limit: number = 10) => {
    setEnhancing(true);
    try {
      const res = await fetch(`${API_BASE}/enhance-batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ limit })
      });
      if (!res.ok) throw new Error('Batch enhancement failed');
      const data = await res.json();
      setError(null);
      // Refresh data
      fetchDashboard();
      if (activeTab === 'tenders') fetchTenders();
      alert(data.message || `Enhanced ${data.enhanced} tenders`);
    } catch (err: any) {
      setError(err.message);
    }
    setEnhancing(false);
  };

  const reloadData = async () => {
    setLoading(true);
    try {
      await fetch(`${API_BASE}/reload`, { method: 'POST' });
      fetchDashboard();
      if (activeTab === 'tenders') fetchTenders();
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  const exportTenders = async () => {
    try {
      const res = await fetch(`${API_BASE}/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      if (!res.ok) throw new Error('Export failed');
      const data = await res.json();
      // Download the file
      window.open(`${API_BASE}/export/download/${data.filename}`, '_blank');
    } catch (err: any) {
      setError(err.message);
    }
  };

  const formatNumber = (num: number) => num.toLocaleString();

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-900">Tender Manager</h1>
        <div className="flex gap-2">
          <button
            onClick={reloadData}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Reload
          </button>
          <button
            onClick={() => enhanceBatch(10)}
            disabled={enhancing}
            className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-400"
          >
            <Sparkles className="w-4 h-4" />
            AI Enhance (10)
          </button>
          <button
            onClick={exportTenders}
            className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
          >
            <Download className="w-4 h-4" />
            Export Excel
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-4 mb-6 border-b">
        <button
          onClick={() => setActiveTab('dashboard')}
          className={`px-4 py-2 font-medium ${activeTab === 'dashboard' ? 'border-b-2 border-blue-600 text-blue-600' : 'text-gray-600'}`}
        >
          Dashboard
        </button>
        <button
          onClick={() => setActiveTab('tenders')}
          className={`px-4 py-2 font-medium ${activeTab === 'tenders' ? 'border-b-2 border-blue-600 text-blue-600' : 'text-gray-600'}`}
        >
          All Tenders
        </button>
      </div>

      {/* Dashboard Tab */}
      {activeTab === 'dashboard' && stats && (
        <div className="space-y-6">
          {/* Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            <div className="bg-white rounded-lg shadow p-4">
              <p className="text-sm text-gray-500">Total Tenders</p>
              <p className="text-2xl font-bold text-blue-600">{formatNumber(stats.total_tenders)}</p>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
              <p className="text-sm text-gray-500">Total Items</p>
              <p className="text-2xl font-bold text-green-600">{formatNumber(stats.total_items)}</p>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
              <p className="text-sm text-gray-500">Departments</p>
              <p className="text-2xl font-bold text-purple-600">{stats.unique_departments}</p>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
              <p className="text-sm text-gray-500">AI Enhanced</p>
              <p className="text-2xl font-bold text-orange-600">{formatNumber(stats.ai_enhanced_count)}</p>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
              <p className="text-sm text-gray-500">Avg Confidence</p>
              <p className="text-2xl font-bold text-cyan-600">{stats.avg_confidence}%</p>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
              <p className="text-sm text-gray-500">Items/Tender</p>
              <p className="text-2xl font-bold text-pink-600">{stats.items_per_tender}</p>
            </div>
          </div>

          {/* AI Stats */}
          {stats.ai_stats && (
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold mb-4">AI Enhancement Engine</h2>
              <div className="flex gap-6">
                <div className={`px-4 py-2 rounded ${stats.ai_stats.available ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                  {stats.ai_stats.available ? 'Online' : 'Offline'}
                </div>
                <div className="text-gray-600">
                  <span className="font-medium">{stats.ai_stats.total_rpm}</span> RPM Combined
                </div>
                {stats.ai_stats.workers.map((w, i) => (
                  <div key={i} className="text-gray-600">
                    <span className="font-medium capitalize">{w.name}:</span> {w.requests} requests
                    {w.errors > 0 && <span className="text-red-500"> ({w.errors} errors)</span>}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Department Breakdown */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold mb-4">Tenders by Department</h2>
              <div className="space-y-2">
                {Object.entries(stats.department_breakdown).slice(0, 10).map(([dept, count]) => (
                  <div key={dept} className="flex justify-between items-center">
                    <span className="text-gray-700 truncate" title={dept}>{dept || 'Unknown'}</span>
                    <span className="font-medium text-blue-600">{count}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Top Items */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold mb-4">Top Items</h2>
              <div className="space-y-2">
                {stats.top_items.map((item, i) => (
                  <div key={i} className="flex justify-between items-center">
                    <span className="text-gray-700 truncate text-sm" title={item.description}>
                      {item.description}
                    </span>
                    <span className="font-medium text-green-600 ml-2">{item.count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Recent Tenders */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4">Recent Tenders</h2>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Reference</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Title</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Items</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {stats.recent_tenders.map((t, i) => (
                    <tr key={i} className="hover:bg-gray-50 cursor-pointer" onClick={() => {
                      setActiveTab('tenders');
                      setSearch(t.reference);
                    }}>
                      <td className="px-4 py-3 text-sm font-medium text-blue-600">{t.reference}</td>
                      <td className="px-4 py-3 text-sm text-gray-700 truncate max-w-xs">{t.title || '-'}</td>
                      <td className="px-4 py-3 text-sm text-gray-700">{t.items}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Tenders Tab */}
      {activeTab === 'tenders' && (
        <div className="space-y-4">
          {/* Filters */}
          <div className="flex gap-4 flex-wrap">
            <div className="flex-1 min-w-64">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <input
                  type="text"
                  placeholder="Search tenders or items..."
                  value={search}
                  onChange={(e) => { setSearch(e.target.value); setPage(1); }}
                  className="w-full pl-10 pr-4 py-2 border rounded-lg"
                />
              </div>
            </div>
            <select
              value={department}
              onChange={(e) => { setDepartment(e.target.value); setPage(1); }}
              className="border rounded-lg px-4 py-2"
            >
              <option value="">All Departments</option>
              {departments.map(d => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>

          {/* Results count */}
          <div className="text-sm text-gray-600">
            Showing {tenders.length} of {totalTenders} tenders (Page {page} of {totalPages})
          </div>

          {/* Tenders Table */}
          <div className="bg-white rounded-lg shadow overflow-hidden">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Reference</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Department</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Items</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Confidence</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">AI</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {tenders.map((tender) => (
                  <tr key={tender.reference} className="hover:bg-gray-50">
                    <td className="px-4 py-3 text-sm font-medium text-blue-600">{tender.reference}</td>
                    <td className="px-4 py-3 text-sm text-gray-700 truncate max-w-xs">{tender.department || '-'}</td>
                    <td className="px-4 py-3 text-sm text-gray-700">{tender.items?.length || 0}</td>
                    <td className="px-4 py-3 text-sm">
                      <span className={`px-2 py-1 rounded text-xs ${tender.ocr_confidence >= 70 ? 'bg-green-100 text-green-700' : tender.ocr_confidence >= 50 ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'}`}>
                        {tender.ocr_confidence?.toFixed(0) || 0}%
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm">
                      {tender.ai_enhanced ? (
                        <span className="text-purple-600">Enhanced</span>
                      ) : (
                        <span className="text-gray-400">-</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-sm">
                      <div className="flex gap-2">
                        <button
                          onClick={() => setSelectedTender(tender)}
                          className="text-blue-600 hover:text-blue-800"
                          title="View Details"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        {!tender.ai_enhanced && (
                          <button
                            onClick={() => enhanceTender(tender.reference)}
                            disabled={enhancing}
                            className="text-purple-600 hover:text-purple-800 disabled:text-gray-400"
                            title="AI Enhance"
                          >
                            <Sparkles className="w-4 h-4" />
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex justify-center gap-2">
            <button
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page === 1}
              className="px-4 py-2 border rounded-lg disabled:opacity-50"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="px-4 py-2">Page {page} of {totalPages}</span>
            <button
              onClick={() => setPage(p => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
              className="px-4 py-2 border rounded-lg disabled:opacity-50"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Tender Detail Modal */}
      {selectedTender && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h2 className="text-xl font-bold text-gray-900">{selectedTender.reference}</h2>
                  <p className="text-gray-600">{selectedTender.department}</p>
                </div>
                <button onClick={() => setSelectedTender(null)} className="text-gray-500 hover:text-gray-700">
                  &times;
                </button>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="bg-gray-50 p-3 rounded">
                  <p className="text-xs text-gray-500">Items</p>
                  <p className="font-semibold">{selectedTender.items?.length || 0}</p>
                </div>
                <div className="bg-gray-50 p-3 rounded">
                  <p className="text-xs text-gray-500">Confidence</p>
                  <p className="font-semibold">{selectedTender.ocr_confidence?.toFixed(1)}%</p>
                </div>
                <div className="bg-gray-50 p-3 rounded">
                  <p className="text-xs text-gray-500">AI Enhanced</p>
                  <p className="font-semibold">{selectedTender.ai_enhanced ? 'Yes' : 'No'}</p>
                </div>
                <div className="bg-gray-50 p-3 rounded">
                  <p className="text-xs text-gray-500">Has Arabic</p>
                  <p className="font-semibold">{selectedTender.has_arabic ? 'Yes' : 'No'}</p>
                </div>
              </div>

              {!selectedTender.ai_enhanced && (
                <button
                  onClick={() => enhanceTender(selectedTender.reference)}
                  disabled={enhancing}
                  className="mb-4 flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-400"
                >
                  <Sparkles className="w-4 h-4" />
                  {enhancing ? 'Enhancing...' : 'AI Enhance This Tender'}
                </button>
              )}

              <h3 className="font-semibold mb-2">Items ({selectedTender.items?.length || 0})</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">#</th>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Description</th>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Qty</th>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Unit</th>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">AI</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {selectedTender.items?.map((item, i) => (
                      <tr key={i} className={item.ai_cleaned ? 'bg-purple-50' : ''}>
                        <td className="px-3 py-2 text-sm">{item.item_number || i + 1}</td>
                        <td className="px-3 py-2 text-sm">{item.description}</td>
                        <td className="px-3 py-2 text-sm">{item.quantity}</td>
                        <td className="px-3 py-2 text-sm">{item.unit}</td>
                        <td className="px-3 py-2 text-sm">
                          {item.ai_cleaned && <Sparkles className="w-3 h-3 text-purple-600" />}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
