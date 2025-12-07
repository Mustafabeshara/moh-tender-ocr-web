import { useState } from 'react';
import { ChevronDown, ChevronUp, FileText, Calendar, Package, Globe } from 'lucide-react';
import { TenderData } from '../types';

interface ResultsTableProps {
  results: TenderData[];
  onExport: (format: 'json' | 'csv') => void;
}

export function ResultsTable({ results, onExport }: ResultsTableProps) {
  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  const [sortField, setSortField] = useState<keyof TenderData>('reference_number');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');

  const toggleExpand = (id: string) => {
    setExpandedRow(expandedRow === id ? null : id);
  };

  const handleSort = (field: keyof TenderData) => {
    if (field === sortField) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDir('asc');
    }
  };

  const sortedResults = [...results].sort((a, b) => {
    const aVal = a[sortField];
    const bVal = b[sortField];
    if (typeof aVal === 'string' && typeof bVal === 'string') {
      return sortDir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
    }
    if (typeof aVal === 'number' && typeof bVal === 'number') {
      return sortDir === 'asc' ? aVal - bVal : bVal - aVal;
    }
    return 0;
  });

  const getLanguageBadge = (language: string) => {
    const colors: Record<string, string> = {
      english: 'bg-blue-100 text-blue-700',
      arabic: 'bg-green-100 text-green-700',
      mixed: 'bg-purple-100 text-purple-700',
      unknown: 'bg-gray-100 text-gray-700',
    };
    return colors[language] || colors.unknown;
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'text-green-600';
    if (confidence >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
      {/* Header with Export */}
      <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
        <h3 className="font-semibold text-gray-800">
          Extraction Results ({results.length} tenders)
        </h3>
        <div className="flex gap-2">
          <button
            onClick={() => onExport('json')}
            className="px-3 py-1.5 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200"
          >
            Export JSON
          </button>
          <button
            onClick={() => onExport('csv')}
            className="px-3 py-1.5 text-sm bg-primary-100 text-primary-700 rounded-lg hover:bg-primary-200"
          >
            Export CSV
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-50">
            <tr>
              <th className="w-10 px-4 py-3"></th>
              <th
                className="px-4 py-3 text-left text-sm font-medium text-gray-600 cursor-pointer hover:text-gray-900"
                onClick={() => handleSort('reference_number')}
              >
                <div className="flex items-center gap-1">
                  Reference
                  {sortField === 'reference_number' && (
                    sortDir === 'asc' ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />
                  )}
                </div>
              </th>
              <th
                className="px-4 py-3 text-left text-sm font-medium text-gray-600 cursor-pointer hover:text-gray-900"
                onClick={() => handleSort('closing_date')}
              >
                <div className="flex items-center gap-1">
                  Closing Date
                  {sortField === 'closing_date' && (
                    sortDir === 'asc' ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />
                  )}
                </div>
              </th>
              <th
                className="px-4 py-3 text-left text-sm font-medium text-gray-600 cursor-pointer hover:text-gray-900"
                onClick={() => handleSort('items_count')}
              >
                <div className="flex items-center gap-1">
                  Items
                  {sortField === 'items_count' && (
                    sortDir === 'asc' ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />
                  )}
                </div>
              </th>
              <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">
                Language
              </th>
              <th
                className="px-4 py-3 text-left text-sm font-medium text-gray-600 cursor-pointer hover:text-gray-900"
                onClick={() => handleSort('ocr_confidence')}
              >
                <div className="flex items-center gap-1">
                  Confidence
                  {sortField === 'ocr_confidence' && (
                    sortDir === 'asc' ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />
                  )}
                </div>
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {sortedResults.map((tender) => (
              <>
                <tr
                  key={tender.id}
                  className="hover:bg-gray-50 cursor-pointer"
                  onClick={() => toggleExpand(tender.id || tender.reference_number)}
                >
                  <td className="px-4 py-3">
                    {expandedRow === (tender.id || tender.reference_number) ? (
                      <ChevronUp className="w-4 h-4 text-gray-400" />
                    ) : (
                      <ChevronDown className="w-4 h-4 text-gray-400" />
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <FileText className="w-4 h-4 text-red-500" />
                      <span className="font-mono font-medium text-gray-800">
                        {tender.reference_number || 'N/A'}
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2 text-gray-600">
                      <Calendar className="w-4 h-4" />
                      {tender.closing_date || 'Not found'}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2 text-gray-600">
                      <Package className="w-4 h-4" />
                      {tender.items_count}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getLanguageBadge(tender.language)}`}>
                      {tender.language}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <span className={`font-medium ${getConfidenceColor(tender.ocr_confidence)}`}>
                      {tender.ocr_confidence.toFixed(1)}%
                    </span>
                  </td>
                </tr>

                {/* Expanded Row */}
                {expandedRow === (tender.id || tender.reference_number) && (
                  <tr>
                    <td colSpan={6} className="bg-gray-50 px-6 py-4">
                      <div className="grid grid-cols-2 gap-6">
                        {/* Items List */}
                        <div>
                          <h4 className="text-sm font-semibold text-gray-700 mb-3">
                            Items ({tender.items.length})
                          </h4>
                          <div className="space-y-2 max-h-64 overflow-y-auto">
                            {tender.items.slice(0, 10).map((item, idx) => (
                              <div
                                key={idx}
                                className="p-2 bg-white rounded border border-gray-200"
                              >
                                <div className="flex justify-between items-start">
                                  <span className="text-xs text-gray-400">#{item.item_number}</span>
                                  <span className="text-xs font-medium text-primary-600">
                                    {item.quantity} {item.unit}
                                  </span>
                                </div>
                                <p className="text-sm text-gray-700 mt-1">
                                  {item.description.slice(0, 100)}
                                  {item.description.length > 100 ? '...' : ''}
                                </p>
                              </div>
                            ))}
                            {tender.items.length > 10 && (
                              <p className="text-sm text-gray-500 text-center py-2">
                                +{tender.items.length - 10} more items
                              </p>
                            )}
                          </div>
                        </div>

                        {/* Specifications & Info */}
                        <div>
                          <h4 className="text-sm font-semibold text-gray-700 mb-3">
                            Details
                          </h4>
                          <div className="space-y-3">
                            <div>
                              <span className="text-xs text-gray-500">Department</span>
                              <p className="text-sm text-gray-800">{tender.department}</p>
                            </div>
                            <div>
                              <span className="text-xs text-gray-500">Posting Date</span>
                              <p className="text-sm text-gray-800">{tender.posting_date || 'N/A'}</p>
                            </div>
                            <div>
                              <span className="text-xs text-gray-500">Source Files</span>
                              <p className="text-sm text-gray-800">
                                {tender.source_files.join(', ') || 'N/A'}
                              </p>
                            </div>
                            {tender.specifications_text && (
                              <div>
                                <span className="text-xs text-gray-500">Specifications</span>
                                <p className="text-sm text-gray-800 bg-white p-2 rounded border max-h-32 overflow-y-auto">
                                  {tender.specifications_text.slice(0, 300)}
                                  {tender.specifications_text.length > 300 ? '...' : ''}
                                </p>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </td>
                  </tr>
                )}
              </>
            ))}
          </tbody>
        </table>
      </div>

      {results.length === 0 && (
        <div className="py-12 text-center text-gray-500">
          No results to display
        </div>
      )}
    </div>
  );
}
