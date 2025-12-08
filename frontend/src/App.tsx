import { useState } from 'react';
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import { FileText, Upload, BarChart3, Settings, Download, Brain } from 'lucide-react';
import Dashboard from './pages/Dashboard';
import TenderList from './pages/TenderList';
import TenderDetail from './pages/TenderDetail';
import UploadPage from './pages/UploadPage';
import ScraperPage from './pages/ScraperPage';
import TenderManagerPage from './pages/TenderManagerPage';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="bg-white border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              <div className="flex items-center gap-3">
                <FileText className="w-8 h-8 text-blue-600" />
                <h1 className="text-xl font-bold text-gray-900">MOH Tender OCR Manager</h1>
              </div>
              
              <nav className="flex items-center gap-1">
                <NavLink
                  to="/"
                  className={({ isActive }) =>
                    `flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                      isActive ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100'
                    }`
                  }
                >
                  <BarChart3 className="w-4 h-4" />
                  Dashboard
                </NavLink>
                
                <NavLink
                  to="/tenders"
                  className={({ isActive }) =>
                    `flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                      isActive ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100'
                    }`
                  }
                >
                  <FileText className="w-4 h-4" />
                  Tenders
                </NavLink>
                
                <NavLink
                  to="/upload"
                  className={({ isActive }) =>
                    `flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                      isActive ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100'
                    }`
                  }
                >
                  <Upload className="w-4 h-4" />
                  Upload
                </NavLink>

                <NavLink
                  to="/scraper"
                  className={({ isActive }) =>
                    `flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                      isActive ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100'
                    }`
                  }
                >
                  <Download className="w-4 h-4" />
                  Scraper
                </NavLink>

                <NavLink
                  to="/manager"
                  className={({ isActive }) =>
                    `flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                      isActive ? 'bg-purple-100 text-purple-700' : 'text-gray-600 hover:bg-gray-100'
                    }`
                  }
                >
                  <Brain className="w-4 h-4" />
                  AI Manager
                </NavLink>
              </nav>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/tenders" element={<TenderList />} />
            <Route path="/tenders/:id" element={<TenderDetail />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/scraper" element={<ScraperPage />} />
            <Route path="/manager" element={<TenderManagerPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
