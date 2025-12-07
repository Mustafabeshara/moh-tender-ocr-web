import { Settings } from 'lucide-react';
import { ExtractionConfig } from '../types';

interface ConfigPanelProps {
  config: ExtractionConfig;
  onChange: (config: ExtractionConfig) => void;
}

export function ConfigPanel({ config, onChange }: ConfigPanelProps) {
  const handleChange = (key: keyof ExtractionConfig, value: any) => {
    onChange({ ...config, [key]: value });
  };

  const toggleLanguage = (lang: string) => {
    const current = config.languages;
    if (current.includes(lang)) {
      if (current.length > 1) {
        handleChange('languages', current.filter(l => l !== lang));
      }
    } else {
      handleChange('languages', [...current, lang]);
    }
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6">
      <div className="flex items-center gap-2 mb-4">
        <Settings className="w-5 h-5 text-gray-500" />
        <h3 className="font-semibold text-gray-800">Extraction Settings</h3>
      </div>

      <div className="space-y-4">
        {/* Languages */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            OCR Languages
          </label>
          <div className="flex gap-2">
            {['eng', 'ara'].map((lang) => (
              <button
                key={lang}
                onClick={() => toggleLanguage(lang)}
                className={`
                  px-4 py-2 rounded-lg text-sm font-medium transition-colors
                  ${config.languages.includes(lang)
                    ? 'bg-primary-100 text-primary-700 border-2 border-primary-300'
                    : 'bg-gray-100 text-gray-600 border-2 border-transparent hover:bg-gray-200'
                  }
                `}
              >
                {lang === 'eng' ? 'English' : 'Arabic'}
              </button>
            ))}
          </div>
        </div>

        {/* DPI */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Image DPI: {config.dpi}
          </label>
          <input
            type="range"
            min="150"
            max="600"
            step="50"
            value={config.dpi}
            onChange={(e) => handleChange('dpi', parseInt(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
          <div className="flex justify-between text-xs text-gray-400 mt-1">
            <span>150 (Fast)</span>
            <span>300 (Balanced)</span>
            <span>600 (High Quality)</span>
          </div>
        </div>

        {/* Max Pages */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Max Pages per PDF: {config.max_pages}
          </label>
          <input
            type="range"
            min="1"
            max="50"
            value={config.max_pages}
            onChange={(e) => handleChange('max_pages', parseInt(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
          <div className="flex justify-between text-xs text-gray-400 mt-1">
            <span>1</span>
            <span>25</span>
            <span>50</span>
          </div>
        </div>

        {/* Department */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Department
          </label>
          <select
            value={config.department}
            onChange={(e) => handleChange('department', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="Biomedical Engineering">Biomedical Engineering</option>
            <option value="Medical Store">Medical Store</option>
            <option value="Laboratory">Laboratory</option>
            <option value="Radiology">Radiology</option>
            <option value="Other">Other</option>
          </select>
        </div>
      </div>
    </div>
  );
}
