// File: src/App.tsx
import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Home, MessageSquare, Settings, Globe } from 'lucide-react';
import './App.css';
import './i18n/i18n';
import { API_CONFIG } from './config/api';

// Components
import UserChat from './components/UserChat';
import AdminDashboard from './components/AdminDashboard';
import HomePage from './components/HomePage';

const App: React.FC = () => {
  const { t, i18n } = useTranslation();
  const [currentLanguage, setCurrentLanguage] = useState<string>('en');

  useEffect(() => {
    const savedLanguage = localStorage.getItem('language') || 'en';
    setCurrentLanguage(savedLanguage);
    i18n.changeLanguage(savedLanguage);
  }, [i18n]);

  const changeLanguage = (lang: string): void => {
    setCurrentLanguage(lang);
    i18n.changeLanguage(lang);
    localStorage.setItem('language', lang);
  };

  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        {/* Navigation */}
        <nav className="bg-white shadow-lg">
          <div className="max-w-7xl mx-auto px-4">
            <div className="flex justify-between items-center h-16">
              {/* Logo */}
              <div className="flex items-center">
                <Link to="/" className="flex items-center space-x-2">
                  <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                    <span className="text-white font-bold">⚖️</span>
                  </div>
                  <span className="text-xl font-bold text-gray-800">
                    {t('app.title')}
                  </span>
                </Link>
                
                {/* API Status Info */}
                <div className="ml-3 flex items-center space-x-2">
                  <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                  <span className="text-xs text-blue-600">Ready</span>
                </div>
              </div>

              {/* Navigation Links */}
              <div className="flex items-center space-x-4">
                <Link
                  to="/"
                  className="flex items-center space-x-1 px-3 py-2 rounded-md text-sm font-medium text-gray-700 hover:text-blue-600 hover:bg-gray-100"
                >
                  <Home size={16} />
                  <span>{t('nav.home')}</span>
                </Link>
                
                <Link
                  to="/chat"
                  className="flex items-center space-x-1 px-3 py-2 rounded-md text-sm font-medium text-gray-700 hover:text-blue-600 hover:bg-gray-100"
                >
                  <MessageSquare size={16} />
                  <span>{t('nav.chat')}</span>
                </Link>

                <Link
                  to="/admin"
                  className="flex items-center space-x-1 px-3 py-2 rounded-md text-sm font-medium text-gray-700 hover:text-blue-600 hover:bg-gray-100"
                >
                  <Settings size={16} />
                  <span>{t('nav.admin')}</span>
                </Link>

                {/* Language Switcher */}
                <div className="flex items-center space-x-2">
                  <Globe size={16} className="text-gray-500" />
                  <select
                    value={currentLanguage}
                    onChange={(e) => changeLanguage(e.target.value)}
                    className="border border-gray-300 rounded-md px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="en">English</option>
                    <option value="uk">Українська</option>
                  </select>
                </div>
              </div>
            </div>
          </div>
        </nav>

        {/* Info Banner */}
        <div className="bg-blue-50 border-l-4 border-blue-400 p-4">
          <div className="flex">
            <div className="ml-3">
              <p className="text-sm text-blue-700">
                🚀 Connected to backend: {API_CONFIG.BASE_URL}
              </p>
              <p className="text-xs text-blue-600 mt-1">
                Status checking disabled - functionality will be verified on first use
              </p>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto py-6 px-4">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/chat" element={<UserChat />} />
            <Route path="/admin" element={<AdminDashboard />} />
          </Routes>
        </main>

        {/* Footer */}
        <footer className="bg-white border-t mt-auto">
          <div className="max-w-7xl mx-auto py-6 px-4">
            <div className="text-center text-gray-500 text-sm">
              {t('footer.copyright')}
              <div className="mt-1 text-xs">
                <span>Backend API: {API_CONFIG.BASE_URL}</span>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </Router>
  );
};

export default App;