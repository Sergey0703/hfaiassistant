// File: src/config/api.ts
// API Configuration for Legal Assistant
// Change BASE_API_URL to switch to different hosting

// ========================================
// MAIN CONFIGURATION - CHANGE HERE
// ========================================
const USE_PROXY = true; // Set to false to use direct HF connection
const PROXY_BASE_URL = ''; // Leave empty to use same domain
const DIRECT_API_URL = 'https://sergey070373-aiassistant.hf.space';

// If USE_PROXY is true, use /api/proxy on same domain
// If false, use direct HuggingFace URL
const BASE_API_URL = USE_PROXY 
  ? '/api/proxy' 
  : DIRECT_API_URL;

// Alternative options (uncomment as needed):
// const BASE_API_URL = 'http://localhost:8000';  // For local development
// const BASE_API_URL = 'https://your-app.railway.app';  // For Railway
// const BASE_API_URL = 'https://your-app.herokuapp.com';  // For Heroku
// const BASE_API_URL = 'https://your-domain.com';  // For custom domain

// ========================================
// API ENDPOINTS CONFIGURATION
// ========================================
export const API_CONFIG = {
  BASE_URL: BASE_API_URL,
  TIMEOUT: 30000, // 30 seconds timeout
  
  // Default headers
  HEADERS: {
    'Content-Type': 'application/json',
  },
  
  // File upload headers
  UPLOAD_HEADERS: {
    'Content-Type': 'multipart/form-data',
  }
};

// API Endpoints
export const API_ENDPOINTS = {
  // Health check - use root endpoint instead of /health
  HEALTH: USE_PROXY ? `${BASE_API_URL}?path=/` : `${BASE_API_URL}/`,
  
  // User endpoints
  CHAT: USE_PROXY ? `${BASE_API_URL}?path=/api/user/chat` : `${BASE_API_URL}/api/user/chat`,
  CHAT_HISTORY: USE_PROXY ? `${BASE_API_URL}?path=/api/user/chat/history` : `${BASE_API_URL}/api/user/chat/history`,
  
  // Admin endpoints
  ADMIN_STATS: USE_PROXY ? `${BASE_API_URL}?path=/api/admin/stats` : `${BASE_API_URL}/api/admin/stats`,
  DOCUMENTS_LIST: USE_PROXY ? `${BASE_API_URL}?path=/api/admin/documents` : `${BASE_API_URL}/api/admin/documents`,
  DOCUMENTS_UPLOAD: USE_PROXY ? `${BASE_API_URL}?path=/api/admin/documents/upload` : `${BASE_API_URL}/api/admin/documents/upload`,
  DOCUMENTS_DELETE: (id: number) => USE_PROXY 
    ? `${BASE_API_URL}?path=/api/admin/documents/${id}` 
    : `${BASE_API_URL}/api/admin/documents/${id}`,
  
  // Scraper endpoints
  SCRAPE_URL: USE_PROXY ? `${BASE_API_URL}?path=/api/admin/scrape/url` : `${BASE_API_URL}/api/admin/scrape/url`,
  SCRAPE_BULK: USE_PROXY ? `${BASE_API_URL}?path=/api/admin/scrape/bulk` : `${BASE_API_URL}/api/admin/scrape/bulk`,
  SCRAPE_PREDEFINED: USE_PROXY ? `${BASE_API_URL}?path=/api/admin/scrape/predefined` : `${BASE_API_URL}/api/admin/scrape/predefined`,
  PREDEFINED_SITES: USE_PROXY ? `${BASE_API_URL}?path=/api/admin/predefined-sites` : `${BASE_API_URL}/api/admin/predefined-sites`,
};

// Helper function to create full URL (for backward compatibility)
export const createApiUrl = (endpoint: string): string => {
  return `${BASE_API_URL}${endpoint}`;
};

// Export base URL for direct usage
export const getBaseUrl = (): string => BASE_API_URL;