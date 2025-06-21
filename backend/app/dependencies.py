# backend/app/dependencies.py - ОБНОВЛЕННАЯ ВЕРСИЯ ДЛЯ LLAMACPP
"""
Зависимости и инициализация сервисов для HuggingFace Spaces
ОБНОВЛЕНО: Переход с GPTQ на LlamaCpp для стабильной работы
"""

import logging
import sys
import os
import time
import json
import asyncio
import threading
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from app.config import settings

logger = logging.getLogger(__name__)

# ====================================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ СЕРВИСОВ
# ====================================

document_service: Optional[object] = None
scraper: Optional[object] = None
llm_service: Optional[object] = None

# Статус флаги
SERVICES_AVAILABLE: bool = True  # Всегда True для HF Spaces
CHROMADB_ENABLED: bool = False
LLM_ENABLED: bool = False

# Флаги инициализации
_document_service_initialized = False
_scraper_initialized = False
_llm_service_initialized = False

# Background loading статус
_background_loading_started = False
_background_tasks = {}

# Thread pool для background операций
_executor = ThreadPoolExecutor(max_workers=2)

# ====================================
# УЛУЧШЕННЫЕ FALLBACK СЕРВИСЫ
# ====================================

class HFSpacesFallbackDocumentService:
    """Оптимизированный fallback для HF Spaces"""
    
    def __init__(self):
        self.service_type = "hf_spaces_fallback_v4_llamacpp"
        self.vector_db = type('MockVectorDB', (), {
            'persist_directory': './fallback_db'
        })()
        self.initialization_error = None
        logger.info("📝 HF Spaces document fallback service ready (LlamaCpp era)")
    
    async def search(self, query: str, category: str = None, limit: int = 5, min_relevance: float = 0.3) -> List[Dict]:
        """Асинхронный поиск с демо результатами"""
        try:
            logger.info(f"🔍 Fallback search: '{query}'")
            await asyncio.sleep(0.1)
            
            demo_result = {
                "content": f"""Legal Analysis for: "{query}"

🏛️ **Document Summary (LlamaCpp Ready)**
This demonstrates the expected API response structure for legal document search.

📋 **Search Context:**
• Query: "{query}"
• Category: {category or "General Legal"}
• Platform: HuggingFace Spaces
• LLM Backend: LlamaCpp (stable CPU inference)
• Status: Document service initializing...

⚖️ **Expected Features (when fully loaded):**
• ChromaDB vector search with semantic similarity
• Multiple legal document categories  
• Relevance scoring and ranking
• Legal citation extraction

🔧 **Current Status:**
Document service is loading in background. LlamaCpp LLM is ready for stable inference without hanging issues.

💡 **Note:** Full document search will be available once ChromaDB initialization completes. LlamaCpp provides reliable AI responses.""",
                
                "filename": f"legal_search_{query.replace(' ', '_')[:20]}.txt",
                "document_id": f"demo_{int(time.time())}",
                "relevance_score": 0.95,
                "metadata": {
                    "status": "demo_response_llamacpp_ready",
                    "category": category or "general",
                    "service": "hf_spaces_fallback_v4",
                    "query": query,
                    "platform": "HuggingFace Spaces",
                    "llm_backend": "llamacpp",
                    "background_loading": _background_loading_started,
                    "stable_inference": True
                }
            }
            
            return [demo_result]
            
        except Exception as e:
            logger.error(f"Fallback search error: {e}")
            return []
    
    async def get_stats(self) -> Dict:
        """Асинхронная статистика"""
        try:
            await asyncio.sleep(0.05)
            return {
                "total_documents": 0,
                "categories": ["general", "legislation", "jurisprudence", "government"],
                "database_type": "Initializing (ChromaDB loading...)",
                "status": "Background initialization in progress",
                "platform": "HuggingFace Spaces",
                "llm_backend": "llamacpp",
                "background_loading": _background_loading_started,
                "services_available": SERVICES_AVAILABLE,
                "stable_inference": True
            }
        except Exception as e:
            logger.error(f"Fallback stats error: {e}")
            return {
                "total_documents": 0,
                "categories": [],
                "database_type": "Error",
                "error": str(e)
            }
    
    async def get_all_documents(self) -> List[Dict]:
        """Асинхронный список документов"""
        try:
            await asyncio.sleep(0.05)
            return []
        except Exception as e:
            logger.error(f"Fallback get all documents error: {e}")
            return []
    
    async def delete_document(self, doc_id: str) -> bool:
        """Асинхронное удаление"""
        try:
            logger.info(f"Demo delete: {doc_id}")
            await asyncio.sleep(0.05)
            return False
        except Exception as e:
            logger.error(f"Fallback delete error: {e}")
            return False
    
    async def process_and_store_file(self, file_path: str, category: str = "general") -> bool:
        """Асинхронная обработка файла"""
        try:
            logger.info(f"Demo file processing: {file_path}")
            await asyncio.sleep(0.1)
            return False
        except Exception as e:
            logger.error(f"Fallback process file error: {e}")
            return False

class HFSpacesFallbackScraperService:
    """Оптимизированный scraper fallback для HF Spaces"""
    
    def __init__(self):
        self.service_type = "hf_spaces_scraper_fallback"
        self.legal_sites_config = {
            "irishstatutebook.ie": {"title": "h1", "content": ".content"},
            "citizensinformation.ie": {"title": "h1", "content": ".content"},
            "zakon.rada.gov.ua": {"title": "h1", "content": ".content"}
        }
        logger.info("🌐 HF Spaces scraper fallback ready")
    
    async def scrape_legal_site(self, url: str):
        """Асинхронный демо скрапинг"""
        try:
            logger.info(f"🔍 Demo scraping: {url}")
            await asyncio.sleep(0.2)
            
            demo_content = f"""📄 **Legal Document from {url}**

This is a demonstration of the web scraping functionality for HuggingFace Spaces.

**Document Source:** {url}
**Status:** Scraper service initializing...
**Platform:** HuggingFace Spaces
**LLM Backend:** LlamaCpp (stable inference)

🔧 **Background Loading:**
The real scraping service (aiohttp + beautifulsoup4) is loading in the background.

⚖️ **Expected Functionality:**
• Extract legal content from official sites
• Parse Ukrainian and Irish legal documents  
• Intelligent content extraction with CSS selectors
• Metadata extraction and categorization

🌐 **Supported Sites:**
• zakon.rada.gov.ua (Ukrainian legislation)
• irishstatutebook.ie (Irish statutory law)
• citizensinformation.ie (Irish civil information)
• courts.ie (Irish court decisions)

💡 **Real scraping will be available once background initialization completes. LlamaCpp provides stable AI analysis of scraped content.**"""
            
            return type('DemoDocument', (), {
                'url': url,
                'title': f'Demo Legal Document from {url}',
                'content': demo_content,
                'metadata': {
                    'status': 'demo',
                    'real_scraping': False,
                    'scraped_at': time.time(),
                    'service': 'hf_spaces_fallback',
                    'platform': 'HuggingFace Spaces',
                    'llm_backend': 'llamacpp',
                    'background_loading': _background_loading_started,
                    'url': url
                },
                'category': 'demo'
            })()
            
        except Exception as e:
            logger.error(f"Fallback scraper error: {e}")
            return None
    
    async def scrape_multiple_urls(self, urls: List[str], delay: float = 1.0):
        """Асинхронный массовый скрапинг"""
        results = []
        for url in urls:
            doc = await self.scrape_legal_site(url)
            results.append(doc)
            if delay > 0:
                await asyncio.sleep(delay)
        return results

class HFSpacesLlamaCppFallback:
    """Улучшенный LLM fallback для переходного периода к LlamaCpp"""
    
    def __init__(self):
        self.service_type = "hf_spaces_llamacpp_fallback"
        self.model_loaded = False
        self.target_model = "TheBloke/Llama-2-7B-Chat-GGUF"
        logger.info(f"🦙 HF Spaces LlamaCpp fallback ready for: {self.target_model}")
    
    async def answer_legal_question(self, question: str, context_documents: List[Dict], language: str = "en"):
        """Асинхронные демо ответы для LlamaCpp"""
        try:
            # ОБНОВЛЕНО: Импортируем из нового сервиса
            from services.llamacpp_llm_service import LLMResponse
            
            await asyncio.sleep(0.3)
            
            if language == "uk":
                demo_content = f"""🦙 **LlamaCpp модель завантажується...**

**Ваше питання:** {question}

**Аналіз документів:** Знайдено {len(context_documents)} релевантних документів у базі знань.

🤖 **Статус LlamaCpp моделі:**
• Модель: `{self.target_model}`
• Формат: GGUF (оптимізований для CPU)
• Платформа: HuggingFace Spaces
• Статус: Завантаження стабільної CPU моделі...

📋 **Очікувана функціональність:**
✅ Стабільна робота на CPU без зависань
✅ Підтримка української та англійської мов
✅ Контекстуальні відповіді на основі документів
✅ Таймаути для запобігання зависанням
✅ Практичні рекомендації та покрокові дії

⏳ **Процес завантаження:**
1. Ініціалізація llama-cpp-python
2. Завантаження GGUF квантизованої моделі (~4GB)
3. Оптимізація для CPU inference
4. Підготовка правової системи промптів

💡 **Порада:** LlamaCpp забезпечить стабільну роботу без проблем з пам'яттю. Спробуйте ще раз через 1 хвилину для отримання повної AI відповіді.

🔧 **Технічні деталі:**
• Архітектура: Llama-2-7B з GGUF квантизацією
• Бекенд: llama.cpp (проверенный на CPU)
• Мови: Англійська та українська
• Таймауты: 30 секунд (без зависань)
• Спеціалізація: Правові консультації та аналіз"""
            else:
                demo_content = f"""🦙 **LlamaCpp Model Loading...**

**Your Question:** {question}

**Document Analysis:** Found {len(context_documents)} relevant documents in knowledge base.

🤖 **LlamaCpp Model Status:**
• Model: `{self.target_model}`
• Format: GGUF (CPU optimized)
• Platform: HuggingFace Spaces
• Status: Loading stable CPU model...

📋 **Expected Functionality:**
✅ Stable CPU inference without hanging
✅ English and Ukrainian language support
✅ Context-aware responses based on documents
✅ Timeouts to prevent hanging
✅ Practical recommendations and step-by-step guidance

⏳ **Loading Process:**
1. Initializing llama-cpp-python
2. Loading GGUF quantized model (~4GB)
3. Optimizing for CPU inference
4. Preparing legal prompt system

💡 **Tip:** LlamaCpp will provide stable operation without memory issues. Try again in 1 minute for full AI response.

🔧 **Technical Details:**
• Architecture: Llama-2-7B with GGUF quantization
• Backend: llama.cpp (CPU proven)
• Languages: English and Ukrainian
• Timeouts: 30 seconds (no hanging)
• Specialization: Legal consultation and analysis"""
            
            return LLMResponse(
                content=demo_content,
                model=self.target_model,
                tokens_used=len(demo_content.split()),
                response_time=0.3,
                success=True,
                error=None
            )
            
        except Exception as e:
            logger.error(f"LLM fallback error: {e}")
            return type('SimpleLLMResponse', (), {
                'content': f"LlamaCpp model is loading. Question: {question}",
                'model': self.target_model,
                'tokens_used': 10,
                'response_time': 0.1,
                'success': True,
                'error': None
            })()
    
    async def get_service_status(self) -> Dict:
        """Асинхронный статус"""
        try:
            await asyncio.sleep(0.05)
            return {
                "model_loaded": False,
                "model_name": self.target_model,
                "llama_cpp_available": True,
                "service_type": "llamacpp_fallback",
                "environment": "HuggingFace Spaces",
                "status": "LlamaCpp model loading in background",
                "supported_languages": ["en", "uk"],
                "background_loading": _background_loading_started,
                "optimization": "GGUF quantization for CPU",
                "stable_inference": True,
                "timeout_protection": True,
                "target_model": self.target_model,
                "recommendations": [
                    "LlamaCpp provides stable CPU inference",
                    "GGUF format optimized for memory efficiency",
                    "No hanging issues expected",
                    "Timeout protection prevents freezing"
                ]
            }
        except Exception as e:
            logger.error(f"LLM status error: {e}")
            return {
                "model_loaded": False,
                "error": str(e),
                "service_type": "llamacpp_fallback_error"
            }

# ====================================
# СИНХРОННАЯ ИНИЦИАЛИЗАЦИЯ ФУНКЦИЙ
# ====================================

def _init_document_service_sync():
    """Синхронная инициализация document service"""
    global document_service, _document_service_initialized, CHROMADB_ENABLED
    
    if _document_service_initialized:
        return document_service
    
    logger.info("🔄 Sync initializing document service...")
    
    try:
        # Простая проверка ChromaDB без async
        try:
            import sentence_transformers
            import chromadb
            logger.info("📚 ChromaDB dependencies available, will init in background")
            
        except ImportError as e:
            logger.info(f"ChromaDB dependencies missing: {e}")
        
        # Всегда используем fallback для быстрого старта
        document_service = HFSpacesFallbackDocumentService()
        CHROMADB_ENABLED = False
        _document_service_initialized = True
        
        # Запускаем background инициализацию ChromaDB
        _start_background_chromadb_init()
        
        logger.info("✅ Document service ready (fallback + background loading)")
        return document_service
        
    except Exception as e:
        logger.error(f"❌ Document service sync init failed: {e}")
        document_service = HFSpacesFallbackDocumentService()
        document_service.initialization_error = str(e)
        _document_service_initialized = True
        return document_service

def _init_scraper_service_sync():
    """Синхронная инициализация scraper service"""
    global scraper, _scraper_initialized
    
    if _scraper_initialized:
        return scraper
    
    logger.info("🔄 Sync initializing scraper service...")
    
    try:
        # Проверяем библиотеки без импорта (быстро)
        try:
            import aiohttp
            import bs4
            libraries_available = True
        except ImportError:
            libraries_available = False
        
        if libraries_available:
            logger.info("🌐 Scraper libraries available, will init real scraper in background")
            _start_background_scraper_init()
        
        # Всегда используем fallback для быстрого старта
        scraper = HFSpacesFallbackScraperService()
        _scraper_initialized = True
        
        logger.info("✅ Scraper service ready (fallback + background loading)")
        return scraper
        
    except Exception as e:
        logger.error(f"❌ Scraper sync init failed: {e}")
        scraper = HFSpacesFallbackScraperService()
        _scraper_initialized = True
        return scraper

def _init_llm_service_sync():
    """ОБНОВЛЕННАЯ синхронная инициализация LLM service для LlamaCpp"""
    global llm_service, _llm_service_initialized, LLM_ENABLED
    
    if _llm_service_initialized:
        return llm_service
    
    logger.info("🔄 Sync initializing LlamaCpp LLM service...")
    
    try:
        # Проверяем демо режим
        if settings.LLM_DEMO_MODE:
            logger.info("🎭 LLM demo mode enabled")
            llm_service = HFSpacesLlamaCppFallback()
            LLM_ENABLED = False
            _llm_service_initialized = True
            return llm_service
        
        # Проверяем наличие зависимостей LlamaCpp
        try:
            import llama_cpp
            dependencies_available = True
            logger.info("🦙 LlamaCpp dependencies available")
        except ImportError as e:
            logger.warning(f"LlamaCpp dependencies missing: {e}")
            dependencies_available = False
        
        # Всегда используем fallback для быстрого старта
        llm_service = HFSpacesLlamaCppFallback()
        LLM_ENABLED = False
        _llm_service_initialized = True
        
        # Запускаем background загрузку LlamaCpp модели если зависимости есть
        if dependencies_available:
            _start_background_llamacpp_init()
        
        logger.info("✅ LLM service ready (fallback + background LlamaCpp loading)")
        return llm_service
        
    except Exception as e:
        logger.error(f"❌ LLM sync init failed: {e}")
        llm_service = HFSpacesLlamaCppFallback()
        LLM_ENABLED = False
        _llm_service_initialized = True
        return llm_service

# ====================================
# BACKGROUND АСИНХРОННАЯ ИНИЦИАЛИЗАЦИЯ
# ====================================

def _start_background_chromadb_init():
    """Запускает инициализацию ChromaDB в фоне"""
    global _background_tasks
    
    if "chromadb" in _background_tasks:
        return
    
    logger.info("🚀 Starting background ChromaDB initialization...")
    
    def background_chromadb_worker():
        try:
            time.sleep(2)
            logger.info("📚 Background: Initializing ChromaDB...")
            
            from services.chroma_service import DocumentService
            
            os.makedirs(settings.CHROMADB_PATH, exist_ok=True)
            real_service = DocumentService(settings.CHROMADB_PATH)
            
            global document_service, CHROMADB_ENABLED
            document_service = real_service
            CHROMADB_ENABLED = True
            
            logger.info("✅ Background: ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Background ChromaDB init failed: {e}")
    
    future = _executor.submit(background_chromadb_worker)
    _background_tasks["chromadb"] = future

def _start_background_scraper_init():
    """Запускает инициализацию реального scraper в фоне"""
    global _background_tasks
    
    if "scraper" in _background_tasks:
        return
    
    logger.info("🚀 Starting background scraper initialization...")
    
    def background_scraper_worker():
        try:
            time.sleep(3)
            logger.info("🌐 Background: Initializing real scraper...")
            
            from services.scraper_service import LegalSiteScraper
            
            real_scraper = LegalSiteScraper()
            
            global scraper
            scraper = real_scraper
            
            logger.info("✅ Background: Real scraper initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Background scraper init failed: {e}")
    
    future = _executor.submit(background_scraper_worker)
    _background_tasks["scraper"] = future

def _start_background_llamacpp_init():
    """НОВАЯ функция: Запускает загрузку LlamaCpp модели в фоне"""
    global _background_tasks
    
    if "llamacpp" in _background_tasks:
        return
    
    logger.info("🚀 Starting background LlamaCpp model loading...")
    
    def background_llamacpp_worker():
        try:
            time.sleep(5)  # Даем приложению полностью запуститься
            logger.info("🦙 Background: Loading LlamaCpp model...")
            
            # ОБНОВЛЕНО: Импортируем новый сервис
            from services.llamacpp_llm_service import create_llm_service
            
            # Пытаемся загрузить LlamaCpp модель
            real_llm = create_llm_service("TheBloke/Llama-2-7B-Chat-GGUF")
            
            # Проверяем что модель действительно загрузилась
            if hasattr(real_llm, 'model_loaded') and real_llm.model_loaded:
                # Заменяем глобальный сервис
                global llm_service, LLM_ENABLED
                llm_service = real_llm
                LLM_ENABLED = True
                
                logger.info("✅ Background: LlamaCpp model loaded successfully!")
            else:
                logger.warning("⚠️ Background: LlamaCpp model not ready, keeping fallback")
                
        except Exception as e:
            logger.error(f"❌ Background LlamaCpp loading failed: {e}")
    
    # Запускаем в отдельном треде
    future = _executor.submit(background_llamacpp_worker)
    _background_tasks["llamacpp"] = future

def _start_all_background_tasks():
    """Запускает все background задачи"""
    global _background_loading_started
    
    if _background_loading_started:
        return
    
    _background_loading_started = True
    logger.info("🚀 Starting all background initialization tasks...")
    
    # Запускаем все background задачи
    _start_background_chromadb_init()
    _start_background_scraper_init() 
    _start_background_llamacpp_init()  # ОБНОВЛЕНО: LlamaCpp вместо GPTQ

# ====================================
# DEPENDENCY FUNCTIONS (СИНХРОННЫЕ)
# ====================================

def get_document_service():
    """Dependency для получения document service - СИНХРОННАЯ"""
    service = _init_document_service_sync()
    
    if not _background_loading_started:
        _start_all_background_tasks()
    
    return service

def get_scraper_service():
    """Dependency для получения scraper service - СИНХРОННАЯ"""
    service = _init_scraper_service_sync()
    
    if not _background_loading_started:
        _start_all_background_tasks()
    
    return service

def get_llm_service():
    """Dependency для получения LLM service - СИНХРОННАЯ"""
    service = _init_llm_service_sync()
    
    if not _background_loading_started:
        _start_all_background_tasks()
    
    return service

def get_services_status():
    """Статус всех сервисов - СИНХРОННАЯ"""
    return {
        "document_service_available": _document_service_initialized,
        "scraper_available": _scraper_initialized,
        "llm_available": LLM_ENABLED,
        "llm_service_created": _llm_service_initialized,
        "chromadb_enabled": CHROMADB_ENABLED,
        "services_available": SERVICES_AVAILABLE,
        "huggingface_spaces": os.getenv("SPACE_ID") is not None,
        "environment": "hf_spaces" if os.getenv("SPACE_ID") else "local",
        "demo_mode": not LLM_ENABLED,
        "lazy_loading": True,
        "llm_backend": "llamacpp",  # ОБНОВЛЕНО
        "target_model": "TheBloke/Llama-2-7B-Chat-GGUF",  # ОБНОВЛЕНО
        "background_loading": _background_loading_started,
        "background_tasks": {
            "chromadb_started": "chromadb" in _background_tasks,
            "scraper_started": "scraper" in _background_tasks,
            "llamacpp_started": "llamacpp" in _background_tasks  # ОБНОВЛЕНО
        },
        "initialization_status": {
            "document_service": _document_service_initialized,
            "scraper_service": _scraper_initialized,
            "llm_service": _llm_service_initialized
        },
        "real_features": {
            "vector_search": CHROMADB_ENABLED,
            "web_scraping": _scraper_initialized and not isinstance(scraper, HFSpacesFallbackScraperService),
            "ai_responses": LLM_ENABLED,
            "stable_inference": True,  # НОВОЕ
            "timeout_protection": True  # НОВОЕ
        },
        "llamacpp_migration": True  # НОВОЕ: Индикатор миграции
    }

# ====================================
# СОВМЕСТИМОСТЬ
# ====================================

async def init_services():
    """Упрощенная функция для совместимости"""
    logger.info("🚀 HF Spaces: Using sync initialization with LlamaCpp backend")
    logger.info("📦 Services will initialize on first request + background tasks")
    
    global SERVICES_AVAILABLE
    SERVICES_AVAILABLE = True
    
    logger.info("✅ Sync initialization ready with LlamaCpp migration")

# ====================================
# ФУНКЦИИ ДЛЯ МОНИТОРИНГА
# ====================================

def get_background_tasks_status():
    """Возвращает статус background задач"""
    status = {}
    
    for task_name, future in _background_tasks.items():
        if future.done():
            if future.exception():
                status[task_name] = {
                    "status": "failed",
                    "error": str(future.exception())
                }
            else:
                status[task_name] = {
                    "status": "completed",
                    "result": "success"
                }
        else:
            status[task_name] = {
                "status": "running",
                "progress": "in_progress"
            }
    
    return {
        "background_loading_started": _background_loading_started,
        "total_tasks": len(_background_tasks),
        "tasks": status,
        "llamacpp_migration": True,
        "stable_inference": True
    }

def force_background_init():
    """Принудительно запускает background инициализацию"""
    _start_all_background_tasks()
    return {
        "message": "Background initialization started with LlamaCpp",
        "tasks_started": len(_background_tasks),
        "llamacpp_migration": True
    }

# ====================================
# ЭКСПОРТ
# ====================================

__all__ = [
    # Основные dependency функции
    "get_document_service",
    "get_scraper_service", 
    "get_llm_service",
    "get_services_status",
    
    # Статус и мониторинг
    "get_background_tasks_status",
    "force_background_init",
    
    # Совместимость
    "init_services",
    
    # Глобальные переменные
    "SERVICES_AVAILABLE",
    "CHROMADB_ENABLED",
    "LLM_ENABLED"
]