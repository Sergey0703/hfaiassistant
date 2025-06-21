# backend/app/dependencies.py - ИСПРАВЛЕННАЯ ВЕРСИЯ ДЛЯ HUGGINGFACE SPACES
"""
Зависимости и инициализация сервисов для HuggingFace Spaces
ИСПРАВЛЕНИЕ: Полностью синхронная инициализация + background async loading
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
        self.service_type = "hf_spaces_fallback_v2"
        self.vector_db = type('MockVectorDB', (), {
            'persist_directory': './fallback_db'
        })()
        self.initialization_error = None
        logger.info("📝 HF Spaces document fallback service ready")
    
    def search(self, query: str, category: str = None, limit: int = 5, min_relevance: float = 0.3):
        """Синхронный поиск с демо результатами"""
        logger.info(f"🔍 Fallback search: '{query}'")
        
        demo_result = {
            "content": f"""Legal Analysis for: "{query}"

🏛️ **Document Summary (Demo Mode)**
This demonstrates the expected API response structure for legal document search.

📋 **Search Context:**
• Query: "{query}"
• Category: {category or "General Legal"}
• Platform: HuggingFace Spaces
• Mode: Document service initializing...

⚖️ **Expected Features (when fully loaded):**
• ChromaDB vector search with semantic similarity
• Multiple legal document categories
• Relevance scoring and ranking
• Legal citation extraction

🔧 **Current Status:**
Document service is loading in background. This demo shows the expected response format.

💡 **Note:** Full document search will be available once ChromaDB initialization completes.""",
            
            "filename": f"legal_search_{query.replace(' ', '_')[:20]}.txt",
            "document_id": f"demo_{int(time.time())}",
            "relevance_score": 0.95,
            "metadata": {
                "status": "demo_response",
                "category": category or "general",
                "service": "hf_spaces_fallback_v2",
                "query": query,
                "platform": "HuggingFace Spaces",
                "background_loading": _background_loading_started
            }
        }
        
        return [demo_result]
    
    def get_stats(self):
        """Синхронная статистика"""
        return {
            "total_documents": 0,
            "categories": ["general", "legislation", "jurisprudence", "government"],
            "database_type": "Initializing (ChromaDB loading...)",
            "status": "Background initialization in progress",
            "platform": "HuggingFace Spaces",
            "background_loading": _background_loading_started,
            "services_available": SERVICES_AVAILABLE
        }
    
    def get_all_documents(self):
        """Синхронный список документов"""
        return []
    
    def delete_document(self, doc_id: str):
        """Синхронное удаление"""
        logger.info(f"Demo delete: {doc_id}")
        return False
    
    def process_and_store_file(self, file_path: str, category: str = "general"):
        """Синхронная обработка файла"""
        logger.info(f"Demo file processing: {file_path}")
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
    
    def scrape_legal_site(self, url: str):
        """Синхронный демо скрапинг"""
        logger.info(f"🔍 Demo scraping: {url}")
        
        demo_content = f"""📄 **Legal Document from {url}**

This is a demonstration of the web scraping functionality for HuggingFace Spaces.

**Document Source:** {url}
**Status:** Scraper service initializing...
**Platform:** HuggingFace Spaces

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

💡 **Real scraping will be available once background initialization completes.**"""
        
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
                'background_loading': _background_loading_started,
                'url': url
            },
            'category': 'demo'
        })()
    
    def scrape_multiple_urls(self, urls: List[str], delay: float = 1.0):
        """Синхронный массовый скрапинг"""
        results = []
        for url in urls:
            doc = self.scrape_legal_site(url)
            results.append(doc)
        return results

class HFSpacesImprovedLLMFallback:
    """Улучшенный LLM fallback с поддержкой украинского языка"""
    
    def __init__(self):
        self.service_type = "hf_spaces_gptq_fallback_improved"
        self.model_loaded = False
        self.target_model = "TheBloke/Llama-2-7B-Chat-GPTQ"
        logger.info(f"🤖 HF Spaces GPTQ fallback ready for: {self.target_model}")
    
    def answer_legal_question(self, question: str, context_documents: List[Dict], language: str = "en"):
        """Синхронные улучшенные демо ответы"""
        from services.huggingface_llm_service import LLMResponse
        
        if language == "uk":
            demo_content = f"""🏛️ **Юридична консультація (GPTQ модель завантажується)**

**Ваше питання:** {question}

**Аналіз документів:** Знайдено {len(context_documents)} релевантних документів у базі знань.

🤖 **Статус GPTQ моделі:**
• Модель: `{self.target_model}`
• Оптимізація: 4-bit GPTQ квантизація
• Платформа: HuggingFace Spaces
• Статус: Завантаження в фоновому режимі...

📋 **Очікувана функціональність:**
✅ Високоякісний аналіз юридичних питань
✅ Підтримка української та англійської мов
✅ Контекстуальні відповіді на основі документів
✅ Посилання на конкретні статті законів
✅ Практичні рекомендації та покрокові дії

⏳ **Процес завантаження:**
1. Ініціалізація трансформерів HuggingFace
2. Завантаження GPTQ квантизованої моделі (~4GB)
3. Оптимізація для HuggingFace Spaces (обмеження пам'яті)
4. Підготовка правової системи промптів

💡 **Порада:** GPTQ модель забезпечить високу якість відповідей при мінімальному використанні пам'яті. Спробуйте ще раз через 1-2 хвилини для отримання повної AI відповіді.

🔧 **Технічні деталі:**
• Архітектура: Llama-2-7B з 4-bit квантизацією
• Пам'ять: Оптимізовано для 16GB лімітів HF Spaces
• Мови: Англійська та українська
• Спеціалізація: Правові консультації та аналіз"""
        else:
            demo_content = f"""🏛️ **Legal Consultation (GPTQ Model Loading)**

**Your Question:** {question}

**Document Analysis:** Found {len(context_documents)} relevant documents in knowledge base.

🤖 **GPTQ Model Status:**
• Model: `{self.target_model}`
• Optimization: 4-bit GPTQ quantization
• Platform: HuggingFace Spaces
• Status: Loading in background...

📋 **Expected Functionality:**
✅ High-quality legal question analysis
✅ English and Ukrainian language support
✅ Context-aware responses based on documents
✅ Specific law and regulation references
✅ Practical recommendations and step-by-step guidance

⏳ **Loading Process:**
1. Initializing HuggingFace Transformers
2. Loading GPTQ quantized model (~4GB)
3. Optimizing for HuggingFace Spaces memory limits
4. Preparing legal prompt system

💡 **Tip:** GPTQ model will provide high-quality responses with minimal memory usage. Try again in 1-2 minutes for full AI response.

🔧 **Technical Details:**
• Architecture: Llama-2-7B with 4-bit quantization
• Memory: Optimized for 16GB HF Spaces limits
• Languages: English and Ukrainian
• Specialization: Legal consultation and analysis"""
        
        return LLMResponse(
            content=demo_content,
            model=self.target_model,
            tokens_used=len(demo_content.split()),
            response_time=0.3,
            success=True,
            error=None
        )
    
    def get_service_status(self):
        """Синхронный статус"""
        return {
            "model_loaded": False,
            "model_name": self.target_model,
            "huggingface_available": True,
            "service_type": "gptq_fallback_improved",
            "environment": "HuggingFace Spaces",
            "status": "GPTQ model loading in background",
            "supported_languages": ["en", "uk"],
            "background_loading": _background_loading_started,
            "optimization": "4-bit GPTQ quantization",
            "memory_efficient": True,
            "target_model": self.target_model,
            "recommendations": [
                "GPTQ model provides production-quality legal analysis",
                "4-bit quantization enables efficient memory usage",
                "Background loading ensures fast API startup",
                "Full AI responses available after model loads"
            ]
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
            # НЕ инициализируем ChromaDB здесь - слишком медленно
            # Используем fallback и запускаем ChromaDB в фоне
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
            # Библиотеки есть, но используем fallback для быстрого старта
            # Реальный scraper инициализируем в фоне
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
    """Синхронная инициализация LLM service"""
    global llm_service, _llm_service_initialized, LLM_ENABLED
    
    if _llm_service_initialized:
        return llm_service
    
    logger.info("🔄 Sync initializing LLM service...")
    
    try:
        # Проверяем демо режим
        if settings.LLM_DEMO_MODE:
            logger.info("🎭 LLM demo mode enabled")
            llm_service = HFSpacesImprovedLLMFallback()
            LLM_ENABLED = False
            _llm_service_initialized = True
            return llm_service
        
        # Проверяем наличие зависимостей без загрузки модели
        try:
            import torch
            import transformers
            dependencies_available = True
            logger.info("🤖 GPTQ dependencies available")
        except ImportError as e:
            logger.warning(f"GPTQ dependencies missing: {e}")
            dependencies_available = False
        
        # Всегда используем fallback для быстрого старта
        llm_service = HFSpacesImprovedLLMFallback()
        LLM_ENABLED = False
        _llm_service_initialized = True
        
        # Запускаем background загрузку GPTQ модели если зависимости есть
        if dependencies_available:
            _start_background_gptq_init()
        
        logger.info("✅ LLM service ready (fallback + background GPTQ loading)")
        return llm_service
        
    except Exception as e:
        logger.error(f"❌ LLM sync init failed: {e}")
        llm_service = HFSpacesImprovedLLMFallback()
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
            time.sleep(2)  # Даем приложению запуститься
            logger.info("📚 Background: Initializing ChromaDB...")
            
            from services.chroma_service import DocumentService
            
            # Создаем директорию
            os.makedirs(settings.CHROMADB_PATH, exist_ok=True)
            
            # Инициализируем ChromaDB
            real_service = DocumentService(settings.CHROMADB_PATH)
            
            # Заменяем глобальный сервис
            global document_service, CHROMADB_ENABLED
            document_service = real_service
            CHROMADB_ENABLED = True
            
            logger.info("✅ Background: ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Background ChromaDB init failed: {e}")
    
    # Запускаем в отдельном треде
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
            time.sleep(3)  # Даем приложению запуститься
            logger.info("🌐 Background: Initializing real scraper...")
            
            from services.scraper_service import LegalSiteScraper
            
            # Создаем реальный scraper
            real_scraper = LegalSiteScraper()
            
            # Заменяем глобальный сервис
            global scraper
            scraper = real_scraper
            
            logger.info("✅ Background: Real scraper initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Background scraper init failed: {e}")
    
    # Запускаем в отдельном треде
    future = _executor.submit(background_scraper_worker)
    _background_tasks["scraper"] = future

def _start_background_gptq_init():
    """Запускает загрузку GPTQ модели в фоне"""
    global _background_tasks
    
    if "gptq" in _background_tasks:
        return
    
    logger.info("🚀 Starting background GPTQ model loading...")
    
    def background_gptq_worker():
        try:
            time.sleep(5)  # Даем приложению полностью запуститься
            logger.info("🤖 Background: Loading GPTQ model...")
            
            from services.huggingface_llm_service import create_llm_service
            
            # Пытаемся загрузить GPTQ модель
            real_llm = create_llm_service("TheBloke/Llama-2-7B-Chat-GPTQ")
            
            # Проверяем что модель действительно загрузилась
            if hasattr(real_llm, 'model_loaded') and real_llm.model_loaded:
                # Заменяем глобальный сервис
                global llm_service, LLM_ENABLED
                llm_service = real_llm
                LLM_ENABLED = True
                
                logger.info("✅ Background: GPTQ model loaded successfully!")
            else:
                logger.warning("⚠️ Background: GPTQ model not ready, keeping fallback")
                
        except Exception as e:
            logger.error(f"❌ Background GPTQ loading failed: {e}")
    
    # Запускаем в отдельном треде
    future = _executor.submit(background_gptq_worker)
    _background_tasks["gptq"] = future

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
    _start_background_gptq_init()

# ====================================
# DEPENDENCY FUNCTIONS (СИНХРОННЫЕ)
# ====================================

def get_document_service():
    """Dependency для получения document service - СИНХРОННАЯ"""
    service = _init_document_service_sync()
    
    # Запускаем background tasks если еще не запущены
    if not _background_loading_started:
        _start_all_background_tasks()
    
    return service

def get_scraper_service():
    """Dependency для получения scraper service - СИНХРОННАЯ"""
    service = _init_scraper_service_sync()
    
    # Запускаем background tasks если еще не запущены
    if not _background_loading_started:
        _start_all_background_tasks()
    
    return service

def get_llm_service():
    """Dependency для получения LLM service - СИНХРОННАЯ"""
    service = _init_llm_service_sync()
    
    # Запускаем background tasks если еще не запущены
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
        "gptq_model": "TheBloke/Llama-2-7B-Chat-GPTQ",
        "background_loading": _background_loading_started,
        "background_tasks": {
            "chromadb_started": "chromadb" in _background_tasks,
            "scraper_started": "scraper" in _background_tasks,
            "gptq_started": "gptq" in _background_tasks
        },
        "initialization_status": {
            "document_service": _document_service_initialized,
            "scraper_service": _scraper_initialized,
            "llm_service": _llm_service_initialized
        },
        "real_features": {
            "vector_search": CHROMADB_ENABLED,
            "web_scraping": _scraper_initialized and not isinstance(scraper, HFSpacesFallbackScraperService),
            "ai_responses": LLM_ENABLED
        }
    }

# ====================================
# УБИРАЕМ АСИНХРОННУЮ ИНИЦИАЛИЗАЦИЮ
# ====================================

async def init_services():
    """Упрощенная функция для совместимости - НЕ ВЫЗЫВАЕТСЯ"""
    logger.info("🚀 HF Spaces: Using sync initialization with background loading")
    logger.info("📦 Services will initialize on first request + background tasks")
    
    # Просто помечаем что система готова
    global SERVICES_AVAILABLE
    SERVICES_AVAILABLE = True
    
    logger.info("✅ Sync initialization ready")

# ====================================
# НОВЫЕ ФУНКЦИИ ДЛЯ МОНИТОРИНГА
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
        "tasks": status
    }

def force_background_init():
    """Принудительно запускает background инициализацию"""
    _start_all_background_tasks()
    return {
        "message": "Background initialization started",
        "tasks_started": len(_background_tasks)
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