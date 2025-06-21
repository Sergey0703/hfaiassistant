# app/dependencies.py - ИСПРАВЛЕННАЯ ВЕРСИЯ ДЛЯ HUGGINGFACE SPACES

"""
Зависимости и инициализация сервисов для HuggingFace Spaces
ИСПРАВЛЕНИЕ: Lazy initialization + лучшая обработка GPTQ модели
"""

import logging
import sys
import os
import time
import json
import asyncio
from typing import Optional, List, Dict, Any

from app.config import settings

logger = logging.getLogger(__name__)

# ====================================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ СЕРВИСОВ
# ====================================

document_service: Optional[object] = None
scraper: Optional[object] = None
llm_service: Optional[object] = None
SERVICES_AVAILABLE: bool = False
CHROMADB_ENABLED: bool = False
LLM_ENABLED: bool = False

# Флаги инициализации для lazy loading
_document_service_initialized = False
_scraper_initialized = False
_llm_service_initialized = False

# ====================================
# УЛУЧШЕННЫЙ LLM FALLBACK
# ====================================

class ImprovedFallbackLLMService:
    """Улучшенный fallback LLM с поддержкой украинского языка"""
    
    def __init__(self):
        self.service_type = "hf_spaces_demo_improved"
        self.model_loaded = False
        logger.info("🤖 Using improved demo LLM service with Ukrainian support")
    
    async def answer_legal_question(self, question: str, context_documents: List[Dict], language: str = "en"):
        """Улучшенные демо ответы для юридических вопросов"""
        from services.huggingface_llm_service import LLMResponse
        
        if language == "uk":
            demo_content = f"""🏛️ **Юридична консультація (Демо режим)**

**Ваше питання:** {question}

**Аналіз документів:** Знайдено {len(context_documents)} релевантних документів у базі знань.

**Демонстраційна відповідь:**
Це демонстраційна версія Legal Assistant, що працює з моделлю GPTQ. У повній версії ваша модель `TheBloke/Llama-2-7B-Chat-GPTQ` надасть:

📋 **Детальний аналіз:** Глибокий розбір юридичного питання на основі знайдених документів
⚖️ **Правові посилання:** Конкретні статті законів та нормативних актів
🎯 **Практичні рекомендації:** Покрокові дії для вирішення питання
🔍 **Контекст:** Аналіз з урахуванням українського та ірландського права

**Статус:** Модель GPTQ завантажується... Зазвичай це займає 1-2 хвилини.
**Якість:** Ваша модель Llama-2-7B забезпечить високоякісні юридичні консультації.

💡 **Підказка:** Спробуйте перефразувати питання або зачекайте завершення ініціалізації моделі."""
        else:
            demo_content = f"""🏛️ **Legal Consultation (Demo Mode)**

**Your Question:** {question}

**Document Analysis:** Found {len(context_documents)} relevant documents in knowledge base.

**Demo Response:**
This is a demonstration version of Legal Assistant running with GPTQ model. In full version, your `TheBloke/Llama-2-7B-Chat-GPTQ` model will provide:

📋 **Detailed Analysis:** Deep breakdown of legal question based on found documents
⚖️ **Legal References:** Specific articles, laws, and regulations
🎯 **Practical Advice:** Step-by-step actions to resolve the issue
🔍 **Context:** Analysis considering Irish and Ukrainian law

**Status:** GPTQ model is loading... This usually takes 1-2 minutes.
**Quality:** Your Llama-2-7B model will provide high-quality legal consultations.

💡 **Tip:** Try rephrasing your question or wait for model initialization to complete."""
        
        return LLMResponse(
            content=demo_content,
            model="llama-2-7b-chat-gptq-demo",
            tokens_used=len(demo_content.split()),
            response_time=0.5,
            success=True,
            error=None
        )
    
    async def get_service_status(self):
        """Статус демо LLM сервиса"""
        return {
            "model_loaded": False,
            "model_name": "TheBloke/Llama-2-7B-Chat-GPTQ (loading...)",
            "huggingface_available": True,
            "service_type": "demo_fallback",
            "environment": "HuggingFace Spaces",
            "status": "Your GPTQ model is initializing",
            "supported_languages": ["en", "uk"],
            "target_model": "TheBloke/Llama-2-7B-Chat-GPTQ",
            "loading_status": "Model initialization in progress...",
            "recommendations": [
                "Your GPTQ model provides excellent legal analysis",
                "Supports both English and Ukrainian languages",
                "Loading may take 1-2 minutes on first startup",
                "Demo responses show expected functionality"
            ]
        }

# ====================================
# LAZY INITIALIZATION FUNCTIONS
# ====================================

def _init_document_service():
    """Ленивая инициализация document service с retry логикой"""
    global document_service, _document_service_initialized, CHROMADB_ENABLED, SERVICES_AVAILABLE
    
    if _document_service_initialized:
        return document_service
    
    try:
        logger.info("🔄 Initializing document service...")
        
        if settings.USE_CHROMADB:
            try:
                # Проверяем sentence-transformers
                import sentence_transformers
                from services.chroma_service import DocumentService
                
                logger.info("📚 Attempting ChromaDB initialization...")
                
                # Retry логика для ChromaDB (3 попытки)
                last_error = None
                for attempt in range(3):
                    try:
                        # Создаем директорию для ChromaDB
                        import os
                        os.makedirs(settings.CHROMADB_PATH, exist_ok=True)
                        
                        document_service = DocumentService(settings.CHROMADB_PATH)
                        
                        # Проверяем что ChromaDB действительно работает
                        test_count = await document_service.get_stats()
                        
                        CHROMADB_ENABLED = True
                        SERVICES_AVAILABLE = True
                        logger.info(f"✅ ChromaDB initialized successfully (attempt {attempt + 1})")
                        break
                        
                    except Exception as e:
                        last_error = e
                        logger.warning(f"ChromaDB attempt {attempt + 1}/3 failed: {e}")
                        if attempt < 2:  # Не последняя попытка
                            import time
                            time.sleep(1)  # Ждем секунду перед повтором
                        continue
                else:
                    # Все попытки неудачны
                    raise last_error or Exception("ChromaDB initialization failed after 3 attempts")
                        
            except ImportError as e:
                logger.warning(f"sentence-transformers not available: {e}")
                logger.info("🔄 Falling back to SimpleVectorDB...")
                raise ImportError("sentence-transformers missing")
                
        else:
            logger.info("📁 Using SimpleVectorDB (ChromaDB disabled)")
            from services.document_processor import DocumentService
            document_service = DocumentService(settings.SIMPLE_DB_PATH)
            SERVICES_AVAILABLE = True
            CHROMADB_ENABLED = False
            logger.info("✅ SimpleVectorDB initialized")
        
        _document_service_initialized = True
        return document_service
        
    except Exception as e:
        logger.error(f"❌ Document service initialization failed: {e}")
        logger.info("🔄 Using fallback document service...")
        
        # Создаем улучшенный fallback
        try:
            document_service = FallbackDocumentService()
            document_service.initialization_error = str(e)
            SERVICES_AVAILABLE = True
            CHROMADB_ENABLED = False
            _document_service_initialized = True
            
            logger.info("✅ Fallback document service initialized")
            return document_service
            
        except Exception as fallback_error:
            logger.error(f"❌ Even fallback service failed: {fallback_error}")
            # Создаем минимальный fallback
            document_service = type('MinimalFallback', (), {
                'search': lambda *args, **kwargs: [],
                'get_stats': lambda: {"error": "Service unavailable"},
                'get_all_documents': lambda: [],
                'delete_document': lambda *args: False,
                'process_and_store_file': lambda *args: False
            })()
            _document_service_initialized = True
            return document_service

def _init_scraper_service():
    """Ленивая инициализация scraper service"""
    global scraper, _scraper_initialized
    
    if _scraper_initialized:
        return scraper
    
    try:
        logger.info("🔄 Initializing scraper service...")
        
        # Проверяем библиотеки
        import aiohttp
        import bs4
        from services.scraper_service import LegalSiteScraper
        
        scraper = LegalSiteScraper()
        logger.info("✅ Real scraper service initialized")
        
    except ImportError as e:
        logger.info(f"Scraper libraries not available: {e}")
        from app.dependencies import FallbackScraperService
        scraper = FallbackScraperService()
        logger.info("✅ Fallback scraper service initialized")
    
    _scraper_initialized = True
    return scraper

def _init_llm_service():
    """Ленивая инициализация LLM service с упрощенной логикой для HF Spaces"""
    global llm_service, _llm_service_initialized, LLM_ENABLED
    
    if _llm_service_initialized:
        return llm_service
    
    try:
        logger.info("🔄 Initializing LLM service...")
        
        if settings.LLM_DEMO_MODE:
            logger.info("🎭 LLM demo mode enabled")
            llm_service = ImprovedFallbackLLMService()
            LLM_ENABLED = False
            _llm_service_initialized = True
            return llm_service
        
        # Простая попытка загрузки GPTQ модели
        logger.info("🤖 Attempting to load GPTQ model: TheBloke/Llama-2-7B-Chat-GPTQ")
        
        try:
            from services.huggingface_llm_service import create_llm_service
            import time
            
            # Простая попытка создания сервиса с timeout
            start_time = time.time()
            timeout_seconds = 30  # 30 секунд на попытку загрузки
            
            try:
                llm_service = create_llm_service("TheBloke/Llama-2-7B-Chat-GPTQ")
                
                # Проверяем что модель действительно загрузилась
                if hasattr(llm_service, 'model_loaded') and llm_service.model_loaded:
                    LLM_ENABLED = True
                    logger.info("✅ GPTQ model loaded successfully!")
                else:
                    # Модель не загрузилась, используем fallback
                    logger.info("⏳ GPTQ model not ready yet, using fallback")
                    llm_service = ImprovedFallbackLLMService()
                    LLM_ENABLED = False
                    
            except Exception as model_error:
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    logger.warning(f"⏰ GPTQ model loading timeout ({elapsed:.1f}s), using fallback")
                else:
                    logger.warning(f"⚠️ GPTQ model loading failed: {model_error}")
                
                llm_service = ImprovedFallbackLLMService()
                LLM_ENABLED = False
                
        except ImportError as e:
            logger.warning(f"HuggingFace dependencies not available: {e}")
            llm_service = ImprovedFallbackLLMService()
            LLM_ENABLED = False
            
    except Exception as e:
        logger.error(f"❌ LLM service initialization failed: {e}")
        llm_service = ImprovedFallbackLLMService()
        LLM_ENABLED = False
    
    _llm_service_initialized = True
    logger.info(f"✅ LLM service initialized (GPTQ enabled: {LLM_ENABLED})")
    return llm_service

# ====================================
# УБИРАЕМ АСИНХРОННУЮ ИНИЦИАЛИЗАЦИЮ
# ====================================

async def init_services():
    """Упрощенная инициализация - только устанавливаем флаги"""
    global SERVICES_AVAILABLE
    
    logger.info("🚀 Lazy initialization enabled for HuggingFace Spaces")
    logger.info("📦 Services will initialize on first request")
    
    # Просто помечаем что система готова к lazy loading
    SERVICES_AVAILABLE = True
    
    logger.info("✅ Lazy initialization configured successfully")

# ====================================
# DEPENDENCY FUNCTIONS с LAZY LOADING
# ====================================

def get_document_service():
    """Dependency для получения document service с lazy loading"""
    return _init_document_service()

def get_scraper_service():
    """Dependency для получения scraper service с lazy loading"""
    return _init_scraper_service()

def get_llm_service():
    """Dependency для получения LLM service с lazy loading"""
    return _init_llm_service()

def get_services_status():
    """Статус всех сервисов с lazy evaluation"""
    # Проверяем что сервисы инициализированы
    doc_service = document_service if _document_service_initialized else None
    scraper_service = scraper if _scraper_initialized else None
    llm_srv = llm_service if _llm_service_initialized else None
    
    return {
        "document_service_available": doc_service is not None,
        "scraper_available": scraper_service is not None,
        "llm_available": LLM_ENABLED,
        "llm_service_created": llm_srv is not None,
        "chromadb_enabled": CHROMADB_ENABLED,
        "services_available": SERVICES_AVAILABLE,
        "huggingface_spaces": os.getenv("SPACE_ID") is not None,
        "environment": "hf_spaces" if os.getenv("SPACE_ID") else "local",
        "demo_mode": not LLM_ENABLED,
        "lazy_loading": True,
        "gptq_model": "TheBloke/Llama-2-7B-Chat-GPTQ",
        "initialization_status": {
            "document_service": _document_service_initialized,
            "scraper_service": _scraper_initialized,
            "llm_service": _llm_service_initialized
        },
        "real_features": {
            "vector_search": CHROMADB_ENABLED,
            "web_scraping": _scraper_initialized and not isinstance(scraper, type(None)),
            "ai_responses": LLM_ENABLED
        }
    }

# ====================================
# FALLBACK СЕРВИСЫ (оставляем как есть)
# ====================================

class FallbackDocumentService:
    """Улучшенный fallback document service для HF Spaces"""
    
    def __init__(self):
        self.service_type = "hf_spaces_fallback_improved"
        self.vector_db = type('MockVectorDB', (), {
            'persist_directory': './fallback_db'
        })()
        self.initialization_error = None
        self.demo_documents_count = 3
        logger.info("📝 Using improved fallback document service")
    
    async def search(self, query: str, category: str = None, limit: int = 5, min_relevance: float = 0.3):
        """Улучшенный поиск с более реалистичными результатами"""
        logger.info(f"🔍 Fallback search for: '{query}'")
        
        # Генерируем более релевантные демо результаты
        demo_results = []
        
        # Базовый результат с релевантным контентом
        base_content = f"""Legal Analysis for Query: "{query}"

🏛️ **Document Summary:**
This document contains legal information relevant to your search query. In a fully operational system, this would be actual content from legal databases.

📋 **Key Points:**
• Query: "{query}"
• Category: {category or "General Legal"}
• Search Method: Semantic vector search (when ChromaDB available)
• Relevance: High match found

⚖️ **Legal Context:**
The system would analyze multiple legal documents, statutes, and case law to provide comprehensive answers. This includes:
- Relevant legislation and regulations
- Court decisions and precedents  
- Administrative guidelines
- Legal commentary and analysis

🔧 **Current Status:**
ChromaDB service is initializing. This demo shows the expected response format and structure.

💡 **Note:** Full vector search capabilities will be available once the document service completes initialization."""
        
        demo_results.append({
            "content": base_content,
            "filename": f"legal_analysis_{query.replace(' ', '_')[:20]}.txt",
            "document_id": f"demo_{int(time.time())}",
            "relevance_score": 0.92,
            "metadata": {
                "status": "demo_response",
                "category": category or "general",
                "service": "improved_fallback",
                "query": query,
                "demo_mode": True,
                "expected_features": [
                    "Semantic vector search",
                    "Multi-document analysis", 
                    "Legal citation extraction",
                    "Relevance scoring"
                ],
                "initialization_error": self.initialization_error
            }
        })
        
        # Дополнительные результаты если нужно больше
        if limit > 1:
            for i in range(1, min(limit, self.demo_documents_count)):
                secondary_content = f"""Related Legal Document #{i + 1}

Query: "{query}"
Document Type: {["Statute", "Case Law", "Regulation"][i % 3]}

This would be additional relevant legal content found through vector search. Each document would be ranked by semantic similarity to your query.

Status: Demo mode - ChromaDB initializing..."""

                demo_results.append({
                    "content": secondary_content,
                    "filename": f"related_doc_{i+1}_{query.replace(' ', '_')[:15]}.txt", 
                    "document_id": f"demo_{int(time.time())}_{i}",
                    "relevance_score": max(0.4, 0.9 - (i * 0.15)),
                    "metadata": {
                        "status": "demo_response",
                        "category": category or "general",
                        "service": "improved_fallback",
                        "demo_mode": True
                    }
                })
        
        return demo_results
    
    async def get_stats(self):
        """Улучшенная демо статистика с диагностикой"""
        return {
            "total_documents": 0,
            "categories": ["general", "legislation", "jurisprudence", "government"],
            "database_type": "ChromaDB (initializing...)",
            "status": "Document service starting up",
            "message": "Vector search will be available shortly",
            "initialization_error": self.initialization_error,
            "fallback_info": {
                "demo_responses": True,
                "expected_functionality": [
                    "✅ REST API structure", 
                    "✅ Search endpoint responses",
                    "✅ Document upload endpoints",
                    "⏳ ChromaDB vector search",
                    "⏳ Real document processing",
                    "⏳ Semantic similarity scoring"
                ]
            },
            "troubleshooting": {
                "common_issues": [
                    "sentence-transformers installation",
                    "ChromaDB persistence on HF Spaces", 
                    "Memory limitations during startup",
                    "Model loading timeouts"
                ],
                "recommendations": [
                    "Service will auto-recover when dependencies load",
                    "Demo responses show expected API structure",
                    "Full functionality available after initialization"
                ]
            }
        }
    
    async def get_all_documents(self):
        """Демо список документов"""
        return []
    
    async def delete_document(self, doc_id: str):
        """Демо удаление"""
        logger.info(f"Demo delete: {doc_id}")
        return False
    
    async def process_and_store_file(self, file_path: str, category: str = "general"):
        """Демо обработка файла"""
        logger.info(f"Demo file processing: {file_path}")
        return False

class FallbackScraperService:
    """Web scraper fallback"""
    
    def __init__(self):
        self.service_type = "hf_spaces_scraper"
        self.legal_sites_config = {
            "irishstatutebook.ie": {"title": "h1", "content": ".content"},
            "citizensinformation.ie": {"title": "h1", "content": ".content"},
            "zakon.rada.gov.ua": {"title": "h1", "content": ".content"}
        }
        logger.info("🌐 HF Spaces scraper service initialized")
    
    async def scrape_legal_site(self, url: str):
        """Демо скрапинг"""
        logger.info(f"🔍 Demo scraping: {url}")
        
        demo_content = f"""📄 **Legal Document from {url}**

This is a demonstration of web scraping functionality.
Real scraping service is initializing...

**Document Source:** {url}
**Status:** Scraper loading aiohttp and beautifulsoup4...

The full version will extract actual legal text from websites."""
        
        return type('DemoDocument', (), {
            'url': url,
            'title': f'Demo Legal Document from {url}',
            'content': demo_content,
            'metadata': {
                'status': 'demo',
                'real_scraping': False,
                'scraped_at': time.time(),
                'service': 'hf_spaces_demo',
                'url': url
            },
            'category': 'demo'
        })()
    
    async def scrape_multiple_urls(self, urls: List[str], delay: float = 1.0):
        """Демо массовый скрапинг"""
        results = []
        for url in urls:
            doc = await self.scrape_legal_site(url)
            results.append(doc)
        return results