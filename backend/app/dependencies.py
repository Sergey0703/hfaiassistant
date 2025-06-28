# backend/app/dependencies.py - ИСПРАВЛЕННАЯ ЛОГИКА
"""
ИСПРАВЛЕНО: Правильная логика fallback - LLM отвечает из знаний если нет документов
"""

import logging
import os
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# ====================================
# ГЛОБАЛЬНЫЕ СЕРВИСЫ
# ====================================

_document_service: Optional[object] = None
_scraper_service: Optional[object] = None
_llm_service: Optional[object] = None

# Простые флаги состояния
_initialization_errors = {}

# ====================================
# DEPENDENCY FUNCTIONS
# ====================================

def get_document_service():
    """Получает document service с правильной логикой fallback"""
    global _document_service
    
    if _document_service is None:
        logger.info("🔄 Initializing document service...")
        
        try:
            # Пробуем ChromaDB
            use_chromadb = os.getenv("USE_CHROMADB", "true").lower() == "true"
            
            if use_chromadb:
                try:
                    # ИСПРАВЛЕНИЕ: Добавлен try-except для импорта
                    try:
                        from services.chroma_service import DocumentService
                    except ImportError:
                        logger.warning("ChromaDB service not available, using empty document service")
                        _document_service = _create_empty_document_service()
                        return _document_service
                    
                    chromadb_path = os.getenv("CHROMADB_PATH", "./chromadb_data")
                    os.makedirs(chromadb_path, exist_ok=True)
                    
                    _document_service = DocumentService(chromadb_path)
                    logger.info("✅ ChromaDB document service initialized")
                    
                except ImportError as e:
                    logger.warning(f"ChromaDB not available: {e}, using empty document service")
                    _document_service = _create_empty_document_service()
                    
            else:
                logger.info("ChromaDB disabled, using empty document service")
                _document_service = _create_empty_document_service()
                
        except Exception as e:
            logger.error(f"❌ Document service initialization failed: {e}")
            _initialization_errors['document_service'] = str(e)
            # ИСПРАВЛЕНО: Используем пустой сервис вместо fallback
            _document_service = _create_empty_document_service()
    
    return _document_service

def get_scraper_service():
    """Получает scraper service"""
    global _scraper_service
    
    if _scraper_service is None:
        logger.info("🔄 Initializing scraper service...")
        
        try:
            from services.scraper_service import LegalSiteScraper
            _scraper_service = LegalSiteScraper()
            logger.info("✅ Scraper service initialized")
            
        except ImportError as e:
            logger.warning(f"Scraper dependencies not available: {e}")
            _initialization_errors['scraper_service'] = str(e)
            _scraper_service = _create_simple_scraper_fallback()
            
        except Exception as e:
            logger.error(f"❌ Scraper service initialization failed: {e}")
            _initialization_errors['scraper_service'] = str(e)
            _scraper_service = _create_simple_scraper_fallback()
    
    return _scraper_service

def get_llm_service():
    """Получает LLM service"""
    global _llm_service
    
    if _llm_service is None:
        logger.info("🔄 Initializing Llama LLM service...")
        
        try:
            # ИСПРАВЛЕНИЕ: Добавлен try-except для импорта create_llama_service
            try:
                from services.llama_service import create_llama_service
                _llm_service = create_llama_service()
                logger.info("✅ Llama LLM service initialized")
            except ImportError:
                logger.warning("create_llama_service not available, using fallback")
                _llm_service = _create_simple_llm_fallback()
            
        except Exception as e:
            logger.error(f"❌ LLM service initialization failed: {e}")
            _initialization_errors['llm_service'] = str(e)
            _llm_service = _create_simple_llm_fallback()
    
    return _llm_service

def get_services_status() -> Dict[str, Any]:
    """Простой статус всех сервисов"""
    # Инициализируем сервисы если ещё не сделали
    doc_service = get_document_service()
    scraper_service = get_scraper_service() 
    llm_service = get_llm_service()
    
    return {
        # Основные статусы
        "document_service_available": doc_service is not None,
        "scraper_available": scraper_service is not None,
        "llm_available": llm_service is not None and getattr(llm_service, 'ready', False),
        
        # Типы сервисов
        "document_service_type": getattr(doc_service, 'service_type', 'unknown'),
        "scraper_service_type": getattr(scraper_service, 'service_type', 'unknown'),
        "llm_service_type": getattr(llm_service, 'service_type', 'unknown'),
        
        # Простые флаги
        "chromadb_enabled": _is_chromadb_enabled(),
        "huggingface_spaces": os.getenv("SPACE_ID") is not None,
        "demo_mode": _is_demo_mode(),
        
        # Ошибки инициализации
        "initialization_errors": _initialization_errors,
        "total_errors": len(_initialization_errors),
        
        # Окружение
        "environment": "hf_spaces" if os.getenv("SPACE_ID") else "local",
        "platform": "HuggingFace Spaces" if os.getenv("SPACE_ID") else "Local Development",
        
        # Время статуса
        "status_time": time.time(),
        "services_ready": all([
            doc_service is not None,
            scraper_service is not None, 
            llm_service is not None
        ])
    }

# ====================================
# ИСПРАВЛЕННЫЕ СЕРВИСЫ
# ====================================

def _create_empty_document_service():
    """ИСПРАВЛЕНО: Создаёт ПУСТОЙ сервис документов (не fallback!)"""
    
    class EmptyDocumentService:
        def __init__(self):
            self.service_type = "empty_document_service"
            
        async def search(self, query: str, category: str = None, limit: int = 5, min_relevance: float = 0.3):
            """ИСПРАВЛЕНО: Возвращает пустой список (не fallback сообщения!)"""
            logger.debug(f"Empty document service: no results for '{query}'")
            return []  # Пустой список позволяет LLM отвечать из своих знаний
        
        async def get_stats(self):
            """Простая статистика"""
            return {
                "total_documents": 0,
                "categories": ["general", "legislation", "jurisprudence"],
                "database_type": "Empty Document Service",
                "status": "no_documents_available"
            }
        
        async def get_all_documents(self):
            return []
        
        async def delete_document(self, doc_id: str):
            return False
        
        async def process_and_store_file(self, file_path: str, category: str = "general"):
            logger.warning("Cannot store files: document service not available")
            return False
    
    return EmptyDocumentService()

def _create_simple_scraper_fallback():
    """Создаёт простой fallback для скрапера"""
    
    class SimpleScraperFallback:
        def __init__(self):
            self.service_type = "scraper_fallback"
            self.legal_sites_config = {}
        
        async def scrape_legal_site(self, url: str):
            """Простой fallback скрапинг"""
            return type('SimpleDoc', (), {
                'url': url,
                'title': f'Demo content from {url}',
                'content': f'🌐 Web scraping service is initializing. Content from {url} will be available once the service is ready.',
                'metadata': {
                    'status': 'fallback_mode',
                    'url': url,
                    'service_type': 'scraper_fallback'
                },
                'category': 'demo'
            })()
        
        async def scrape_multiple_urls(self, urls, delay=1.0):
            results = []
            for url in urls:
                doc = await self.scrape_legal_site(url)
                results.append(doc)
            return results
    
    return SimpleScraperFallback()

def _create_simple_llm_fallback():
    """Создаёт fallback для LLM ТОЛЬКО когда LLM недоступен"""
    
    class SimpleLLMFallback:
        def __init__(self):
            self.service_type = "llm_fallback"
            self.ready = True
        
        async def answer_legal_question(self, question: str, context_documents: list, language: str = "en"):
            """FALLBACK: Только когда LLM сервис полностью недоступен"""
            # Создаём простую структуру ответа без импортов
            class SimpleResponse:
                def __init__(self, content, model, tokens_used, response_time, success):
                    self.content = content
                    self.model = model
                    self.tokens_used = tokens_used
                    self.response_time = response_time
                    self.success = success
            
            if language == "uk":
                content = f"""🤖 **LLM сервіс тимчасово недоступний**

**Ваше питання:** {question}

❌ На жаль, Llama-3.1-8B-Instruct сервіс наразі недоступний.

💡 **Рекомендації:**
• Спробуйте ще раз через кілька хвилин
• Перевірте підключення до інтернету
• Зверніться до адміністратора системи

🔧 **Для адміністратора:** Перевірте налаштування LLM сервісу"""
            else:
                content = f"""🤖 **LLM Service Temporarily Unavailable**

**Your Question:** {question}

❌ Unfortunately, the Llama-3.1-8B-Instruct service is currently unavailable.

💡 **Recommendations:**
• Try again in a few minutes
• Check your internet connection
• Contact system administrator

🔧 **For administrator:** Check LLM service configuration"""
            
            return SimpleResponse(
                content=content,
                model="llm_fallback",
                tokens_used=len(content.split()),
                response_time=0.1,
                success=True
            )
        
        async def get_service_status(self):
            return {
                "service_type": "llm_fallback",
                "ready": True,
                "initialization_error": "LLM service not available"
            }
    
    return SimpleLLMFallback()

# ====================================
# UTILITY FUNCTIONS
# ====================================

def _is_chromadb_enabled() -> bool:
    """Проверяет включён ли ChromaDB"""
    if _document_service is None:
        return False
    return getattr(_document_service, 'service_type', '') not in ['empty_document_service', 'simple_fallback']

def _is_demo_mode() -> bool:
    """Проверяет режим демо"""
    demo_env = os.getenv("LLM_DEMO_MODE", "false").lower()
    return demo_env in ["true", "1", "yes"] or bool(_initialization_errors.get('llm_service'))

# ====================================
# СОВМЕСТИМОСТЬ
# ====================================

# Константы для совместимости с существующим кодом
SERVICES_AVAILABLE = True
CHROMADB_ENABLED = True

# Функция для совместимости
async def init_services():
    """Функция для совместимости - сервисы инициализируются сразу"""
    logger.info("📦 Services initialize on first use")
    return True

# Экспорт основных функций
__all__ = [
    "get_document_service",
    "get_scraper_service", 
    "get_llm_service",
    "get_services_status",
    "init_services",
    "SERVICES_AVAILABLE",
    "CHROMADB_ENABLED"
]