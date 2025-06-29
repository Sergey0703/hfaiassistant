# backend/app/dependencies.py - ПОЛНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ
"""
Зависимости для RAG системы с поддержкой всех сервисов
ИСПРАВЛЕНИЯ: Добавлены все недостающие импорты и функции
"""

import logging
import os
import time
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# ====================================
# ГЛОБАЛЬНЫЕ СЕРВИСЫ
# ====================================

_document_service: Optional[object] = None
_llm_service: Optional[object] = None
_scraper_service: Optional[object] = None

# Простые флаги состояния
_initialization_errors = {}

# ====================================
# ОСНОВНЫЕ DEPENDENCY FUNCTIONS
# ====================================

def get_document_service():
    """Получает document service"""
    global _document_service
    
    if _document_service is None:
        logger.info("🔄 Initializing document service...")
        
        try:
            # Пробуем ChromaDB
            use_chromadb = os.getenv("USE_CHROMADB", "true").lower() == "true"
            
            if use_chromadb:
                try:
                    from services.chroma_service import DocumentService
                    chromadb_path = os.getenv("CHROMADB_PATH", "./chromadb_data")
                    os.makedirs(chromadb_path, exist_ok=True)
                    
                    _document_service = DocumentService(chromadb_path)
                    logger.info("✅ ChromaDB document service initialized")
                    
                except ImportError as e:
                    logger.warning(f"ChromaDB not available: {e}")
                    _document_service = _create_empty_document_service()
                    
            else:
                logger.info("ChromaDB disabled")
                _document_service = _create_empty_document_service()
                
        except Exception as e:
            logger.error(f"❌ Document service initialization failed: {e}")
            _initialization_errors['document_service'] = str(e)
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
            
        except Exception as e:
            logger.error(f"❌ Scraper service initialization failed: {e}")
            _initialization_errors['scraper_service'] = str(e)
            _scraper_service = _create_fallback_scraper_service()
    
    return _scraper_service

def get_llm_service():
    """Получает FLAN-T5 LLM service"""
    global _llm_service
    
    if _llm_service is None:
        logger.info("🔄 Initializing FLAN-T5 service...")
        
        try:
            from services.flan_t5_service import create_flan_t5_service
            _llm_service = create_flan_t5_service()
            logger.info("✅ FLAN-T5 service initialized")
            
        except Exception as e:
            logger.error(f"❌ FLAN-T5 service initialization failed: {e}")
            _initialization_errors['llm_service'] = str(e)
            _llm_service = _create_fallback_llm_service()
    
    return _llm_service

def get_services_status() -> Dict[str, Any]:
    """Возвращает простой статус всех сервисов"""
    # Инициализируем сервисы если ещё не сделали
    doc_service = get_document_service()
    llm_service = get_llm_service()
    scraper_service = get_scraper_service()
    
    return {
        # Основные статусы
        "document_service_available": doc_service is not None,
        "llm_available": llm_service is not None and getattr(llm_service, 'ready', False),
        "scraper_available": scraper_service is not None,
        
        # Типы сервисов
        "document_service_type": getattr(doc_service, 'service_type', 'empty'),
        "llm_service_type": getattr(llm_service, 'service_type', 'unknown'),
        "scraper_service_type": getattr(scraper_service, 'service_type', 'unknown'),
        
        # Простые флаги
        "chromadb_enabled": _is_chromadb_enabled(),
        "huggingface_spaces": os.getenv("SPACE_ID") is not None,
        "scraping_enabled": getattr(scraper_service, 'service_type', '') != 'scraper_fallback',
        
        # Ошибки инициализации
        "initialization_errors": _initialization_errors,
        "total_errors": len(_initialization_errors),
        
        # Окружение
        "environment": "hf_spaces" if os.getenv("SPACE_ID") else "local",
        "platform": "HuggingFace Spaces" if os.getenv("SPACE_ID") else "Local",
        
        # Время статуса
        "status_time": time.time(),
        "services_ready": all([
            doc_service is not None,
            llm_service is not None,
            scraper_service is not None
        ]),
        
        # Модели
        "llm_model": "google/flan-t5-small",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "memory_estimate": "~920 MB",
        
        # Демо режим
        "demo_mode": os.getenv("LLM_DEMO_MODE", "false").lower() == "true"
    }

# ====================================
# FALLBACK СЕРВИСЫ
# ====================================

def _create_empty_document_service():
    """Создаёт пустой сервис документов"""
    
    class EmptyDocumentService:
        def __init__(self):
            self.service_type = "empty_document_service"
            
        async def search(self, query: str, category: str = None, limit: int = 5, min_relevance: float = 0.3):
            """Возвращает пустой список"""
            logger.debug(f"Empty document service: no results for '{query}'")
            return []
        
        async def get_stats(self):
            return {
                "total_documents": 0,
                "categories": ["general", "legislation", "jurisprudence"],
                "database_type": "Empty Service",
                "status": "no_documents_available"
            }
        
        async def get_all_documents(self):
            return []
        
        async def delete_document(self, doc_id: str):
            return False
        
        async def process_and_store_file(self, file_path: str, category: str = "general"):
            logger.warning("Cannot store files: document service not available")
            return False
        
        async def update_document(self, doc_id: str, new_content: str = None, new_metadata: Dict = None):
            return False
    
    return EmptyDocumentService()

def _create_fallback_scraper_service():
    """Создаёт fallback для scraper"""
    
    class FallbackScraperService:
        def __init__(self):
            self.service_type = "scraper_fallback"
            self.legal_sites_config = {
                "zakon.rada.gov.ua": {
                    "title": "h1",
                    "content": ".content",
                    "exclude": "nav, footer"
                },
                "irishstatutebook.ie": {
                    "title": "h1",
                    "content": ".content",
                    "exclude": "nav, footer"
                }
            }
        
        async def scrape_legal_site(self, url: str):
            """Fallback scraping - возвращает заглушку"""
            logger.warning(f"Scraper fallback mode: cannot scrape {url}")
            
            # Создаем фиктивный документ
            from dataclasses import dataclass
            from typing import Dict, Any
            
            @dataclass
            class ScrapedDocument:
                url: str
                title: str
                content: str
                metadata: Dict[str, Any]
                category: str = "scraped"
            
            return ScrapedDocument(
                url=url,
                title="Scraper Service Unavailable",
                content=f"""🔧 **Scraper Service в режиме fallback**

URL: {url}

⚠️ Реальный парсинг недоступен. Возможные причины:
• Отсутствуют библиотеки requests/beautifulsoup4
• Проблемы с сетью
• Сервис временно недоступен

💡 Для включения реального парсинга:
• Установите: pip install requests beautifulsoup4
• Перезапустите сервер
• Проверьте подключение к интернету

Этот документ является заглушкой и не содержит реального контента.""",
                metadata={
                    "scraped_at": time.time(),
                    "real_scraping": False,
                    "fallback_mode": True,
                    "error": "Scraper service not available",
                    "recommendations": [
                        "Install scraping dependencies",
                        "Check network connectivity",
                        "Restart the service"
                    ]
                },
                category="scraped"
            )
        
        async def scrape_multiple_urls(self, urls: list, delay: float = 1.0):
            """Fallback для массового парсинга"""
            results = []
            for url in urls:
                doc = await self.scrape_legal_site(url)
                results.append(doc)
            return results
        
        async def validate_url(self, url: str):
            """Простая валидация URL"""
            return {
                "url": url,
                "valid": url.startswith(('http://', 'https://')),
                "reachable": False,
                "error": "Scraper service in fallback mode",
                "recommendations": [
                    "Install scraping dependencies",
                    "Check real scraper service status"
                ]
            }
        
        def get_supported_sites(self):
            """Возвращает поддерживаемые сайты"""
            return {
                "sites": list(self.legal_sites_config.keys()),
                "total": len(self.legal_sites_config),
                "real_scraping_available": False,
                "fallback_mode": True,
                "message": "Scraper service in fallback mode"
            }
        
        async def close(self):
            """Закрытие сервиса"""
            logger.debug("🔒 Fallback scraper service cleanup completed")
    
    return FallbackScraperService()

def _create_fallback_llm_service():
    """Создаёт fallback для LLM"""
    
    class FallbackLLMService:
        def __init__(self):
            self.service_type = "llm_fallback"
            self.ready = True
        
        async def answer_legal_question(self, question: str, context_documents: list, language: str = "en"):
            # Простая структура ответа
            class SimpleResponse:
                def __init__(self, content, model, tokens_used, response_time, success, error=None):
                    self.content = content
                    self.model = model
                    self.tokens_used = tokens_used
                    self.response_time = response_time
                    self.success = success
                    self.error = error
            
            if language == "uk":
                content = f"""🤖 **FLAN-T5 сервіс недоступний**

**Ваше питання:** {question}

❌ На жаль, FLAN-T5 Small модель наразі недоступна.

📚 **Знайдено документів:** {len(context_documents)}

💡 **Рекомендації:**
• Спробуйте ще раз через кілька хвилин
• Перевірте підключення до інтернету
• Зверніться до адміністратора системи

🔧 **Для відновлення AI:**
• Перевірте наявність transformers
• Встановіть HF_TOKEN якщо потрібно
• Перезапустіть сервер"""
            else:
                content = f"""🤖 **FLAN-T5 Service Unavailable**

**Your Question:** {question}

❌ Unfortunately, the FLAN-T5 Small model is currently unavailable.

📚 **Documents Found:** {len(context_documents)}

💡 **Recommendations:**
• Try again in a few minutes
• Check your internet connection
• Contact system administrator

🔧 **To restore AI:**
• Check transformers installation
• Set HF_TOKEN if needed
• Restart the server"""
            
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
                "error": "FLAN-T5 service not available",
                "recommendations": [
                    "Check transformers installation",
                    "Verify HF_TOKEN configuration",
                    "Check internet connectivity",
                    "Restart the service"
                ]
            }
    
    return FallbackLLMService()

# ====================================
# UTILITY FUNCTIONS
# ====================================

def _is_chromadb_enabled() -> bool:
    """Проверяет включён ли ChromaDB"""
    if _document_service is None:
        return False
    return getattr(_document_service, 'service_type', '') not in ['empty_document_service']

def get_all_services():
    """Возвращает все инициализированные сервисы"""
    return {
        "document": get_document_service(),
        "llm": get_llm_service(),
        "scraper": get_scraper_service()
    }

def check_services_health():
    """Проверяет здоровье всех сервисов"""
    services = get_all_services()
    health_status = {}
    
    for service_name, service in services.items():
        try:
            service_type = getattr(service, 'service_type', 'unknown')
            is_fallback = 'fallback' in service_type or 'empty' in service_type
            
            health_status[service_name] = {
                "available": service is not None,
                "type": service_type,
                "is_fallback": is_fallback,
                "status": "degraded" if is_fallback else "healthy"
            }
            
        except Exception as e:
            health_status[service_name] = {
                "available": False,
                "error": str(e),
                "status": "error"
            }
    
    # Общий статус
    overall_status = "healthy"
    fallback_count = sum(1 for status in health_status.values() if status.get("is_fallback", False))
    error_count = sum(1 for status in health_status.values() if status.get("status") == "error")
    
    if error_count > 0:
        overall_status = "error"
    elif fallback_count > 0:
        overall_status = "degraded"
    
    return {
        "overall_status": overall_status,
        "services": health_status,
        "summary": {
            "total_services": len(health_status),
            "healthy_services": len([s for s in health_status.values() if s.get("status") == "healthy"]),
            "fallback_services": fallback_count,
            "error_services": error_count
        },
        "recommendations": _get_health_recommendations(health_status)
    }

def _get_health_recommendations(health_status: Dict) -> List[str]:
    """Генерирует рекомендации по улучшению здоровья сервисов"""
    recommendations = []
    
    for service_name, status in health_status.items():
        if status.get("is_fallback", False):
            if service_name == "document":
                recommendations.append("Install ChromaDB for better document storage")
            elif service_name == "llm":
                recommendations.append("Check transformers installation and HF_TOKEN")
            elif service_name == "scraper":
                recommendations.append("Install requests and beautifulsoup4 for web scraping")
        elif status.get("status") == "error":
            recommendations.append(f"Fix {service_name} service errors - check logs")
    
    if not recommendations:
        recommendations.append("All services are running optimally")
    
    return recommendations

# ====================================
# КОНФИГУРАЦИОННЫЕ ФУНКЦИИ
# ====================================

def get_llm_config() -> dict:
    """Возвращает конфигурацию LLM"""
    try:
        from app.config import settings
        return {
            "model": settings.LLM_MODEL,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "temperature": settings.LLM_TEMPERATURE,
            "timeout": settings.LLM_TIMEOUT,
            "hf_token_configured": bool(settings.HF_TOKEN),
            "model_type": "text2text-generation"
        }
    except Exception as e:
        logger.error(f"Error getting LLM config: {e}")
        return {
            "model": "google/flan-t5-small",
            "max_tokens": 150,
            "temperature": 0.3,
            "timeout": 20,
            "hf_token_configured": False,
            "model_type": "text2text-generation",
            "error": str(e)
        }

def validate_llm_config() -> dict:
    """Валидирует конфигурацию LLM"""
    try:
        from app.config import validate_config
        return validate_config()
    except Exception as e:
        logger.error(f"Error validating LLM config: {e}")
        return {
            "valid": False,
            "issues": [f"Configuration validation failed: {e}"],
            "warnings": [],
            "memory_estimate": "~920 MB total"
        }

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
    
    # Проверяем инициализацию всех сервисов
    services = get_all_services()
    health = check_services_health()
    
    logger.info(f"🏥 Services health check: {health['overall_status']}")
    logger.info(f"   Healthy: {health['summary']['healthy_services']}")
    logger.info(f"   Fallback: {health['summary']['fallback_services']}")
    logger.info(f"   Errors: {health['summary']['error_services']}")
    
    return True

# ====================================
# ДОПОЛНИТЕЛЬНЫЕ UTILITY ФУНКЦИИ
# ====================================

def get_memory_usage_estimate() -> Dict[str, Any]:
    """Оценивает использование памяти сервисами"""
    return {
        "flan_t5_small": "~300 MB",
        "sentence_transformers": "~90 MB",
        "chromadb": "~20 MB",
        "fastapi": "~50 MB",
        "python_runtime": "~100 MB",
        "total_estimated": "~560 MB",
        "target": "<1GB RAM",
        "efficiency": "56% of 1GB target"
    }

def get_platform_info() -> Dict[str, Any]:
    """Информация о платформе"""
    is_hf_spaces = os.getenv("SPACE_ID") is not None
    
    return {
        "platform": "HuggingFace Spaces" if is_hf_spaces else "Local",
        "is_hf_spaces": is_hf_spaces,
        "space_id": os.getenv("SPACE_ID"),
        "python_version": os.sys.version.split()[0],
        "environment_variables": {
            "USE_CHROMADB": os.getenv("USE_CHROMADB", "true"),
            "LLM_MODEL": os.getenv("LLM_MODEL", "google/flan-t5-small"),
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
            "LLM_DEMO_MODE": os.getenv("LLM_DEMO_MODE", "false")
        }
    }

def reset_services():
    """Сбрасывает все сервисы (для отладки)"""
    global _document_service, _llm_service, _scraper_service, _initialization_errors
    
    _document_service = None
    _llm_service = None
    _scraper_service = None
    _initialization_errors.clear()
    
    logger.info("🔄 All services reset")

# ====================================
# ЭКСПОРТ
# ====================================

__all__ = [
    # Основные сервисы
    "get_document_service",
    "get_llm_service", 
    "get_scraper_service",
    "get_services_status",
    "get_all_services",
    
    # Здоровье и диагностика
    "check_services_health",
    "init_services",
    
    # Конфигурация
    "get_llm_config",
    "validate_llm_config",
    
    # Дополнительные утилиты
    "get_memory_usage_estimate",
    "get_platform_info",
    "reset_services",
    
    # Константы
    "SERVICES_AVAILABLE",
    "CHROMADB_ENABLED"
]