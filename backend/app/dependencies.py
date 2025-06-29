# backend/app/dependencies.py - УПРОЩЕННЫЕ ЗАВИСИМОСТИ
"""
Минимальные зависимости для RAG системы
Убрана сложная логика, оставлены только критические сервисы
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
_llm_service: Optional[object] = None

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
    
    return {
        # Основные статусы
        "document_service_available": doc_service is not None,
        "llm_available": llm_service is not None and getattr(llm_service, 'ready', False),
        
        # Типы сервисов
        "document_service_type": getattr(doc_service, 'service_type', 'empty'),
        "llm_service_type": getattr(llm_service, 'service_type', 'unknown'),
        
        # Простые флаги
        "chromadb_enabled": _is_chromadb_enabled(),
        "huggingface_spaces": os.getenv("SPACE_ID") is not None,
        
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
            llm_service is not None
        ]),
        
        # Модели
        "llm_model": "google/flan-t5-small",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "memory_estimate": "~920 MB"
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
    
    return EmptyDocumentService()

def _create_fallback_llm_service():
    """Создаёт fallback для LLM"""
    
    class FallbackLLMService:
        def __init__(self):
            self.service_type = "llm_fallback"
            self.ready = True
        
        async def answer_legal_question(self, question: str, context_documents: list, language: str = "en"):
            # Простая структура ответа
            class SimpleResponse:
                def __init__(self, content, model, tokens_used, response_time, success):
                    self.content = content
                    self.model = model
                    self.tokens_used = tokens_used
                    self.response_time = response_time
                    self.success = success
            
            if language == "uk":
                content = f"""🤖 **FLAN-T5 сервіс недоступний**

**Ваше питання:** {question}

❌ На жаль, FLAN-T5 Small модель наразі недоступна.

💡 **Рекомендації:**
• Спробуйте ще раз через кілька хвилин
• Перевірте підключення до інтернету
• Зверніться до адміністратора системи"""
            else:
                content = f"""🤖 **FLAN-T5 Service Unavailable**

**Your Question:** {question}

❌ Unfortunately, the FLAN-T5 Small model is currently unavailable.

💡 **Recommendations:**
• Try again in a few minutes
• Check your internet connection
• Contact system administrator"""
            
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
                "error": "FLAN-T5 service not available"
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

# ====================================
# ЭКСПОРТ
# ====================================

__all__ = [
    "get_document_service",
    "get_llm_service", 
    "get_services_status",
    "init_services",
    "SERVICES_AVAILABLE",
    "CHROMADB_ENABLED"
]