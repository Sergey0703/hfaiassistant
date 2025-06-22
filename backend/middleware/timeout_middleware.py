# backend/middleware/timeout_middleware.py
"""
Middleware для контроля таймаутов различных типов запросов
"""

import asyncio
import time
import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from config.timeouts import (
    GLOBAL_REQUEST_TIMEOUT, HF_SPACES_HEALTH_TIMEOUT, CHROMADB_SEARCH_TIMEOUT,
    CHROMADB_ADD_DOC_TIMEOUT, SCRAPER_TIMEOUT, GPTQ_MODEL_LOADING_TIMEOUT
)

logger = logging.getLogger(__name__)

def setup_timeout_middleware(app):
    """Настраивает comprehensive middleware для контроля всех типов таймаутов"""
    
    @app.middleware("http")
    async def comprehensive_timeout_middleware(request: Request, call_next):
        """Comprehensive middleware для контроля всех типов таймаутов"""
        start_time = time.time()
        path = str(request.url.path)
        method = request.method
        
        # Определяем специфичный таймаут для разных типов запросов
        timeout = _get_timeout_for_request(path, method)
        request_type = _classify_request_type(path, method)
        
        try:
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Оборачиваем каждый запрос в соответствующий таймаут
            response = await asyncio.wait_for(
                call_next(request), 
                timeout=timeout
            )
            
            process_time = time.time() - start_time
            
            # Добавляем заголовки с информацией о таймауте
            response.headers["X-Process-Time"] = str(round(process_time, 3))
            response.headers["X-Timeout-Limit"] = str(timeout)
            response.headers["X-Request-Type"] = request_type
            
            # Логируем долгие запросы с учетом типа
            _log_request_time(method, path, process_time, timeout, request_type)
            
            return response
            
        except asyncio.TimeoutError:
            process_time = time.time() - start_time
            
            logger.error(f"❌ {request_type} timeout: {method} {path} after {process_time:.2f}s (limit: {timeout}s)")
            
            return JSONResponse(
                status_code=408,  # Request Timeout
                content={
                    "detail": f"{request_type} timeout after {timeout} seconds",
                    "path": path,
                    "method": method,
                    "timeout_limit": timeout,
                    "actual_time": round(process_time, 2),
                    "request_type": request_type,
                    "suggestion": _get_timeout_suggestion(request_type, timeout),
                    "platform": "HuggingFace Spaces"
                }
            )
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"❌ Request error: {method} {path} after {process_time:.2f}s: {e}")
            raise

def _get_timeout_for_request(path: str, method: str) -> int:
    """Определяет таймаут для конкретного запроса"""
    
    if "/api/admin/llm" in path and "status" in path:
        return HF_SPACES_HEALTH_TIMEOUT  # 15s для статуса LLM
    elif "/api/user/chat" in path:
        return GLOBAL_REQUEST_TIMEOUT    # 10min для чата с GPTQ
    elif "/api/user/search" in path:
        return CHROMADB_SEARCH_TIMEOUT + 30  # 60s для поиска
    elif "/api/admin/documents" in path and method == "POST":
        return CHROMADB_ADD_DOC_TIMEOUT + 30  # 90s для добавления документов
    elif "/api/admin/scraper" in path:
        return SCRAPER_TIMEOUT + 30       # 90s для парсинга
    elif "/model-status" in path:
        return GPTQ_MODEL_LOADING_TIMEOUT # 8min для статуса GPTQ модели
    elif "/hf-spaces-health" in path:
        return HF_SPACES_HEALTH_TIMEOUT  # 15s для health check
    elif "/startup-progress" in path:
        return HF_SPACES_HEALTH_TIMEOUT  # 15s для прогресса
    elif "/timeout-status" in path:
        return HF_SPACES_HEALTH_TIMEOUT  # 15s для статуса таймаутов
    elif "/memory-status" in path:
        return HF_SPACES_HEALTH_TIMEOUT  # 15s для статуса памяти
    elif path.startswith("/static/") or path in ["/", "/favicon.ico", "/manifest.json"]:
        return 30  # 30s для статических файлов React
    else:
        return GLOBAL_REQUEST_TIMEOUT    # 10min по умолчанию

def _classify_request_type(path: str, method: str) -> str:
    """Классифицирует тип запроса для лучшего сообщения об ошибке"""
    if "/api/user/chat" in path:
        return "GPTQ Chat Request"
    elif "/api/user/search" in path:
        return "ChromaDB Search"
    elif "/api/admin/documents" in path and method == "POST":
        return "Document Upload"
    elif "/api/admin/scraper" in path:
        return "Web Scraping"
    elif "/model-status" in path:
        return "GPTQ Model Status"
    elif "/hf-spaces-health" in path:
        return "Health Check"
    elif "/timeout-status" in path:
        return "Timeout Status"
    elif "/memory-status" in path:
        return "Memory Status"
    elif "/startup-progress" in path:
        return "Startup Progress"
    elif path.startswith("/static/") or path in ["/", "/favicon.ico", "/manifest.json"]:
        return "React Static File"
    else:
        return "API Request"

def _get_timeout_suggestion(request_type: str, timeout: int) -> str:
    """Предоставляет специфичные рекомендации по таймауту"""
    suggestions = {
        "GPTQ Chat Request": "Try with a shorter question. GPTQ model is loading or generating response.",
        "ChromaDB Search": "Use more specific keywords. Try simpler search terms.",
        "Document Upload": "Upload smaller files. Large documents may take time to process.",
        "Web Scraping": "The target website may be slow or unreachable.",
        "GPTQ Model Status": "GPTQ model is loading in background. Check /startup-progress for details.",
        "Health Check": "System may be under heavy load. Try again in a moment.",
        "Timeout Status": "System monitoring timeout. Check system health.",
        "Memory Status": "Memory monitoring timeout. System may be busy.",
        "Startup Progress": "Startup monitoring timeout. Services may still be initializing.",
        "React Static File": "Static file serving timeout. Check React build."
    }
    return suggestions.get(request_type, f"Request exceeded {timeout}s limit. Try a simpler operation.")

def _log_request_time(method: str, path: str, process_time: float, timeout: int, request_type: str):
    """Логирует время выполнения запроса"""
    
    if process_time > 60:  # Более 1 минуты
        logger.warning(f"⏰ Slow request: {method} {path} took {process_time:.2f}s (limit: {timeout}s, type: {request_type})")
    elif process_time > 30:  # Более 30 секунд
        logger.info(f"⏰ Long request: {method} {path} took {process_time:.2f}s (type: {request_type})")
    elif process_time > 10:  # Более 10 секунд
        logger.debug(f"⏰ Medium request: {method} {path} took {process_time:.2f}s (type: {request_type})")

# ====================================
# СПЕЦИАЛЬНЫЕ ENDPOINTS ДЛЯ МОНИТОРИНГА ТАЙМАУТОВ
# ====================================

def setup_timeout_endpoints(app):
    """Настраивает специальные endpoints для мониторинга таймаутов"""
    
    @app.get("/timeout-status")
    async def comprehensive_timeout_status():
        """Comprehensive endpoint для мониторинга всех таймаутов"""
        from config.timeouts import get_timeout_config
        
        return {
            "timeout_configuration": get_timeout_config(),
            "timeout_recommendations": {
                "chat_queries": "Keep questions concise for faster GPTQ responses",
                "document_uploads": "Upload smaller files if timeouts occur",
                "search_queries": "Use specific keywords rather than long phrases",
                "admin_operations": "Large operations may take time - be patient",
                "gptq_model": "First load may take 8+ minutes, subsequent loads faster",
                "chromadb_operations": "Large vector operations limited by 16GB RAM",
                "react_static": "Static files should load quickly, check React build"
            },
            "platform_limits": {
                "hf_spaces_memory": "16GB RAM limit",
                "hf_spaces_cpu": "2 CPU cores",
                "hf_spaces_disk": "50GB temporary storage",
                "model_size": "TheBloke/Llama-2-7B-Chat-GPTQ ~4GB quantized"
            },
            "optimization_status": {
                "memory_optimized": True,
                "timeout_middleware": "active",
                "cors_fixed": True,
                "lazy_loading": True,
                "background_loading": True,
                "react_spa": True
            },
            "request_type_timeouts": {
                "gptq_chat": GLOBAL_REQUEST_TIMEOUT,
                "chromadb_search": CHROMADB_SEARCH_TIMEOUT + 30,
                "document_upload": CHROMADB_ADD_DOC_TIMEOUT + 30,
                "web_scraping": SCRAPER_TIMEOUT + 30,
                "model_status": GPTQ_MODEL_LOADING_TIMEOUT,
                "health_check": HF_SPACES_HEALTH_TIMEOUT,
                "react_static": 30
            }
        }

# ====================================
# MIDDLEWARE REGISTRY
# ====================================

def setup_all_timeout_middleware(app):
    """Настраивает все timeout-related middleware и endpoints"""
    
    logger.info("🔧 Setting up comprehensive timeout middleware...")
    
    # Основной timeout middleware
    setup_timeout_middleware(app)
    
    # Endpoints для мониторинга
    setup_timeout_endpoints(app)
    
    logger.info("✅ Timeout middleware configured:")
    logger.info(f"   • Global timeout: {GLOBAL_REQUEST_TIMEOUT}s")
    logger.info(f"   • Health check timeout: {HF_SPACES_HEALTH_TIMEOUT}s") 
    logger.info(f"   • GPTQ model timeout: {GPTQ_MODEL_LOADING_TIMEOUT}s")
    logger.info(f"   • ChromaDB search timeout: {CHROMADB_SEARCH_TIMEOUT}s")
    logger.info("   • Request type classification: enabled")
    logger.info("   • Timeout monitoring endpoints: /timeout-status")