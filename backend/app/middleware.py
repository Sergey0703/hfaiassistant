# backend/app/middleware.py - УПРОЩЁННЫЙ MIDDLEWARE
"""
Упрощённый middleware без сложных timeout систем и множественных обработчиков
Заменяет переусложнённый middleware.py с BaseHTTPMiddleware и rate limiting
"""

import time
import logging
import uuid
from typing import Dict, Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Простые статистики
_request_stats = {
    "total_requests": 0,
    "requests_by_method": {},
    "response_times": [],
    "errors": 0
}

def setup_middleware(app):
    """
    Настраивает упрощённый middleware без сложностей
    Убирает BaseHTTPMiddleware, rate limiting, security проверки
    """
    
    try:
        logger.info("🚀 Configuring simplified middleware...")
        
        @app.middleware("http")
        async def simple_request_middleware(request: Request, call_next):
            """
            Простой middleware для логирования и базовой статистики
            Убирает все сложности из предыдущей версии
            """
            # Генерируем ID запроса
            request_id = str(uuid.uuid4())[:8]
            start_time = time.time()
            
            # Добавляем ID в контекст
            request.state.request_id = request_id
            
            # Получаем базовую информацию
            method = request.method
            path = request.url.path
            
            # Обновляем простую статистику
            _update_request_stats(method)
            
            # Логируем только важные запросы (не статические файлы)
            if not _is_static_path(path):
                logger.info(f"🌐 [{request_id}] {method} {path}")
            
            try:
                # Выполняем запрос
                response = await call_next(request)
                
                # Считаем время
                process_time = time.time() - start_time
                
                # Добавляем заголовки
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Process-Time"] = str(round(process_time, 3))
                
                # Обновляем статистику
                _update_response_stats(process_time)
                
                # Логируем завершение (только для важных запросов)
                if not _is_static_path(path):
                    status_emoji = "✅" if response.status_code < 400 else "❌"
                    logger.info(f"{status_emoji} [{request_id}] {method} {path} - {response.status_code} - {process_time:.3f}s")
                
                return response
                
            except Exception as e:
                process_time = time.time() - start_time
                
                logger.error(f"❌ [{request_id}] {method} {path} - Error: {str(e)} - {process_time:.3f}s")
                
                # Обновляем статистику ошибок
                _request_stats["errors"] += 1
                
                # Возвращаем понятную ошибку
                return JSONResponse(
                    status_code=500,
                    content={
                        "detail": "Internal server error",
                        "request_id": request_id,
                        "path": path,
                        "method": method,
                        "timestamp": time.time(),
                        "help": "Check server logs for details"
                    }
                )
        
        logger.info("✅ Simplified middleware configured successfully")
        
    except Exception as e:
        logger.error(f"❌ Error configuring middleware: {e}")
        # Не падаем, middleware не критичен

def _is_static_path(path: str) -> bool:
    """Проверяет является ли путь статическим файлом"""
    static_paths = ["/docs", "/redoc", "/openapi.json", "/favicon.ico", "/static", "/manifest.json"]
    return any(path.startswith(static) for static in static_paths)

def _update_request_stats(method: str):
    """Обновляет простую статистику запросов"""
    _request_stats["total_requests"] += 1
    _request_stats["requests_by_method"][method] = _request_stats["requests_by_method"].get(method, 0) + 1

def _update_response_stats(response_time: float):
    """Обновляет статистику времени ответа"""
    _request_stats["response_times"].append(response_time)
    
    # Ограничиваем размер списка
    if len(_request_stats["response_times"]) > 100:
        _request_stats["response_times"] = _request_stats["response_times"][-50:]

def get_middleware_stats() -> Dict[str, Any]:
    """Возвращает простую статистику middleware"""
    response_times = _request_stats["response_times"]
    
    stats = {
        "total_requests": _request_stats["total_requests"],
        "requests_by_method": _request_stats["requests_by_method"],
        "total_errors": _request_stats["errors"],
        "average_response_time": 0.0,
        "min_response_time": 0.0,
        "max_response_time": 0.0
    }
    
    if response_times:
        stats["average_response_time"] = sum(response_times) / len(response_times)
        stats["min_response_time"] = min(response_times)
        stats["max_response_time"] = max(response_times)
    
    return stats

def reset_middleware_stats():
    """Сбрасывает статистику middleware"""
    global _request_stats
    _request_stats = {
        "total_requests": 0,
        "requests_by_method": {},
        "response_times": [],
        "errors": 0
    }
    logger.info("🔄 Middleware statistics reset")

# ====================================
# ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ СОВМЕСТИМОСТИ
# ====================================

def get_security_stats() -> Dict[str, Any]:
    """Заглушка для security статистики (для совместимости)"""
    return {
        "blocked_ips_count": 0,
        "rate_limited_ips": 0,
        "total_errors": _request_stats["errors"],
        "security_enabled": False,
        "note": "Security features disabled in simplified version"
    }

def block_ip(ip: str, reason: str = "Manual block"):
    """Заглушка для блокировки IP (для совместимости)"""
    logger.info(f"⚠️ IP blocking not implemented in simplified version: {ip} ({reason})")

def unblock_ip(ip: str):
    """Заглушка для разблокировки IP (для совместимости)"""
    logger.info(f"⚠️ IP unblocking not implemented in simplified version: {ip}")

# ====================================
# ЭКСПОРТ ФУНКЦИЙ
# ====================================

__all__ = [
    # Основные функции
    'setup_middleware',
    'get_middleware_stats',
    'reset_middleware_stats',
    
    # Совместимость с предыдущей версией
    'get_security_stats',
    'block_ip',
    'unblock_ip'
]