# ====================================
# ФАЙЛ: backend/app/middleware.py (ПОЛНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ)
# Заменить существующий файл полностью
# ====================================

"""
Middleware для FastAPI приложения Legal Assistant
ИСПРАВЛЕНИЯ: Заменен BaseHTTPMiddleware на функциональный middleware для решения проблемы POST 404
"""

import time
import logging
import json
import uuid
from datetime import datetime
from typing import Callable, Dict, Any, Optional
from urllib.parse import urlparse

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse

# Пытаемся импортировать utils, но делаем fallback если недоступно
try:
    from utils.helpers import notification_manager, PerformanceTimer
    _utils_available = True
except ImportError:
    _utils_available = False
    # Создаем заглушки
    class MockNotificationManager:
        def add_notification(self, message, type="info"):
            logging.getLogger("notifications").info(f"[{type}] {message}")
    
    class MockPerformanceTimer:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    notification_manager = MockNotificationManager()
    PerformanceTimer = MockPerformanceTimer

try:
    from app.config import settings
    _config_available = True
except ImportError:
    _config_available = False
    # Заглушка для настроек
    class MockSettings:
        def __init__(self):
            pass
    settings = MockSettings()

logger = logging.getLogger(__name__)

# Глобальные переменные для сбора статистики
_request_stats = {
    "total_requests": 0,
    "requests_by_method": {},
    "requests_by_path": {},
    "response_times": [],
    "status_codes": {},
    "user_agents": {},
    "errors": [],
    "blocked_ips": set(),
    "rate_limit_storage": {}  # IP -> {count, reset_time}
}

def setup_middleware(app):
    """
    ИСПРАВЛЕННАЯ функция настройки middleware
    Заменяет BaseHTTPMiddleware на функциональные middleware
    """
    
    try:
        logger.info("🚀 Configuring middleware with POST fixes...")
        
        # ====================================
        # ГЛАВНЫЙ ИСПРАВЛЕННЫЙ MIDDLEWARE
        # ====================================
        @app.middleware("http")
        async def main_request_middleware(request: Request, call_next: Callable):
            """
            Основной middleware объединяющий все функции
            ИСПРАВЛЕНИЕ: Функциональный middleware вместо BaseHTTPMiddleware
            """
            # Генерируем уникальный ID запроса
            request_id = str(uuid.uuid4())[:8]
            start_time = time.time()
            
            # Добавляем ID запроса в контекст
            request.state.request_id = request_id
            
            # Получаем информацию о клиенте
            client_ip = _get_client_ip(request)
            user_agent = request.headers.get("user-agent", "Unknown")
            path = request.url.path
            method = request.method
            
            # Обновляем статистику запросов
            _update_request_stats(method, path, user_agent)
            
            # Проверки безопасности
            security_check = _perform_security_checks(request, client_ip)
            if security_check:
                return security_check  # Возвращаем ошибку если заблокировано
            
            # Логируем входящий запрос (исключаем статические файлы)
            if not _is_static_path(path):
                logger.info(
                    f"🌐 [{request_id}] {method} {path} - "
                    f"IP: {client_ip} - UA: {user_agent[:50]}..."
                )
            
            try:
                # КРИТИЧЕСКИ ВАЖНО: Просто передаем запрос дальше
                response = await call_next(request)
                
                # Вычисляем время обработки
                process_time = time.time() - start_time
                
                # Добавляем заголовки к ответу
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Process-Time"] = str(round(process_time, 3))
                
                # Добавляем security заголовки
                _add_security_headers(response)
                
                # Логируем ответ (исключаем статические файлы)
                if not _is_static_path(path):
                    status_emoji = "✅" if response.status_code < 400 else "❌"
                    logger.info(
                        f"{status_emoji} [{request_id}] {method} {path} - "
                        f"Status: {response.status_code} - "
                        f"Time: {process_time:.3f}s"
                    )
                
                # Обновляем статистику ответов
                _update_response_stats(response.status_code, process_time)
                
                # Отправляем уведомление для критических ошибок (если utils доступны)
                if _utils_available and path.startswith("/api/admin") and response.status_code >= 400:
                    notification_manager.add_notification(
                        f"Admin API error: {method} {path} returned {response.status_code}",
                        "error"
                    )
                
                return response
                
            except Exception as e:
                process_time = time.time() - start_time
                
                logger.error(
                    f"❌ [{request_id}] {method} {path} - "
                    f"Error: {str(e)} - "
                    f"Time: {process_time:.3f}s",
                    exc_info=True
                )
                
                # Записываем ошибку в статистику
                _record_error(request_id, path, method, str(e))
                
                # Отправляем уведомление об ошибке (если utils доступны)
                if _utils_available:
                    notification_manager.add_notification(
                        f"Server error on {method} {path}: {str(e)}",
                        "error"
                    )
                
                # Возвращаем пользователю общую ошибку (не раскрываем детали)
                return JSONResponse(
                    status_code=500,
                    content={
                        "detail": "Internal server error",
                        "request_id": request_id,
                        "timestamp": datetime.now().isoformat(),
                        "path": path,
                        "method": method
                    }
                )
        
        logger.info("✅ Main middleware configured successfully")
        
    except Exception as e:
        logger.error(f"❌ Error configuring middleware: {e}")
        raise

# ====================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ====================================

def _get_client_ip(request: Request) -> str:
    """Получает IP адрес клиента с учетом прокси"""
    # Проверяем заголовки прокси
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback к прямому подключению
    if hasattr(request.client, "host"):
        return request.client.host
    
    return "unknown"

def _is_static_path(path: str) -> bool:
    """Проверяет является ли путь статическим файлом"""
    static_paths = {"/docs", "/redoc", "/openapi.json", "/favicon.ico", "/static"}
    return any(path.startswith(static) for static in static_paths)

def _perform_security_checks(request: Request, client_ip: str) -> Optional[JSONResponse]:
    """Выполняет проверки безопасности"""
    
    # Проверяем заблокированные IP
    if client_ip in _request_stats["blocked_ips"]:
        logger.warning(f"🚫 Blocked IP attempted access: {client_ip}")
        return JSONResponse(
            status_code=403,
            content={"detail": "Access denied"}
        )
    
    # Проверяем rate limiting
    if _is_rate_limited(client_ip):
        logger.warning(f"⚠️ Rate limit exceeded for IP: {client_ip}")
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please try again later."}
        )
    
    # Проверяем подозрительные паттерны
    if _is_suspicious_request(request):
        logger.warning(f"🔍 Suspicious request from {client_ip}: {request.url.path}")
        # Можно заблокировать IP или просто логировать
        
    return None

def _is_rate_limited(client_ip: str, requests_per_hour: int = 100) -> bool:
    """Проверяет rate limiting для IP"""
    now = time.time()
    window = 3600  # 1 час
    
    if client_ip not in _request_stats["rate_limit_storage"]:
        _request_stats["rate_limit_storage"][client_ip] = {
            "count": 1,
            "reset_time": now + window
        }
        return False
    
    ip_data = _request_stats["rate_limit_storage"][client_ip]
    
    # Сбрасываем счетчик если окно истекло
    if now > ip_data["reset_time"]:
        ip_data["count"] = 1
        ip_data["reset_time"] = now + window
        return False
    
    # Увеличиваем счетчик
    ip_data["count"] += 1
    
    # Проверяем лимит
    return ip_data["count"] > requests_per_hour

def _is_suspicious_request(request: Request) -> bool:
    """Определяет подозрительные запросы"""
    path = request.url.path.lower()
    user_agent = request.headers.get("user-agent", "").lower()
    
    # Подозрительные пути
    suspicious_paths = [
        "/.env", "/wp-admin", "/admin.php", "/phpmyadmin",
        "/wp-login.php", "/.git", "/config", "/backup"
    ]
    
    if any(suspicious in path for suspicious in suspicious_paths):
        return True
    
    # Подозрительные User-Agent
    suspicious_agents = [
        "sqlmap", "nikto", "nmap", "masscan", "zap",
        "burpsuite", "w3af", "acunetix"
    ]
    
    if any(agent in user_agent for agent in suspicious_agents):
        return True
    
    # Слишком длинные пути
    if len(path) > 500:
        return True
    
    return False

def _add_security_headers(response: Response):
    """Добавляет security заголовки к ответу"""
    security_headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';",
    }
    
    for header, value in security_headers.items():
        response.headers[header] = value

def _update_request_stats(method: str, path: str, user_agent: str):
    """Обновляет статистику входящих запросов"""
    _request_stats["total_requests"] += 1
    
    # Статистика по методам
    _request_stats["requests_by_method"][method] = (
        _request_stats["requests_by_method"].get(method, 0) + 1
    )
    
    # Статистика по путям
    _request_stats["requests_by_path"][path] = (
        _request_stats["requests_by_path"].get(path, 0) + 1
    )
    
    # Статистика User-Agent (ограничиваем длину)
    ua_short = user_agent[:50]
    _request_stats["user_agents"][ua_short] = (
        _request_stats["user_agents"].get(ua_short, 0) + 1
    )

def _update_response_stats(status_code: int, response_time: float):
    """Обновляет статистику ответов"""
    # Записываем время ответа
    _request_stats["response_times"].append(response_time)
    
    # Ограничиваем размер списка времен ответа
    if len(_request_stats["response_times"]) > 1000:
        _request_stats["response_times"] = _request_stats["response_times"][-500:]
    
    # Записываем статус код
    _request_stats["status_codes"][status_code] = (
        _request_stats["status_codes"].get(status_code, 0) + 1
    )

def _record_error(request_id: str, path: str, method: str, error: str):
    """Записывает информацию об ошибке"""
    error_info = {
        "timestamp": time.time(),
        "request_id": request_id,
        "path": path,
        "method": method,
        "error": error,
        "type": type(error).__name__ if hasattr(error, '__class__') else "UnknownError"
    }
    
    _request_stats["errors"].append(error_info)
    
    # Ограничиваем размер списка ошибок
    if len(_request_stats["errors"]) > 100:
        _request_stats["errors"] = _request_stats["errors"][-50:]

# ====================================
# ФУНКЦИИ ДЛЯ ПОЛУЧЕНИЯ СТАТИСТИКИ
# ====================================

def get_middleware_stats() -> Dict[str, Any]:
    """Возвращает статистику middleware"""
    response_times = _request_stats["response_times"]
    
    stats = _request_stats.copy()
    
    if response_times:
        stats["average_response_time"] = sum(response_times) / len(response_times)
        stats["min_response_time"] = min(response_times)
        stats["max_response_time"] = max(response_times)
        
        # Перцентили
        sorted_times = sorted(response_times)
        stats["p50_response_time"] = sorted_times[len(sorted_times) // 2]
        stats["p95_response_time"] = sorted_times[int(len(sorted_times) * 0.95)]
        stats["p99_response_time"] = sorted_times[int(len(sorted_times) * 0.99)]
    else:
        stats.update({
            "average_response_time": 0,
            "min_response_time": 0,
            "max_response_time": 0,
            "p50_response_time": 0,
            "p95_response_time": 0,
            "p99_response_time": 0
        })
    
    # Удаляем сырые данные из ответа
    stats.pop("response_times", None)
    
    # Конвертируем set в list для JSON сериализации
    stats["blocked_ips"] = list(stats["blocked_ips"])
    
    return stats

def get_security_stats() -> Dict[str, Any]:
    """Возвращает статистику безопасности"""
    return {
        "blocked_ips_count": len(_request_stats["blocked_ips"]),
        "blocked_ips": list(_request_stats["blocked_ips"]),
        "rate_limited_ips": len(_request_stats["rate_limit_storage"]),
        "total_errors": len(_request_stats["errors"]),
        "recent_errors": _request_stats["errors"][-10:] if _request_stats["errors"] else []
    }

def reset_middleware_stats():
    """Сбрасывает статистику middleware"""
    global _request_stats
    _request_stats = {
        "total_requests": 0,
        "requests_by_method": {},
        "requests_by_path": {},
        "response_times": [],
        "status_codes": {},
        "user_agents": {},
        "errors": [],
        "blocked_ips": set(),
        "rate_limit_storage": {}
    }
    logger.info("🔄 Middleware statistics reset")

def block_ip(ip: str, reason: str = "Manual block"):
    """Блокирует IP адрес"""
    _request_stats["blocked_ips"].add(ip)
    logger.warning(f"🚫 IP {ip} blocked: {reason}")

def unblock_ip(ip: str):
    """Разблокирует IP адрес"""
    _request_stats["blocked_ips"].discard(ip)
    logger.info(f"✅ IP {ip} unblocked")

# Экспорт основных функций
__all__ = [
    'setup_middleware',
    'get_middleware_stats',
    'get_security_stats', 
    'reset_middleware_stats',
    'block_ip',
    'unblock_ip'
]