# backend/app/middleware.py - ИСПРАВЛЕННАЯ ВЕРСИЯ
"""
Упрощенный middleware для FastAPI без BaseHTTPMiddleware
ИСПРАВЛЕНИЯ: Убираем BaseHTTPMiddleware который может блокировать POST запросы
"""

import time
import logging
import uuid
from fastapi import Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

def setup_middleware(app):
    """
    Настраивает упрощенный middleware для предотвращения проблем с POST
    ИСПРАВЛЕНИЕ: Используем только функциональный middleware
    """
    
    try:
        logger.info("🔧 Setting up simplified middleware (no BaseHTTPMiddleware)")
        
        # ТОЛЬКО один простой middleware для логирования
        @app.middleware("http")
        async def request_logging_middleware(request: Request, call_next):
            """Упрощенный middleware для логирования запросов"""
            
            request_id = str(uuid.uuid4())[:8]
            start_time = time.time()
            
            # Получаем информацию о запросе
            method = request.method
            path = request.url.path
            client_ip = request.client.host if request.client else "unknown"
            
            # НЕ логируем статические файлы
            if not path.startswith(("/favicon.ico", "/static", "/docs", "/openapi.json")):
                logger.info(f"🌐 [{request_id}] {method} {path} - IP: {client_ip}")
            
            try:
                # ВАЖНО: Просто передаем управление дальше
                response = await call_next(request)
                
                # Добавляем заголовки к ответу
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Process-Time"] = str(round(time.time() - start_time, 3))
                
                # Логируем только если это не статика
                if not path.startswith(("/favicon.ico", "/static", "/docs", "/openapi.json")):
                    process_time = time.time() - start_time
                    status_emoji = "✅" if response.status_code < 400 else "❌"
                    logger.info(f"{status_emoji} [{request_id}] {method} {path} - Status: {response.status_code} - Time: {process_time:.3f}s")
                
                return response
                
            except Exception as e:
                process_time = time.time() - start_time
                logger.error(f"❌ [{request_id}] {method} {path} - Error: {str(e)} - Time: {process_time:.3f}s")
                
                # Возвращаем JSON ошибку
                return JSONResponse(
                    status_code=500,
                    content={
                        "detail": "Internal server error",
                        "request_id": request_id,
                        "path": path,
                        "method": method,
                        "timestamp": time.time()
                    }
                )
        
        logger.info("✅ Simplified middleware configured successfully")
        
    except Exception as e:
        logger.error(f"❌ Error configuring middleware: {e}")
        raise

# Убираем все классы BaseHTTPMiddleware - они могут вызывать проблемы с POST

# Простые utility функции для совместимости
def get_client_ip(request: Request) -> str:
    """Получает IP адрес клиента"""
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

def is_suspicious_request(request: Request) -> bool:
    """Простая проверка на подозрительные запросы"""
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
        "sqlmap", "nikto", "nmap", "masscan", "zap"
    ]
    
    if any(agent in user_agent for agent in suspicious_agents):
        return True
    
    return False

# Экспорт для совместимости
__all__ = [
    'setup_middleware',
    'get_client_ip', 
    'is_suspicious_request'
]