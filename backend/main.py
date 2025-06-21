# backend/main.py - ИСПРАВЛЕННАЯ ВЕРСИЯ ДЛЯ POST ENDPOINTS
"""
Legal Assistant API - ИСПРАВЛЕНИЕ ПРОБЛЕМЫ С POST 404
КЛЮЧЕВЫЕ ИСПРАВЛЕНИЯ:
1. Упрощенный middleware (убираем BaseHTTPMiddleware)
2. Правильный порядок CORS middleware
3. Исправление инициализации приложения
"""

import uvicorn
import sys
import os
from pathlib import Path
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging
import uuid

# Добавляем текущую директорию в Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def create_app_fixed() -> FastAPI:
    """
    Создает FastAPI приложение с исправлениями для POST endpoints
    """
    try:
        logger.info("🚀 Creating FastAPI application with POST fixes...")
        
        # ИСПРАВЛЕНИЕ 1: Создаем приложение БЕЗ lifespan сначала
        app = FastAPI(
            title="Legal Assistant API",
            version="2.0.0",
            description="AI Legal Assistant with GPTQ model support",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json"
        )
        
        # ИСПРАВЛЕНИЕ 2: СНАЧАЛА добавляем CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Разрешаем все origins для HF Spaces
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],  # Явно указываем методы
            allow_headers=["*"],
            expose_headers=["*"],
            max_age=3600
        )
        
        # ИСПРАВЛЕНИЕ 3: Простой функциональный middleware вместо BaseHTTPMiddleware
        @app.middleware("http")
        async def simple_logging_middleware(request: Request, call_next):
            """Простой middleware для логирования"""
            request_id = str(uuid.uuid4())[:8]
            start_time = time.time()
            
            # Логируем входящий запрос
            method = request.method
            path = request.url.path
            client_ip = request.client.host if request.client else "unknown"
            
            logger.info(f"🌐 [{request_id}] {method} {path} - IP: {client_ip}")
            
            try:
                # ВАЖНО: Просто передаем запрос дальше
                response = await call_next(request)
                
                # Логируем ответ
                process_time = time.time() - start_time
                status_emoji = "✅" if response.status_code < 400 else "❌"
                logger.info(f"{status_emoji} [{request_id}] {method} {path} - Status: {response.status_code} - Time: {process_time:.3f}s")
                
                # Добавляем заголовки
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Process-Time"] = str(round(process_time, 3))
                
                return response
                
            except Exception as e:
                process_time = time.time() - start_time
                logger.error(f"❌ [{request_id}] {method} {path} - Error: {str(e)} - Time: {process_time:.3f}s")
                raise
        
        # ИСПРАВЛЕНИЕ 4: Базовые маршруты ДО подключения API
        @app.get("/")
        async def root():
            return {
                "message": "Legal Assistant API with GPTQ Model",
                "version": "2.0.0",
                "status": "healthy",
                "platform": "HuggingFace Spaces" if os.getenv("SPACE_ID") else "Local",
                "docs": "/docs",
                "health": "/health"
            }
        
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "platform": "HuggingFace Spaces" if os.getenv("SPACE_ID") else "Local"
            }
        
        # ИСПРАВЛЕНИЕ 5: Подключаем API роутеры ПОСЛЕ middleware
        try:
            logger.info("📡 Configuring API routes...")
            
            # Импортируем и настраиваем API роутеры
            from api.user.chat import router as chat_router
            from api.user.search import router as search_router
            
            # Подключаем пользовательские маршруты
            app.include_router(
                chat_router,
                prefix="/api/user",
                tags=["User Chat"]
            )
            
            app.include_router(
                search_router,
                prefix="/api/user", 
                tags=["User Search"]
            )
            
            logger.info("✅ User API routes configured successfully")
            
            # Пытаемся подключить админские маршруты
            try:
                from api.admin.documents import router as admin_docs_router
                from api.admin.stats import router as admin_stats_router
                
                app.include_router(
                    admin_docs_router,
                    prefix="/api/admin",
                    tags=["Admin Documents"]
                )
                
                app.include_router(
                    admin_stats_router,
                    prefix="/api/admin",
                    tags=["Admin Stats"]
                )
                
                logger.info("✅ Admin API routes configured successfully")
                
            except ImportError as e:
                logger.warning(f"⚠️ Admin routes not available: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error configuring API routes: {e}")
            
            # Создаем fallback API endpoint
            @app.get("/api/status")
            async def api_status():
                return {
                    "status": "❌ API routes configuration failed",
                    "error": str(e),
                    "available_endpoints": [
                        "GET / - Root endpoint",
                        "GET /health - Health check",
                        "GET /docs - API documentation"
                    ]
                }
        
        # ИСПРАВЛЕНИЕ 6: Тестовый POST endpoint для диагностики
        @app.post("/test-post")
        async def test_post(data: dict = None):
            """Тестовый POST endpoint для проверки что POST работает"""
            return {
                "message": "POST endpoint works!",
                "received_data": data,
                "timestamp": time.time(),
                "method": "POST"
            }
        
        @app.options("/test-post")
        async def test_post_options():
            """OPTIONS для тестового POST endpoint"""
            return Response(status_code=200, headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            })
        
        # ИСПРАВЛЕНИЕ 7: Специальные endpoints для HF Spaces
        @app.get("/hf-spaces-health")
        async def hf_spaces_health():
            try:
                from app.dependencies import get_services_status
                services = get_services_status()
                
                return {
                    "status": "healthy",
                    "platform": "HuggingFace Spaces",
                    "api_version": "2.0.0",
                    "services": services,
                    "endpoints": {
                        "chat": "/api/user/chat",
                        "search": "/api/user/search",
                        "test_post": "/test-post"
                    },
                    "post_fix_applied": True
                }
            except Exception as e:
                return {
                    "status": "error", 
                    "error": str(e),
                    "platform": "HuggingFace Spaces"
                }
        
        # ИСПРАВЛЕНИЕ 8: Глобальный обработчик 404
        @app.exception_handler(404)
        async def not_found_handler(request: Request, exc):
            return JSONResponse(
                status_code=404,
                content={
                    "detail": f"Endpoint not found: {request.method} {request.url.path}",
                    "available_endpoints": {
                        "root": "GET /",
                        "health": "GET /health",
                        "docs": "GET /docs",
                        "test_post": "POST /test-post",
                        "api_chat": "POST /api/user/chat",
                        "api_search": "POST /api/user/search"
                    },
                    "method": request.method,
                    "path": request.url.path,
                    "suggestion": "Check /docs for available endpoints"
                }
            )
        
        logger.info("✅ FastAPI application created with POST fixes")
        return app
        
    except Exception as e:
        logger.error(f"❌ Critical error creating application: {e}")
        
        # Минимальное fallback приложение
        fallback_app = FastAPI(title="Legal Assistant API - Emergency Mode")
        
        fallback_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        @fallback_app.get("/")
        async def emergency_root():
            return {
                "status": "emergency_mode",
                "error": str(e),
                "message": "Application failed to initialize properly"
            }
        
        @fallback_app.post("/test-post") 
        async def emergency_test():
            return {"message": "Emergency POST works", "error": str(e)}
        
        return fallback_app

# ИСПРАВЛЕНИЕ 9: Создаем приложение для deployment
try:
    logger.info("🚀 Initializing Legal Assistant API for HuggingFace Spaces...")
    
    # Проверяем HF Spaces
    is_hf_spaces = os.getenv("SPACE_ID") is not None
    if is_hf_spaces:
        logger.info("🌍 HuggingFace Spaces detected - applying optimizations")
        
        # Применяем оптимизации для HF Spaces
        os.environ.setdefault("LLM_DEMO_MODE", "false")
        os.environ.setdefault("USE_CHROMADB", "true") 
        os.environ.setdefault("LOG_LEVEL", "INFO")
    
    # Создаем исправленное приложение
    app = create_app_fixed()
    
    logger.info("✅ Legal Assistant API ready for deployment")
    logger.info(f"🌍 Platform: {'HuggingFace Spaces' if is_hf_spaces else 'Local'}")
    logger.info("🔧 POST endpoint fixes applied")
    
except Exception as e:
    logger.error(f"❌ Deployment initialization failed: {e}")
    
    # Простейший fallback
    app = FastAPI(title="Legal Assistant API - Recovery Mode")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"], 
        allow_headers=["*"]
    )
    
    @app.get("/")
    async def recovery_root():
        return {
            "status": "recovery_mode",
            "error": str(e),
            "platform": "HuggingFace Spaces" if os.getenv("SPACE_ID") else "Local"
        }
    
    @app.post("/test-post")
    async def recovery_test():
        return {"message": "Recovery POST works", "timestamp": time.time()}

def main():
    """Главная функция для разработки"""
    try:
        host = "0.0.0.0"
        port = 7860 if os.getenv("SPACE_ID") else 8000
        
        logger.info(f"🌐 Starting server on {host}:{port}")
        logger.info("🔧 POST endpoint fixes enabled")
        
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            log_level="info",
            reload=False,
            workers=1,
            timeout_keep_alive=65
        )
        
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        raise

if __name__ == "__main__":
    main()
else:
    logger.info("📦 Legal Assistant API module imported with POST fixes")
    logger.info("🔗 Test endpoint: POST /test-post")
    logger.info("🏥 Health Check: /hf-spaces-health")