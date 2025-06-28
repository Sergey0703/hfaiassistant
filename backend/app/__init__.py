# backend/app/__init__.py - УПРОЩЁННАЯ ФАБРИКА ПРИЛОЖЕНИЯ
"""
Упрощённая фабрика FastAPI приложения без сложных lifespan и background задач
Заменяет переусложнённый файл с lazy loading и множественными проверками
"""

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

def create_app():
    """
    Создаёт FastAPI приложение с простой конфигурацией
    Убирает всю сложность lifespan, background tasks, etc.
    """
    try:
        logger.info("🚀 Creating simplified FastAPI application...")
        
        # Импорты FastAPI
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
        
        # Конфигурация
        try:
            from app.config import settings, API_METADATA, API_TAGS
            app_config = {
                "title": API_METADATA["title"],
                "version": API_METADATA["version"], 
                "description": API_METADATA["description"],
                "openapi_tags": API_TAGS
            }
            config_loaded = True
        except ImportError as e:
            logger.warning(f"Config import failed: {e}, using defaults")
            app_config = {
                "title": "Legal Assistant API",
                "version": "2.0.0",
                "description": "AI Legal Assistant with Llama integration"
            }
            config_loaded = False
        
        # Создаём приложение
        app = FastAPI(**app_config)
        
        # CORS middleware (первым!)
        cors_origins = getattr(settings, 'CORS_ORIGINS', ["*"]) if config_loaded else ["*"]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info("✅ CORS middleware configured")
        
        # Подключаем API роутеры
        try:
            from api import configure_fastapi_app
            configure_fastapi_app(app)
            logger.info("✅ API routes configured")
        except Exception as e:
            logger.error(f"❌ API routes configuration failed: {e}")
            # Добавляем fallback endpoint для диагностики
            _add_fallback_routes(app)
        
        # Подключаем middleware
        try:
            from app.middleware import setup_middleware
            setup_middleware(app)
            logger.info("✅ Middleware configured")
        except Exception as e:
            logger.warning(f"⚠️ Middleware setup failed: {e}")
        
        # Настраиваем статические файлы React (если есть)
        _setup_react_static_files(app)
        
        # Добавляем базовые endpoints
        _add_basic_endpoints(app)
        
        # Глобальные обработчики ошибок
        _setup_error_handlers(app)
        
        logger.info("✅ FastAPI application created successfully")
        return app
        
    except Exception as e:
        logger.error(f"❌ Critical error during application creation: {e}")
        return _create_emergency_app(e)

def _add_basic_endpoints(app):
    """Добавляет базовые системные endpoints"""
    
    @app.get("/health")
    async def health_check():
        """Простая проверка здоровья"""
        try:
            from app.dependencies import get_services_status
            services = get_services_status()
            
            overall_status = "healthy"
            if services.get("total_errors", 0) > 0:
                overall_status = "degraded"
            
            return {
                "status": overall_status,
                "timestamp": time.time(),
                "version": "2.0.0",
                "services": {
                    "document_service": services.get("document_service_available", False),
                    "llm_service": services.get("llm_available", False),
                    "scraper_service": services.get("scraper_available", False)
                },
                "platform": "HuggingFace Spaces" if services.get("huggingface_spaces") else "Local",
                "llm_model": "Llama-3.1-8B-Instruct"
            }
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": time.time()
                }
            )
    
    @app.get("/api-status")
    async def api_status():
        """Информация о API статусе"""
        try:
            from app.config import get_full_config_summary
            config_summary = get_full_config_summary()
            
            return {
                "api": "Legal Assistant API v2.0",
                "status": "running",
                "llm_model": "Llama-3.1-8B-Instruct",
                "features": {
                    "llama_integration": True,
                    "vector_search": config_summary["database"]["use_chromadb"],
                    "web_scraping": True,
                    "multilingual": True,
                    "react_frontend": _check_react_build()
                },
                "config_validation": config_summary["validation"],
                "endpoints": {
                    "docs": "/docs",
                    "health": "/health",
                    "chat": "/api/user/chat",
                    "search": "/api/user/search",
                    "admin": "/api/admin"
                }
            }
        except Exception as e:
            return {
                "api": "Legal Assistant API v2.0",
                "status": "running",
                "error": f"Config error: {e}",
                "basic_endpoints": ["/health", "/docs"]
            }

def _add_fallback_routes(app):
    """Добавляет fallback роуты если API не загрузились"""
    
    @app.get("/api")
    async def api_fallback():
        return {
            "message": "Legal Assistant API",
            "version": "2.0.0",
            "status": "❌ API routes not configured",
            "available_endpoints": ["/health", "/api-status", "/docs"],
            "issue": "API modules failed to load",
            "recommendation": "Check that all API files are present"
        }
    
    @app.get("/api/user/chat")
    async def chat_fallback():
        return {
            "response": "Chat service is initializing. API routes not fully loaded yet.",
            "sources": []
        }

def _setup_react_static_files(app):
    """Настраивает статические файлы React"""
    try:
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse
        
        # Путь к React build
        react_build_path = Path(__file__).parent.parent.parent / "frontend" / "build"
        react_static_path = react_build_path / "static"
        
        if react_static_path.exists():
            app.mount("/static", StaticFiles(directory=react_static_path), name="react_static")
            logger.info(f"✅ React static files mounted: {react_static_path}")
        
        # React assets (manifest, favicon, etc.)
        if react_build_path.exists():
            react_assets = ["manifest.json", "favicon.ico", "robots.txt"]
            
            for asset in react_assets:
                asset_path = react_build_path / asset
                if asset_path.exists():
                    def create_asset_handler(asset_name):
                        async def serve_asset():
                            return FileResponse(react_build_path / asset_name)
                        return serve_asset
                    
                    app.get(f"/{asset}", include_in_schema=False)(create_asset_handler(asset))
        
        # Корневой маршрут для React SPA (если index.html существует)
        index_path = react_build_path / "index.html"
        if index_path.exists():
            @app.get("/", include_in_schema=False)
            async def serve_react_app():
                return FileResponse(index_path, media_type="text/html")
            
            logger.info("✅ React SPA mounted at root path")
        else:
            logger.info("⚠️ React build not found - API-only mode")
            
            @app.get("/")
            async def root_fallback():
                return {
                    "message": "Legal Assistant API v2.0",
                    "llm_model": "Llama-3.1-8B-Instruct", 
                    "status": "API running, React frontend not built",
                    "endpoints": {
                        "api_docs": "/docs",
                        "health": "/health",
                        "chat": "/api/user/chat"
                    },
                    "build_react": [
                        "cd frontend",
                        "npm install",
                        "npm run build"
                    ]
                }
    
    except Exception as e:
        logger.warning(f"⚠️ React static files setup failed: {e}")

def _setup_error_handlers(app):
    """Настраивает обработчики ошибок"""
    
    @app.exception_handler(404)
    async def not_found_handler(request, exc):
        path = str(request.url.path)
        
        # Если это API роут
        if path.startswith("/api/"):
            return JSONResponse(
                status_code=404,
                content={
                    "detail": f"API endpoint not found: {path}",
                    "available_endpoints": ["/api", "/api/user/chat", "/api/user/search"],
                    "documentation": "/docs"
                }
            )
        
        # Для всех остальных - может быть React SPA
        return JSONResponse(
            status_code=404,
            content={
                "detail": f"Page not found: {path}",
                "api_documentation": "/docs",
                "health_check": "/health"
            }
        )
    
    @app.exception_handler(500)
    async def internal_error_handler(request, exc):
        logger.error(f"Internal server error on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "path": str(request.url.path),
                "timestamp": time.time(),
                "support": "Check server logs for details"
            }
        )

def _check_react_build() -> bool:
    """Проверяет собрано ли React приложение"""
    react_build_path = Path(__file__).parent.parent.parent / "frontend" / "build"
    return (react_build_path / "index.html").exists()

def _create_emergency_app(error: Exception):
    """Создаёт аварийное приложение"""
    logger.error(f"Creating emergency app due to: {error}")
    
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        
        emergency_app = FastAPI(title="Legal Assistant API - Emergency Mode")
        
        # CORS даже в аварийном режиме
        emergency_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        @emergency_app.get("/")
        async def emergency_root():
            return {
                "status": "emergency_mode",
                "error": str(error),
                "message": "Application failed to initialize properly",
                "timestamp": time.time(),
                "endpoints": {
                    "this_status": "/",
                    "try_docs": "/docs"
                },
                "recommendations": [
                    "Check that all dependencies are installed",
                    "Verify all Python modules are present",
                    "Check server logs for detailed errors",
                    "Try restarting the application"
                ]
            }
        
        @emergency_app.get("/health")
        async def emergency_health():
            return {
                "status": "emergency",
                "error": str(error),
                "timestamp": time.time()
            }
        
        return emergency_app
        
    except Exception as final_error:
        logger.critical(f"❌ Cannot create even emergency application: {final_error}")
        raise

# ====================================
# ЭКСПОРТ
# ====================================

__all__ = [
    "create_app"
]