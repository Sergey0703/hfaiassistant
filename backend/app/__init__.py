# backend/app/__init__.py - УПРОЩЁННАЯ ФАБРИКА ПРИЛОЖЕНИЯ
"""
Минимальная фабрика FastAPI приложения для RAG системы
Убрана сложность lifespan, background tasks, сложные middleware
"""

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

def create_app():
    """
    Создаёт FastAPI приложение для минимальной RAG системы
    Простая конфигурация без сложностей
    """
    try:
        logger.info("🚀 Creating minimal RAG FastAPI application...")
        
        # Импорты FastAPI
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
        
        # Конфигурация
        try:
            from app.config import API_METADATA, API_TAGS
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
                "title": "Minimal RAG System",
                "version": "1.0.0",
                "description": "Minimal RAG with FLAN-T5 and sentence-transformers"
            }
            config_loaded = False
        
        # Создаём приложение
        app = FastAPI(**app_config)
        
        # CORS middleware (упрощенный)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Для простоты
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
            # Добавляем fallback endpoint
            _add_fallback_routes(app)
        
        # Настраиваем статические файлы React (упрощенно)
        _setup_react_static_files(app)
        
        # Добавляем базовые endpoints
        _add_basic_endpoints(app)
        
        # Простые обработчики ошибок
        _setup_error_handlers(app)
        
        logger.info("✅ Minimal RAG FastAPI application created successfully")
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
                "version": "1.0.0",
                "system": "Minimal RAG",
                "models": {
                    "llm": "google/flan-t5-small",
                    "embedding": "sentence-transformers/all-MiniLM-L6-v2"
                },
                "services": {
                    "document_service": services.get("document_service_available", False),
                    "llm_service": services.get("llm_available", False)
                },
                "memory_target": "<1GB RAM",
                "platform": "HuggingFace Spaces" if services.get("huggingface_spaces") else "Local"
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
        """Информация о статусе API"""
        try:
            from app.config import get_environment_info
            env_info = get_environment_info()
            
            return {
                "api": "Minimal RAG System v1.0",
                "status": "running",
                "features": {
                    "flan_t5_integration": True,
                    "semantic_search": True,
                    "document_upload": True,
                    "multilingual": True,
                    "memory_optimized": True
                },
                "models": {
                    "llm": env_info["model"],
                    "embedding": env_info["embedding_model"],
                    "memory_estimate": env_info["memory_target"]
                },
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
                "api": "Minimal RAG System v1.0",
                "status": "running",
                "error": f"Config error: {e}",
                "basic_endpoints": ["/health", "/docs"]
            }

def _add_fallback_routes(app):
    """Добавляет fallback роуты если API не загрузились"""
    
    @app.get("/api")
    async def api_fallback():
        return {
            "message": "Minimal RAG System",
            "version": "1.0.0",
            "status": "❌ API routes not configured",
            "models": {
                "llm": "google/flan-t5-small",
                "embedding": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "available_endpoints": ["/health", "/api-status", "/docs"],
            "issue": "API modules failed to load"
        }
    
    @app.post("/api/user/chat")
    async def chat_fallback():
        return {
            "response": "Chat service is initializing. FLAN-T5 model loading...",
            "sources": []
        }

def _setup_react_static_files(app):
    """Настраивает статические файлы React (упрощенно)"""
    try:
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse
        
        # Простые пути для статики
        static_paths = [
            Path(__file__).parent.parent / "static",  # HF Spaces
            Path(__file__).parent.parent.parent / "frontend" / "build"  # Local
        ]
        
        react_build_path = None
        for path in static_paths:
            if path.exists() and (path / "index.html").exists():
                react_build_path = path
                break
        
        if react_build_path:
            # Монтируем статические файлы
            static_dir = react_build_path / "static"
            if static_dir.exists():
                app.mount("/static", StaticFiles(directory=static_dir), name="react_static")
            
            # Корневой маршрут для React
            @app.get("/", include_in_schema=False)
            async def serve_react_app():
                return FileResponse(react_build_path / "index.html", media_type="text/html")
            
            logger.info(f"✅ React SPA mounted: {react_build_path}")
        else:
            _setup_api_only_root(app)
    
    except Exception as e:
        logger.warning(f"⚠️ React setup failed: {e}")
        _setup_api_only_root(app)

def _setup_api_only_root(app):
    """Настраивает API-only режим"""
    @app.get("/")
    async def root_fallback():
        return {
            "message": "Minimal RAG System v1.0",
            "models": {
                "llm": "google/flan-t5-small (~300 MB)",
                "embedding": "all-MiniLM-L6-v2 (~90 MB)"
            },
            "memory_target": "<1GB RAM",
            "status": "API running, React frontend may not be built",
            "endpoints": {
                "api_docs": "/docs",
                "health": "/health",
                "chat": "/api/user/chat",
                "search": "/api/user/search",
                "admin": "/api/admin"
            },
            "quick_start": [
                "Try: POST /api/user/chat",
                "Or visit: /docs for interactive API"
            ]
        }

def _setup_error_handlers(app):
    """Настраивает простые обработчики ошибок"""
    
    @app.exception_handler(404)
    async def not_found_handler(request, exc):
        path = str(request.url.path)
        
        if path.startswith("/api/"):
            return JSONResponse(
                status_code=404,
                content={
                    "detail": f"API endpoint not found: {path}",
                    "available_endpoints": ["/api/user/chat", "/api/user/search"],
                    "documentation": "/docs"
                }
            )
        
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

def _create_emergency_app(error: Exception):
    """Создаёт аварийное приложение"""
    logger.error(f"Creating emergency app due to: {error}")
    
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        
        emergency_app = FastAPI(title="Minimal RAG - Emergency Mode")
        
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
                "message": "Minimal RAG failed to initialize properly",
                "timestamp": time.time(),
                "models": {
                    "target_llm": "google/flan-t5-small",
                    "target_embedding": "sentence-transformers/all-MiniLM-L6-v2"
                },
                "recommendations": [
                    "Check transformers installation",
                    "Verify sentence-transformers availability",
                    "Check HuggingFace Hub access",
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