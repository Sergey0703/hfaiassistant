# ====================================
# ФАЙЛ: backend/app/__init__.py (ИСПРАВЛЕННАЯ ВЕРСИЯ БЕЗ КОРНЕВОГО МАРШРУТА)
# Заменить существующий файл полностью
# ====================================

"""
FastAPI Application Factory - Исправленная версия для HuggingFace Spaces
ИСПРАВЛЕНИЯ: Убран корневой маршрут @app.get("/") для корректной работы React SPA
"""

import logging
import sys
import os
import time
from pathlib import Path
from typing import Dict, Any
from contextlib import asynccontextmanager

# Добавляем пути для импорта
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: "FastAPI"):
    """
    Lifespan context manager для FastAPI - современный способ startup/shutdown
    Заменяет устаревшие @app.on_event("startup")/@app.on_event("shutdown")
    """
    # Startup logic
    logger.info("🚀 FastAPI application starting up...")
    
    # НЕ вызываем init_services() - используем lazy loading
    logger.info("🔄 Lazy initialization enabled - services will load on demand")
    
    # Просто помечаем что приложение готово
    app.state.startup_time = time.time()
    app.state.lazy_loading = True
    
    logger.info("✅ Application startup completed with lazy loading")
    
    yield  # Приложение работает
    
    # Shutdown logic
    logger.info("🛑 FastAPI application shutting down...")
    
    # Очищаем ресурсы если нужно
    try:
        # Закрываем LLM сервис если он инициализирован
        from app.dependencies import llm_service, _llm_service_initialized
        if _llm_service_initialized and hasattr(llm_service, 'close'):
            await llm_service.close()
            logger.info("🔒 LLM service closed")
        
        # Закрываем scraper сервис если он инициализирован  
        from app.dependencies import scraper, _scraper_initialized
        if _scraper_initialized and hasattr(scraper, 'close'):
            await scraper.close()
            logger.info("🔒 Scraper service closed")
            
    except Exception as e:
        logger.warning(f"⚠️ Error during cleanup: {e}")
    
    logger.info("✅ Application shutdown completed")

def create_app() -> "FastAPI":
    """
    Создает и настраивает FastAPI приложение с исправлениями для HF Spaces
    ИСПРАВЛЕНИЕ: Убран корневой маршрут @app.get("/") для React SPA
    """
    try:
        # Импорты FastAPI
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
        
        logger.info("🚀 Creating FastAPI application...")
        
        # Загружаем конфигурацию
        try:
            from app.config import settings, API_METADATA, API_TAGS
            app_config = {
                "title": API_METADATA["title"],
                "version": API_METADATA["version"], 
                "description": API_METADATA["description"],
                "openapi_tags": API_TAGS,
                "lifespan": lifespan  # ИСПРАВЛЕНИЕ: Используем современный lifespan
            }
            config_loaded = True
        except ImportError as e:
            logger.warning(f"Config import failed: {e}, using defaults")
            app_config = {
                "title": "Legal Assistant API",
                "version": "2.0.0",
                "description": "AI Legal Assistant with GPTQ model support",
                "lifespan": lifespan
            }
            config_loaded = False
        
        # Создаем приложение
        app = FastAPI(**app_config)
        
        # Отслеживаем статус инициализации
        initialization_status = {
            "config_loaded": config_loaded,
            "api_routes_loaded": False,
            "middleware_loaded": False,
            "errors": [],
            "lazy_loading_enabled": True,  # НОВОЕ: отмечаем что используем lazy loading
            "lifespan_configured": True,   # НОВОЕ: отмечаем что lifespan настроен
            "react_spa_ready": False       # НОВОЕ: статус React SPA
        }
        
        # Сохраняем статус в app.state для доступа из endpoints
        app.state.initialization_status = initialization_status
        app.state.hf_spaces = os.getenv("SPACE_ID") is not None
        
        # Настраиваем CORS
        try:
            cors_origins = getattr(settings, 'CORS_ORIGINS', ["*"]) if config_loaded else ["*"]
            app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            logger.info("✅ CORS middleware configured")
        except Exception as e:
            logger.error(f"❌ CORS configuration failed: {e}")
            initialization_status["errors"].append(f"CORS setup failed: {e}")
        
        # ====================================
        # ИСПРАВЛЕНИЕ: УБИРАЕМ КОРНЕВОЙ МАРШРУТ!
        # ====================================
        # СТАРЫЙ КОД (УДАЛЁН):
        # @app.get("/")
        # async def root():
        #     return {"message": "Legal Assistant API with GPTQ model support", ...}
        
        # ТЕПЕРЬ КОРНЕВОЙ ПУТЬ "/" БУДЕТ ДОСТУПЕН ДЛЯ REACT SPA!
        
        # ====================================
        # БАЗОВЫЕ API МАРШРУТЫ (НЕ КОРНЕВЫЕ)
        # ====================================
        
        @app.get("/health")
        async def health_check():
            """Улучшенная проверка здоровья с lazy loading информацией"""
            try:
                # Не форсируем инициализацию сервисов для health check
                from app.dependencies import get_services_status
                services_status = get_services_status()
                
                # Определяем статус на основе lazy loading
                overall_status = "healthy"  # По умолчанию здоровы с lazy loading
                issues = []
                recommendations = []
                
                # Проверяем ошибки инициализации
                if initialization_status["errors"]:
                    overall_status = "degraded"
                    issues.extend(initialization_status["errors"])
                
                # Добавляем рекомендации для lazy loading
                if not services_status.get("llm_available", False):
                    recommendations.append("GPTQ model will load on first chat request")
                
                if not services_status.get("chromadb_enabled", False):
                    recommendations.append("ChromaDB will initialize on first document search")
                
                # Проверяем React SPA
                react_ready = initialization_status.get("react_spa_ready", False)
                
                response_data = {
                    "status": overall_status,
                    "timestamp": time.time(),
                    "platform": "HuggingFace Spaces" if app.state.hf_spaces else "Local",
                    "initialization": initialization_status,
                    "services": services_status,
                    "react_spa": {
                        "enabled": react_ready,
                        "note": "React SPA mounted on root path (/)"
                    },
                    "lazy_loading": {
                        "enabled": True,
                        "description": "Services initialize on first use",
                        "benefits": [
                            "Faster application startup",
                            "Reduced memory usage at startup", 
                            "Graceful service degradation",
                            "Better error isolation"
                        ]
                    },
                    "version": app_config.get("version", "2.0.0")
                }
                
                if issues:
                    response_data["issues"] = issues
                
                if recommendations:
                    response_data["recommendations"] = recommendations
                
                # Возвращаем правильный HTTP статус
                status_code = 200 if overall_status == "healthy" else 207  # 207 = Multi-Status
                
                return JSONResponse(content=response_data, status_code=status_code)
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "unhealthy", 
                        "error": str(e),
                        "timestamp": time.time(),
                        "message": "Health check failed",
                        "platform": "HuggingFace Spaces" if app.state.hf_spaces else "Local"
                    }
                )
        
        @app.get("/api-status")
        async def api_status():
            """Информация о статусе API (замена корневого маршрута)"""
            return {
                "message": "Legal Assistant API with GPTQ model support",
                "version": app_config.get("version", "2.0.0"),
                "status": "healthy",
                "platform": "HuggingFace Spaces" if app.state.hf_spaces else "Local",
                "model": "TheBloke/Llama-2-7B-Chat-GPTQ",
                "features": {
                    "lazy_loading": True,
                    "gptq_support": True,
                    "ukrainian_language": True,
                    "vector_search": "Available on demand",
                    "demo_responses": "Available immediately",
                    "react_spa": "Mounted on root path (/)"
                },
                "docs": "/docs",
                "health": "/health",
                "hf_health": "/hf-spaces-health",
                "model_status": "/model-status",
                "startup_progress": "/startup-progress",
                "note": "Root path (/) now serves React SPA instead of this JSON"
            }
        
        # Подключаем API роутеры
        try:
            from api import configure_fastapi_app
            configure_fastapi_app(app)
            logger.info("✅ API routes configured")
            initialization_status["api_routes_loaded"] = True
        except Exception as e:
            error_msg = f"Failed to configure API routes: {e}"
            logger.error(f"❌ {error_msg}")
            initialization_status["errors"].append(error_msg)
            
            # Добавляем fallback endpoint для диагностики
            @app.get("/api/status")
            async def api_routes_status():
                return {
                    "status": "❌ API routes NOT CONFIGURED",
                    "error": str(e),
                    "message": "Some API endpoints may be unavailable",
                    "available_endpoints": [
                        "GET /health - Health check",
                        "GET /api-status - API status (former root)",
                        "GET /docs - API documentation",
                        "GET /api/status - This endpoint"
                    ],
                    "recommendations": [
                        "Check that all API modules are present",
                        "Verify import dependencies",
                        "Some functionality may still work via direct endpoints"
                    ]
                }
        
        # Настраиваем middleware
        try:
            from app.middleware import setup_middleware
            setup_middleware(app)
            logger.info("✅ Middleware configured")
            initialization_status["middleware_loaded"] = True
        except Exception as e:
            error_msg = f"Middleware setup failed: {e}"
            logger.warning(f"⚠️ {error_msg}")
            initialization_status["errors"].append(error_msg)
            # Middleware не критичен, продолжаем без него
        
        # Глобальный обработчик ошибок
        @app.exception_handler(404)
        async def not_found_handler(request, exc):
            return JSONResponse(
                status_code=404,
                content={
                    "detail": f"Endpoint not found: {request.url.path}",
                    "available_endpoints": {
                        "health": "/health", 
                        "docs": "/docs",
                        "api_info": "/api-status",
                        "hf_spaces_health": "/hf-spaces-health"
                    },
                    "suggestion": "Check /docs for available endpoints",
                    "platform": "HuggingFace Spaces" if app.state.hf_spaces else "Local",
                    "note": "Root path (/) now serves React SPA"
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
                    "help": "Check server logs for details",
                    "platform": "HuggingFace Spaces" if app.state.hf_spaces else "Local",
                    "lazy_loading": "Services may still be initializing"
                }
            )
        
        # Финальный статус
        total_errors = len(initialization_status["errors"])
        
        if total_errors == 0:
            logger.info("✅ FastAPI application created successfully with lazy loading")
        else:
            logger.warning(f"⚠️ Application created with {total_errors} non-critical errors")
            for error in initialization_status["errors"]:
                logger.warning(f"   - {error}")
        
        logger.info("🔄 Lazy loading enabled - services will initialize on demand")
        logger.info("🎯 ИСПРАВЛЕНИЕ: Корневой маршрут (/) теперь доступен для React SPA")
        
        return app
        
    except ImportError as e:
        logger.error(f"❌ Missing critical dependencies: {e}")
        logger.error("Install: pip install fastapi uvicorn")
        
        # Создаем минимальное приложение для диагностики
        try:
            from fastapi import FastAPI
            fallback_app = FastAPI(title="Legal Assistant API - Dependency Error")
            
            @fallback_app.get("/api-status")
            async def dependency_error():
                return {
                    "status": "dependency_error",
                    "error": str(e),
                    "message": "Critical dependencies missing",
                    "required": ["fastapi", "uvicorn"],
                    "install_command": "pip install fastapi uvicorn",
                    "note": "Root path (/) available for React SPA after fixing dependencies"
                }
            
            return fallback_app
            
        except ImportError:
            # Даже FastAPI недоступен
            logger.critical("❌ FastAPI not available - cannot create any application")
            raise
    
    except Exception as e:
        logger.error(f"❌ Critical error during application creation: {e}")
        
        # Последняя попытка создать хоть что-то
        try:
            from fastapi import FastAPI
            emergency_app = FastAPI(title="Legal Assistant API - Emergency Mode")
            
            @emergency_app.get("/api-status")
            async def emergency_mode():
                return {
                    "status": "emergency_mode",
                    "error": str(e),
                    "message": "Application failed to initialize properly",
                    "timestamp": time.time(),
                    "platform": "HuggingFace Spaces" if os.getenv("SPACE_ID") else "Local",
                    "note": "Root path (/) available for React SPA in normal mode"
                }
            
            return emergency_app
            
        except Exception as final_error:
            logger.critical(f"❌ Cannot create even emergency application: {final_error}")
            raise

# Экспорт основных функций
__all__ = [
    "create_app",
    "lifespan"
]