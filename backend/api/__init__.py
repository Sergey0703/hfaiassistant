# ====================================
# ФАЙЛ: backend/api/__init__.py (ПОЛНАЯ ВЕРСИЯ)
# ====================================

"""
API Package - Конфигурация всех API маршрутов
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import time

logger = logging.getLogger(__name__)

def configure_fastapi_app(app: FastAPI):
    """Конфигурирует все API маршруты"""
    
    initialization_status = {
        "user_routes": False,
        "admin_routes": False,
        "total_routes": 0,
        "errors": []
    }
    
    try:
        logger.info("🚀 Configuring API routes...")
        
        # ====================================
        # ПОЛЬЗОВАТЕЛЬСКИЕ МАРШРУТЫ
        # ====================================
        try:
            logger.info("📱 Loading user routes...")
            
            # Импортируем пользовательские роутеры
            from api.user.chat import router as chat_router
            from api.user.search import router as search_router
            
            # Подключаем пользовательские маршруты
            app.include_router(
                chat_router, 
                prefix="/api/user", 
                tags=["User Chat"],
                responses={
                    404: {"description": "Not found"},
                    500: {"description": "Internal server error"}
                }
            )
            
            app.include_router(
                search_router, 
                prefix="/api/user", 
                tags=["User Search"],
                responses={
                    404: {"description": "Not found"},
                    500: {"description": "Internal server error"}
                }
            )
            
            initialization_status["user_routes"] = True
            initialization_status["total_routes"] += 4  # Примерное количество user routes
            logger.info("✅ User routes configured successfully")
            
        except ImportError as e:
            error_msg = f"User routes import failed: {e}"
            logger.error(f"❌ {error_msg}")
            initialization_status["errors"].append(error_msg)
            
            # Создаем fallback user routes
            @app.get("/api/user/status")
            async def user_status():
                return {
                    "status": "❌ User routes not available",
                    "error": str(e),
                    "message": "User API modules missing"
                }
        
        # ====================================
        # АДМИНСКИЕ МАРШРУТЫ
        # ====================================
        try:
            logger.info("🔧 Loading admin routes...")
            
            # Импортируем админские роутеры
            from api.admin.documents import router as admin_docs_router
            from api.admin.scraper import router as admin_scraper_router
            from api.admin.stats import router as admin_stats_router
            from api.admin.llm import router as admin_llm_router
            
            # Подключаем админские маршруты
            app.include_router(
                admin_docs_router, 
                prefix="/api/admin", 
                tags=["Admin Documents"],
                responses={
                    404: {"description": "Not found"},
                    500: {"description": "Internal server error"}
                }
            )
            
            app.include_router(
                admin_scraper_router, 
                prefix="/api/admin", 
                tags=["Admin Scraper"],
                responses={
                    404: {"description": "Not found"},
                    500: {"description": "Internal server error"}
                }
            )
            
            app.include_router(
                admin_stats_router, 
                prefix="/api/admin", 
                tags=["Admin Statistics"],
                responses={
                    404: {"description": "Not found"},
                    500: {"description": "Internal server error"}
                }
            )
            
            app.include_router(
                admin_llm_router, 
                prefix="/api/admin", 
                tags=["Admin LLM"],
                responses={
                    404: {"description": "Not found"},
                    500: {"description": "Internal server error"}
                }
            )
            
            initialization_status["admin_routes"] = True
            initialization_status["total_routes"] += 15  # Примерное количество admin routes
            logger.info("✅ Admin routes configured successfully")
            
        except ImportError as e:
            error_msg = f"Admin routes import failed: {e}"
            logger.error(f"❌ {error_msg}")
            initialization_status["errors"].append(error_msg)
            
            # Создаем fallback admin routes
            @app.get("/api/admin/status")
            async def admin_status():
                return {
                    "status": "❌ Admin routes not available",
                    "error": str(e),
                    "message": "Admin API modules missing",
                    "recommendations": [
                        "Check that all admin API files exist",
                        "Verify imports in admin modules",
                        "Check dependencies installation"
                    ]
                }
        
        # ====================================
        # ОБЩИЕ API МАРШРУТЫ
        # ====================================
        
        @app.get("/api")
        async def api_root():
            """Корневой API endpoint с информацией о доступных маршрутах"""
            return {
                "message": "Legal Assistant API",
                "version": "2.0.0",
                "status": "operational",
                "initialization": initialization_status,
                "endpoints": {
                    "user": {
                        "chat": "/api/user/chat",
                        "search": "/api/user/search",
                        "history": "/api/user/chat/history"
                    },
                    "admin": {
                        "documents": "/api/admin/documents",
                        "scraper": "/api/admin/scraper",
                        "stats": "/api/admin/stats", 
                        "llm": "/api/admin/llm"
                    },
                    "system": {
                        "health": "/health",
                        "docs": "/docs",
                        "redoc": "/redoc"
                    }
                },
                "documentation": {
                    "swagger": "/docs",
                    "redoc": "/redoc",
                    "openapi": "/openapi.json"
                }
            }
        
        @app.get("/api/health")
        async def api_health():
            """Проверка здоровья API"""
            try:
                from app.dependencies import get_services_status
                services_status = get_services_status()
                
                overall_status = "healthy"
                issues = []
                
                # Проверяем статус маршрутов
                if not initialization_status["user_routes"]:
                    overall_status = "degraded"
                    issues.append("User routes not loaded")
                
                if not initialization_status["admin_routes"]:
                    overall_status = "degraded" if overall_status == "healthy" else "unhealthy"
                    issues.append("Admin routes not loaded")
                
                # Проверяем сервисы
                if not services_status.get("document_service_available", False):
                    overall_status = "degraded" if overall_status == "healthy" else "unhealthy"
                    issues.append("Document service unavailable")
                
                response_data = {
                    "status": overall_status,
                    "timestamp": time.time(),
                    "api_routes": initialization_status,
                    "services": services_status,
                    "total_routes_loaded": initialization_status["total_routes"]
                }
                
                if issues:
                    response_data["issues"] = issues
                    response_data["recommendations"] = [
                        "Check server logs for detailed errors",
                        "Verify all API modules are present",
                        "Restart server after fixing issues"
                    ]
                
                # Возвращаем соответствующий HTTP статус
                status_code = 200 if overall_status == "healthy" else (503 if overall_status == "unhealthy" else 207)
                return JSONResponse(content=response_data, status_code=status_code)
                
            except Exception as e:
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "unhealthy",
                        "error": str(e),
                        "timestamp": time.time(),
                        "message": "API health check failed"
                    }
                )
        
        @app.get("/api/routes")
        async def list_api_routes():
            """Список всех доступных API маршрутов"""
            routes_info = []
            
            for route in app.routes:
                if hasattr(route, 'methods') and hasattr(route, 'path'):
                    route_info = {
                        "path": route.path,
                        "methods": list(route.methods),
                        "name": getattr(route, 'name', 'unnamed'),
                        "tags": getattr(route, 'tags', [])
                    }
                    
                    # Фильтруем только API маршруты
                    if route.path.startswith('/api/'):
                        routes_info.append(route_info)
            
            return {
                "total_api_routes": len(routes_info),
                "routes": sorted(routes_info, key=lambda x: x['path']),
                "initialization_status": initialization_status
            }
        
        # ====================================
        # ФИНАЛЬНЫЙ СТАТУС ИНИЦИАЛИЗАЦИИ
        # ====================================
        
        total_errors = len(initialization_status["errors"])
        
        if total_errors == 0:
            logger.info(f"✅ All API routes configured successfully ({initialization_status['total_routes']} routes)")
        else:
            logger.warning(f"⚠️ API routes configured with {total_errors} errors")
            for error in initialization_status["errors"]:
                logger.warning(f"   - {error}")
        
        # Добавляем глобальный обработчик ошибок
        @app.exception_handler(404)
        async def not_found_handler(request, exc):
            return JSONResponse(
                status_code=404,
                content={
                    "detail": f"Endpoint not found: {request.url.path}",
                    "available_endpoints": "/api",
                    "documentation": "/docs",
                    "suggestion": "Check /api for available endpoints"
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
                    "help": "Check server logs for details"
                }
            )
        
        logger.info("🎯 API configuration completed")
        
    except Exception as e:
        logger.error(f"❌ Critical error during API configuration: {e}")
        
        # Создаем минимальный API в случае критической ошибки
        @app.get("/api/error")
        async def api_error():
            return {
                "status": "❌ API configuration failed",
                "error": str(e),
                "message": "Critical API initialization error",
                "timestamp": time.time()
            }
        
        raise

def get_api_info():
    """Возвращает информацию об API для диагностики"""
    try:
        # Пытаемся импортировать основные модули для проверки
        available_modules = {
            "user_chat": False,
            "user_search": False,
            "admin_documents": False,
            "admin_scraper": False,
            "admin_stats": False,
            "admin_llm": False
        }
        
        # Проверяем user модули
        try:
            from api.user.chat import router
            available_modules["user_chat"] = True
        except ImportError:
            pass
        
        try:
            from api.user.search import router
            available_modules["user_search"] = True
        except ImportError:
            pass
        
        # Проверяем admin модули
        try:
            from api.admin.documents import router
            available_modules["admin_documents"] = True
        except ImportError:
            pass
        
        try:
            from api.admin.scraper import router
            available_modules["admin_scraper"] = True
        except ImportError:
            pass
        
        try:
            from api.admin.stats import router
            available_modules["admin_stats"] = True
        except ImportError:
            pass
        
        try:
            from api.admin.llm import router
            available_modules["admin_llm"] = True
        except ImportError:
            pass
        
        available_count = sum(available_modules.values())
        total_modules = len(available_modules)
        
        return {
            "status": "healthy" if available_count == total_modules else "partial",
            "summary": {
                "total_routes": available_count * 3,  # Примерная оценка
                "available_modules": available_count,
                "total_modules": total_modules,
                "completion_rate": f"{available_count}/{total_modules}"
            },
            "modules": available_modules,
            "missing_modules": [
                module for module, available in available_modules.items() 
                if not available
            ]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "summary": {
                "total_routes": 0,
                "available_modules": 0
            }
        }

# Экспорт основных функций
__all__ = [
    "configure_fastapi_app",
    "get_api_info"
]