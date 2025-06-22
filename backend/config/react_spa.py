# backend/config/react_spa.py
"""
Конфигурация и настройка React SPA для FastAPI
"""

import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

# ====================================
# КОНСТАНТЫ ДЛЯ REACT SPA
# ====================================

# Путь к build папке React (относительно main.py)
REACT_BUILD_PATH = Path(__file__).parent.parent.parent / "frontend" / "build"

# Список React assets файлов для прямой раздачи
REACT_ASSETS = [
    "manifest.json", 
    "favicon.ico", 
    "robots.txt", 
    "logo192.png", 
    "logo512.png",
    "asset-manifest.json"
]

# API роуты которые НЕ должны обрабатываться как React SPA
API_ROUTES = [
    "/api/",
    "/docs",
    "/redoc", 
    "/openapi.json",
    "/health",
    "/hf-spaces-health",
    "/model-status",
    "/timeout-status",
    "/memory-status",
    "/startup-progress"
]

# ====================================
# ФУНКЦИЯ НАСТРОЙКИ REACT SPA
# ====================================

def setup_react_spa(app: FastAPI) -> bool:
    """
    Настраивает FastAPI для раздачи React SPA
    
    Returns:
        bool: True если настройка прошла успешно, False если React build не найден
    """
    
    # Проверяем существует ли build папка
    if not REACT_BUILD_PATH.exists():
        logger.warning(f"⚠️ React build folder not found: {REACT_BUILD_PATH}")
        logger.warning("   Run 'npm run build' in frontend directory first")
        return False
    
    logger.info(f"📁 Setting up React SPA from: {REACT_BUILD_PATH}")
    
    # Маршрут для статических файлов React (CSS, JS, images)
    static_path = REACT_BUILD_PATH / "static"
    if static_path.exists():
        app.mount(
            "/static", 
            StaticFiles(directory=static_path), 
            name="react_static"
        )
        logger.info("   • Static files mounted: /static/*")
    
    # Специальный обработчик для SPA - все неизвестные роуты → index.html
    @app.exception_handler(StarletteHTTPException)
    async def spa_handler(request, exc):
        """
        Обработчик для React SPA routing
        Все 404 ошибки (кроме API роутов) перенаправляем на index.html
        """
        if exc.status_code == 404:
            path = request.url.path
            
            # Проверяем что это не API роут
            is_api_route = any(path.startswith(api_route) for api_route in API_ROUTES)
            
            if not is_api_route:
                # Возвращаем index.html для React роутинга
                index_path = REACT_BUILD_PATH / "index.html"
                if index_path.exists():
                    return FileResponse(index_path, media_type="text/html")
        
        # Для API роутов и других случаев - стандартная обработка
        from fastapi.exception_handlers import http_exception_handler
        return await http_exception_handler(request, exc)
    
    # Корневой роут теперь отдает React приложение
    @app.get("/", include_in_schema=False)
    async def serve_react_app():
        """Serve React app at root path"""
        index_path = REACT_BUILD_PATH / "index.html"
        if index_path.exists():
            return FileResponse(index_path, media_type="text/html")
        else:
            return {
                "message": "React app not built yet",
                "instructions": [
                    "1. cd frontend",
                    "2. npm install", 
                    "3. npm run build",
                    "4. Restart FastAPI server"
                ],
                "build_path": str(REACT_BUILD_PATH)
            }
    
    # Дополнительные роуты для React assets (manifest, favicon, etc.)
    _setup_react_assets(app)
    
    logger.info("✅ React SPA configured successfully")
    logger.info("   • Static files: /static/*")
    logger.info("   • SPA routing: all non-API routes → index.html")
    logger.info("   • React app available at: /")
    
    return True

def _setup_react_assets(app: FastAPI):
    """Настраивает роуты для React assets файлов"""
    
    for asset in REACT_ASSETS:
        asset_path = REACT_BUILD_PATH / asset
        if asset_path.exists():
            # Создаем роут для каждого asset файла
            def create_asset_handler(asset_name: str):
                async def serve_asset():
                    return FileResponse(REACT_BUILD_PATH / asset_name)
                return serve_asset
            
            app.get(f"/{asset}", include_in_schema=False)(create_asset_handler(asset))
            logger.debug(f"   • Asset route: /{asset}")

# ====================================
# API INFO ENDPOINT
# ====================================

def setup_api_info_endpoint(app: FastAPI):
    """Настраивает endpoint с информацией об API (fallback когда React не собран)"""
    
    @app.get("/api-info", include_in_schema=False)
    async def api_info():
        """API информация (доступна когда React не собран)"""
        react_status = "built" if REACT_BUILD_PATH.exists() else "not_built"
        
        return {
            "api": "Legal Assistant API v2.0",
            "react_status": react_status,
            "endpoints": {
                "api_docs": "/docs",
                "redoc": "/redoc", 
                "health": "/health",
                "hf_health": "/hf-spaces-health",
                "model_status": "/model-status",
                "timeout_status": "/timeout-status",
                "memory_status": "/memory-status"
            },
            "react_build_path": str(REACT_BUILD_PATH),
            "react_assets": REACT_ASSETS,
            "instructions": {
                "to_build_react": [
                    "cd frontend",
                    "npm install",
                    "npm run build" 
                ]
            }
        }

# ====================================
# UTILITY FUNCTIONS
# ====================================

def is_react_built() -> bool:
    """Проверяет собрано ли React приложение"""
    return REACT_BUILD_PATH.exists() and (REACT_BUILD_PATH / "index.html").exists()

def get_react_info() -> dict:
    """Возвращает информацию о React SPA"""
    return {
        "build_path": str(REACT_BUILD_PATH),
        "is_built": is_react_built(),
        "assets": REACT_ASSETS,
        "api_routes_excluded": API_ROUTES,
        "static_files_path": str(REACT_BUILD_PATH / "static") if REACT_BUILD_PATH.exists() else None
    }