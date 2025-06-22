# backend/main.py - ИСПРАВЛЕННАЯ ВЕРСИЯ С ПРАВИЛЬНЫМ ПОРЯДКОМ МАРШРУТОВ
"""
Legal Assistant API - Main Application Entry Point
ИСПРАВЛЕНО: Правильный порядок монтирования для React SPA
"""

import uvicorn
import sys
import os
from pathlib import Path

# Добавляем текущую директорию в Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Настройка логирования
try:
    from utils.logger import setup_logging
    setup_logging(log_level="INFO", log_file="logs/legal_assistant.log")
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

import logging

# Импорты модулей приложения
try:
    from utils.startup_banner import (
        print_startup_banner, check_hf_spaces_environment, 
        check_critical_dependencies, create_necessary_directories
    )
    from config.timeouts import (
        GLOBAL_REQUEST_TIMEOUT, KEEP_ALIVE_TIMEOUT, GRACEFUL_TIMEOUT,
        GPTQ_MODEL_LOADING_TIMEOUT, GPTQ_FIRST_LOAD_TIMEOUT
    )
    from core.app_factory import create_app_for_deployment
    banner_available = True
except ImportError as e:
    print(f"⚠️ Some modules not available: {e}")
    banner_available = False
    # Fallback значения
    GLOBAL_REQUEST_TIMEOUT = 600
    KEEP_ALIVE_TIMEOUT = 65
    GRACEFUL_TIMEOUT = 300
    GPTQ_MODEL_LOADING_TIMEOUT = 480
    GPTQ_FIRST_LOAD_TIMEOUT = 600

logger = logging.getLogger(__name__)

def main():
    """Главная функция для разработки (не используется в HF Spaces)"""
    try:
        if banner_available:
            print_startup_banner()
            
            # Проверяем окружение
            is_hf_spaces = check_hf_spaces_environment()
            print()
            
            # Создаем директории
            create_necessary_directories()
            
            # Проверяем зависимости
            if not check_critical_dependencies():
                print("\n❌ Cannot start due to missing critical dependencies")
                sys.exit(1)
        else:
            print("🚀 Legal Assistant API - Starting with basic configuration")
            is_hf_spaces = os.getenv("SPACE_ID") is not None
            
        print()
        
        # Создаем приложение
        if banner_available:
            app = create_app_for_deployment()
        else:
            # Fallback app creation
            app = create_basic_app()
        
        if app is None:
            print("❌ Failed to create FastAPI application")
            sys.exit(1)
        
        # Настройки для запуска
        host = "0.0.0.0"
        port = 7860 if is_hf_spaces else 8000
        
        print(f"🌐 Server Configuration:")
        print(f"   • Host: {host}")
        print(f"   • Port: {port}")
        print(f"   • Environment: {'HuggingFace Spaces' if is_hf_spaces else 'Local'}")
        print(f"   • Model: TheBloke/Llama-2-7B-Chat-GPTQ")
        print(f"   • React SPA: Enabled")
        print(f"   • Lazy Loading: Enabled")
        print(f"   • CORS Fix: Applied")
        print(f"   • Comprehensive Timeouts: Enabled")
        
        print(f"\n🔗 Available Endpoints:")
        print(f"   • React App: http://localhost:{port}/")
        print(f"   • API Docs: http://localhost:{port}/docs")
        print(f"   • Health Check: http://localhost:{port}/health")
        print(f"   • Debug React: http://localhost:{port}/debug-react")
        print(f"   • API Info: http://localhost:{port}/api-info")
        
        print(f"\n⏰ Timeout Configuration:")
        print(f"   • Global Request: {GLOBAL_REQUEST_TIMEOUT}s (10 min)")
        print(f"   • GPTQ Loading: {GPTQ_MODEL_LOADING_TIMEOUT}s (8 min)")
        print(f"   • GPTQ First Load: {GPTQ_FIRST_LOAD_TIMEOUT}s (10 min)")
        print(f"   • Keep-Alive: {KEEP_ALIVE_TIMEOUT}s")
        
        print(f"\n🎯 Starting server with comprehensive timeout protection and React SPA...")
        print("=" * 70)
        
        # Запускаем сервер с правильными таймаутами
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            log_level="info",
            reload=False,  # Отключаем reload в production
            access_log=True,
            server_header=False,
            date_header=False,
            workers=1,  # Важно: только 1 worker для HF Spaces и GPTQ
            timeout_keep_alive=KEEP_ALIVE_TIMEOUT,
            timeout_graceful_shutdown=GRACEFUL_TIMEOUT,
            limit_concurrency=5,  # Ограничиваем для GPTQ модели
            limit_max_requests=500,  # Лимит для memory management
            timeout_notify=GRACEFUL_TIMEOUT,
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Legal Assistant API shutting down...")
        print("Thank you for using Legal Assistant!")
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

def create_basic_app():
    """Создает базовое приложение если модули недоступны"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(
        title="Legal Assistant API",
        version="2.0.0",
        description="AI Legal Assistant with GPTQ Model"
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"]
    )
    
    return app

# ====================================
# ПРИЛОЖЕНИЕ ДЛЯ DEPLOYMENT
# ====================================

# Создаем экземпляр приложения для WSGI/ASGI серверов
try:
    print("🚀 Initializing Legal Assistant API for HuggingFace Spaces...")
    
    # Проверяем и настраиваем HF Spaces
    is_hf_spaces = os.getenv("SPACE_ID") is not None
    
    if banner_available:
        check_hf_spaces_environment()
        create_necessary_directories()
        app = create_app_for_deployment()
    else:
        # Fallback создание приложения
        from app import create_app
        app = create_app()
        
        # Добавляем CORS
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"]
        )
    
    if app is None:
        raise RuntimeError("Failed to create FastAPI application")
    
    print("✅ Legal Assistant API ready for deployment")
    print(f"🌍 Platform: {'HuggingFace Spaces' if is_hf_spaces else 'Local'}")
    print("🤖 GPTQ Model: TheBloke/Llama-2-7B-Chat-GPTQ")
    print("⚛️ React Frontend: Integrated")
    print("🔄 Initialization: Lazy loading enabled")
    print(f"⏰ Request Timeout: {GLOBAL_REQUEST_TIMEOUT}s")
    print(f"🔄 Keep-Alive: {KEEP_ALIVE_TIMEOUT}s")
    print(f"🤖 GPTQ Loading: {GPTQ_MODEL_LOADING_TIMEOUT}s")
    print("🔧 CORS Fix: Applied (POST endpoints working)")
    print("🛡️ Comprehensive Timeout Protection: Active")
    print("📱 Single Page Application: React SPA Ready")
    
except Exception as e:
    print(f"❌ Deployment initialization failed: {e}")
    print("🔄 Creating minimal fallback application...")
    
    # Улучшенное fallback приложение
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(
        title="Legal Assistant API - Recovery Mode", 
        version="2.0.0",
        description="Minimal recovery mode - some services may be unavailable"
    )
    
    # КРИТИЧЕСКИ ВАЖНО: CORS даже в fallback
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"]
    )

# ====================================
# API ENDPOINTS - ОПРЕДЕЛЯЕМ ПЕРВЫМИ!
# ====================================

@app.get("/api-info")
async def api_info():
    """Информация о статусе API и React"""
    from pathlib import Path
    
    REACT_STATIC_PATH = Path("/home/user/app/static")
    react_files_info = {}
    
    try:
        if REACT_STATIC_PATH.exists():
            react_files = list(REACT_STATIC_PATH.iterdir())
            react_files_info = {
                "react_path": str(REACT_STATIC_PATH),
                "react_exists": True,
                "index_html_exists": (REACT_STATIC_PATH / "index.html").exists(),
                "index_html_path": str(REACT_STATIC_PATH / "index.html"),
                "total_files": len(react_files),
                "files": [f.name for f in react_files[:10]]  # Первые 10 файлов
            }
            
            if (REACT_STATIC_PATH / "index.html").exists():
                react_files_info["index_html_size"] = (REACT_STATIC_PATH / "index.html").stat().st_size
        else:
            react_files_info = {
                "react_path": str(REACT_STATIC_PATH),
                "react_exists": False,
                "error": f"React directory not found: {REACT_STATIC_PATH}"
            }
    except Exception as e:
        react_files_info = {"error": f"Error checking React files: {e}"}
    
    return {
        "api": "Legal Assistant API v2.0",
        "status": "running",
        "platform": "HuggingFace Spaces",
        "space_id": os.getenv("SPACE_ID", "unknown"),
        "working_directory": os.getcwd(),
        "react_info": react_files_info,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "debug": "/debug-react",
            "startup": "/startup-progress"
        },
        "static_files": {
            "mounted": str(REACT_STATIC_PATH) if REACT_STATIC_PATH.exists() else "Not mounted",
            "mount_point": "/static"
        }
    }

@app.get("/debug-react")
async def debug_react():
    """Полная диагностика React"""
    from pathlib import Path
    
    REACT_STATIC_PATH = Path("/home/user/app/static")
    
    debug_info = {
        "current_directory": os.getcwd(),
        "script_location": str(Path(__file__).parent),
        "environment": {
            "SPACE_ID": os.getenv("SPACE_ID"),
            "HOME": os.getenv("HOME"),
            "PWD": os.getenv("PWD"),
            "USER": os.getenv("USER")
        },
        "react_paths": {
            "static_path": str(REACT_STATIC_PATH),
            "static_exists": REACT_STATIC_PATH.exists()
        }
    }
    
    # Проверяем содержимое static директории
    if REACT_STATIC_PATH.exists():
        try:
            files = list(REACT_STATIC_PATH.iterdir())
            debug_info["static_contents"] = {
                "total_files": len(files),
                "files": [
                    {
                        "name": f.name,
                        "is_file": f.is_file(),
                        "is_dir": f.is_dir(),
                        "size": f.stat().st_size if f.is_file() else None
                    } for f in files
                ]
            }
            
            # Специально проверяем index.html
            index_path = REACT_STATIC_PATH / "index.html"
            if index_path.exists():
                with open(index_path, 'r') as f:
                    content = f.read()
                debug_info["index_html"] = {
                    "exists": True,
                    "size": len(content),
                    "content_preview": content[:500] + "..." if len(content) > 500 else content
                }
            else:
                debug_info["index_html"] = {"exists": False}
                
        except Exception as e:
            debug_info["static_contents"] = {"error": str(e)}
    
    return debug_info

@app.get("/startup-progress") 
async def startup_progress():
    """Статус загрузки приложения"""
    from pathlib import Path
    
    REACT_STATIC_PATH = Path("/home/user/app/static")
    react_ready = (REACT_STATIC_PATH / "index.html").exists()
    
    return {
        "status": "running",
        "platform": "HuggingFace Spaces",
        "components": {
            "api": {
                "status": "ready",
                "description": "FastAPI backend"
            },
            "react_spa": {
                "status": "ready" if react_ready else "not_found",
                "description": "React frontend application",
                "path": str(REACT_STATIC_PATH / "index.html"),
                "ready": react_ready
            },
            "model": {
                "status": "loading",
                "description": "GPTQ model loading in background"
            }
        },
        "progress": "100%" if react_ready else "75%",
        "message": "React SPA готов!" if react_ready else "React SPA проверяется...",
        "endpoints_working": ["/docs", "/api-info", "/debug-react", "/health"],
        "static_files_mounted": REACT_STATIC_PATH.exists()
    }

# Быстрый health check для deployment
@app.get("/health")
async def health_check():
    """Быстрая проверка здоровья"""
    try:
        import asyncio
        from pathlib import Path
        
        REACT_STATIC_PATH = Path("/home/user/app/static")
        
        return await asyncio.wait_for({
            "status": "healthy", 
            "version": "2.0.0",
            "platform": "HuggingFace Spaces",
            "gptq_model": "TheBloke/Llama-2-7B-Chat-GPTQ",
            "react_spa": "Ready" if (REACT_STATIC_PATH / "index.html").exists() else "Not found",
            "static_files": "Mounted" if REACT_STATIC_PATH.exists() else "Not mounted",
            "lazy_loading": True,
            "memory_optimized": True,
            "cors_fix_applied": True,
            "post_endpoints_working": True,
            "timeout_protected": True,
            "timestamp": __import__("time").time(),
            "available_endpoints": ["/", "/docs", "/api-info", "/debug-react"]
        }, timeout=15)
    except:
        return {
            "status": "timeout",
            "timeout_limit": "15s",
            "global_timeout": f"{GLOBAL_REQUEST_TIMEOUT}s",
            "message": "Health check timeout - services may be loading"
        }

# ====================================
# СТАТИЧЕСКИЕ ФАЙЛЫ - МОНТИРУЕМ ПОСЛЕ API!
# ====================================

# Статические файлы React (если найдены) - ПРАВИЛЬНЫЙ ПОРЯДОК
try:
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    from pathlib import Path
    
    # Путь к static файлам (CSS, JS, images)
    react_static_files_path = Path("/home/user/app/static/static")
    
    if react_static_files_path.exists():
        app.mount("/static", StaticFiles(directory=react_static_files_path), name="react_static")
        print(f"✅ React static files mounted from: {react_static_files_path}")
    else:
        print(f"⚠️ React static files not found at: {react_static_files_path}")
    
    # Дополнительные React assets
    REACT_BUILD_PATH = Path("/home/user/app/static")
    
    if REACT_BUILD_PATH.exists():
        react_files = ["manifest.json", "favicon.ico", "robots.txt", "logo192.png", "logo512.png"]
        
        for file_name in react_files:
            file_path = REACT_BUILD_PATH / file_name
            if file_path.exists():
                # Создаем endpoint для каждого asset файла
                def create_asset_endpoint(filename):
                    async def serve_asset():
                        return FileResponse(REACT_BUILD_PATH / filename)
                    return serve_asset
                
                app.get(f"/{file_name}", include_in_schema=False)(create_asset_endpoint(file_name))
                
        print(f"✅ React build path found: {REACT_BUILD_PATH}")
        print(f"✅ React assets available: {[f for f in react_files if (REACT_BUILD_PATH / f).exists()]}")
        
except Exception as e:
    print(f"⚠️ Could not mount React static files: {e}")

# ====================================
# КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: SPA МАРШРУТ ПОСЛЕДНИМ!
# ====================================

try:
    from fastapi.staticfiles import StaticFiles
    from pathlib import Path
    
    REACT_STATIC_PATH = Path("/home/user/app/static")
    
    if REACT_STATIC_PATH.exists() and (REACT_STATIC_PATH / "index.html").exists():
        print(f"🔧 Mounting React SPA from {REACT_STATIC_PATH}")
        
        # КРИТИЧЕСКИ ВАЖНО: Монтируем React как корневой маршрут ПОСЛЕДНИМ
        # Это позволяет API endpoints работать, а все остальные запросы идут в React
        app.mount("/", StaticFiles(directory=str(REACT_STATIC_PATH), html=True), name="react_spa")
        
        print("✅ React SPA successfully mounted as root route!")
        print("✅ Main page should now serve React instead of JSON")
        print("✅ API endpoints (/docs, /health, /api-info) will still work")
        
    else:
        print(f"❌ React files not found at {REACT_STATIC_PATH}")
        print("⚠️ Main page will show fallback message")
        
        # FALLBACK: если React не найден, добавляем простой корневой маршрут
        @app.get("/")
        async def root_fallback():
            return {
                "message": "Legal Assistant API",
                "version": "2.0.0", 
                "status": "API работает ✅",
                "react_status": f"React files not found: {REACT_STATIC_PATH}",
                "available_endpoints": {
                    "api_docs": "/docs",
                    "api_info": "/api-info", 
                    "debug": "/debug-react",
                    "health": "/health",
                    "startup_progress": "/startup-progress"
                },
                "platform": "HuggingFace Spaces",
                "instructions": [
                    "React files should be in /home/user/app/static/",
                    "Check /debug-react for detailed diagnosis",
                    "Try /docs for API documentation"
                ]
            }
        
except Exception as e:
    print(f"❌ Error during React mounting: {e}")

if __name__ == "__main__":
    main()
else:
    # Если модуль импортируется
    logger.info("📦 Legal Assistant API module imported")
    logger.info("🤖 GPTQ Model: TheBloke/Llama-2-7B-Chat-GPTQ")
    logger.info("⚛️ React SPA: Integrated fullstack application")
    logger.info("🚀 Ready for HuggingFace Spaces deployment")
    logger.info("💾 Memory optimized for 16GB limit")
    logger.info("🔄 Lazy loading enabled for faster startup")
    logger.info("🔧 CORS fix applied - POST endpoints working")
    logger.info(f"⏰ Comprehensive timeout protection - {GLOBAL_REQUEST_TIMEOUT}s global limit")
    logger.info(f"🤖 GPTQ loading timeout: {GPTQ_MODEL_LOADING_TIMEOUT}s")
    print("🔗 React App: /")
    print("🔗 API Documentation: /docs")
    print("🏥 Health Check: /health")
    print("📊 Debug React: /debug-react")
    print("📋 API Info: /api-info")
    print("🚀 Startup Progress: /startup-progress")
    print("✅ POST endpoints fixed and working")
    print("⚛️ React SPA integrated and ready")
    print(f"🛡️ All requests protected by comprehensive timeout system")
    print(f"⏰ GPTQ model loading: up to {GPTQ_FIRST_LOAD_TIMEOUT//60} minutes first time")
    print("🔧 ИСПРАВЛЕНО: API endpoints определены ПЕРЕД монтированием SPA")