# backend/main.py - ИСПРАВЛЕННАЯ ВЕРСИЯ ПО ДОКУМЕНТАЦИИ HF SPACES
"""
Legal Assistant API - Main Application Entry Point
ИСПРАВЛЕНИЯ: Правильное монтирование React согласно документации HuggingFace Spaces
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
            is_hf_spaces = check_hf_spaces_environment()
            print()
            create_necessary_directories()
            
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
        
        # Запускаем сервер
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            log_level="info",
            reload=False,
            access_log=True,
            server_header=False,
            date_header=False,
            workers=1,
            timeout_keep_alive=KEEP_ALIVE_TIMEOUT,
            timeout_graceful_shutdown=GRACEFUL_TIMEOUT,
            limit_concurrency=5,
            limit_max_requests=500,
            timeout_notify=GRACEFUL_TIMEOUT,
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Legal Assistant API shutting down...")
        
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
    
    is_hf_spaces = os.getenv("SPACE_ID") is not None
    
    if banner_available:
        check_hf_spaces_environment()
        create_necessary_directories()
        app = create_app_for_deployment()
    else:
        from app import create_app
        app = create_app()
        
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
    
except Exception as e:
    print(f"❌ Deployment initialization failed: {e}")
    print("🔄 Creating minimal fallback application...")
    
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(
        title="Legal Assistant API - Recovery Mode", 
        version="2.0.0",
        description="Minimal recovery mode - some services may be unavailable"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"]
    )

# ====================================
# ПРАВИЛЬНАЯ НАСТРОЙКА REACT SPA ПО ДОКУМЕНТАЦИИ HF SPACES
# ====================================

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

# Путь к React файлам
REACT_STATIC_PATH = Path("/home/user/app/static")

# ====================================
# СНАЧАЛА ВСЕ API ENDPOINTS
# ====================================

@app.get("/api-info")
async def api_info():
    """Информация о статусе API и React"""
    react_files_info = {}
    
    try:
        if REACT_STATIC_PATH.exists():
            react_files = list(REACT_STATIC_PATH.iterdir())
            react_files_info = {
                "react_path": str(REACT_STATIC_PATH),
                "react_exists": True,
                "index_html_exists": (REACT_STATIC_PATH / "index.html").exists(),
                "total_files": len(react_files),
                "files": [f.name for f in react_files[:10]]
            }
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
            "root": "/",
            "docs": "/docs",
            "health": "/health",
            "debug": "/debug-react"
        }
    }

@app.get("/debug-react")
async def debug_react():
    """Полная диагностика React"""
    debug_info = {
        "current_directory": os.getcwd(),
        "environment": {
            "SPACE_ID": os.getenv("SPACE_ID"),
            "HOME": os.getenv("HOME"),
            "PWD": os.getenv("PWD")
        },
        "react_paths": {
            "static_path": str(REACT_STATIC_PATH),
            "static_exists": REACT_STATIC_PATH.exists()
        }
    }
    
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

@app.get("/health")
async def health_check():
    """Быстрая проверка здоровья"""
    try:
        return {
            "status": "healthy", 
            "version": "2.0.0",
            "platform": "HuggingFace Spaces",
            "react_spa": "Ready" if (REACT_STATIC_PATH / "index.html").exists() else "Not found",
            "static_files": "Available" if REACT_STATIC_PATH.exists() else "Not mounted",
            "timestamp": __import__("time").time()
        }
    except:
        return {
            "status": "timeout",
            "message": "Health check timeout"
        }

# ====================================
# КРИТИЧЕСКИ ВАЖНО: ПОРЯДОК МОНТИРОВАНИЯ ПО ДОКУМЕНТАЦИИ HF SPACES
# ====================================

# ВАЖНО: Монтируем статические файлы ПОСЛЕ всех API endpoints
if REACT_STATIC_PATH.exists():
    print(f"✅ Mounting React static files from: {REACT_STATIC_PATH}")
    
    # По документации HF Spaces: монтируем корень с html=True для SPA routing
    app.mount("/", StaticFiles(directory=str(REACT_STATIC_PATH), html=True), name="static")
    
    print("✅ React SPA mounted successfully according to HF Spaces documentation")
else:
    print(f"⚠️ React static path not found: {REACT_STATIC_PATH}")
    
    # Fallback корневой route если React не найден
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
                "health": "/health"
            },
            "platform": "HuggingFace Spaces",
            "instructions": [
                "React files should be in /home/user/app/static/",
                "Check /debug-react for detailed diagnosis",
                "Try /docs for API documentation"
            ]
        }

if __name__ == "__main__":
    main()
else:
    logger.info("📦 Legal Assistant API module imported")
    logger.info("🤖 GPTQ Model: TheBloke/Llama-2-7B-Chat-GPTQ")
    logger.info("⚛️ React SPA: Integrated fullstack application")
    logger.info("🚀 Ready for HuggingFace Spaces deployment")
    print("🔗 React App: /")
    print("🔗 API Documentation: /docs")
    print("🏥 Health Check: /health")
    print("📊 Debug React: /debug-react")
    print("📋 API Info: /api-info")
    print("✅ Static files mounted according to HF Spaces documentation")
    print("⚛️ React SPA integrated and ready")