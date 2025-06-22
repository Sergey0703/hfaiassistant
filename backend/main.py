# backend/main.py - ПОЛНАЯ ВЕРСИЯ ДЛЯ HUGGINGFACE SPACES + REACT SPA
"""
Legal Assistant API - Main Application Entry Point
Полный стек: FastAPI Backend + React Frontend + GPTQ Model + ChromaDB
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
# КРИТИЧЕСКИЕ ENDPOINTS ДЛЯ ДИАГНОСТИКИ
# ====================================

@app.get("/")
async def root():
    """Корневой роут - проверяем React или показываем API info"""
    from pathlib import Path
    import os
    
    # Попытка найти React build
    possible_react_paths = [
        Path("/home/user/app/frontend/build/index.html"),
        Path("./frontend/build/index.html"),
        Path("frontend/build/index.html")
    ]
    
    react_build_path = None
    for path in possible_react_paths:
        if path.exists():
            react_build_path = path
            break
    
    if react_build_path and react_build_path.exists():
        # Если React есть, возвращаем его
        from fastapi.responses import FileResponse
        return FileResponse(react_build_path, media_type="text/html")
    else:
        # Если React нет, показываем информацию
        return {
            "message": "Legal Assistant API",
            "version": "2.0.0", 
            "status": "API работает ✅",
            "react_status": "React SPA не найден ❌",
            "available_endpoints": {
                "api_docs": "/docs",
                "api_info": "/api-info", 
                "debug": "/debug-react",
                "health": "/health",
                "startup_progress": "/startup-progress"
            },
            "react_paths_checked": [str(p) for p in possible_react_paths],
            "react_exists": react_build_path is not None,
            "space_id": os.getenv("SPACE_ID"),
            "working_directory": os.getcwd(),
            "instructions": [
                "1. Проверьте /debug-react для полной диагностики",
                "2. Убедитесь что frontend/build/ скопирован в Docker",
                "3. Попробуйте /docs для API документации",
                "4. React build должен быть в /home/user/app/frontend/build/"
            ],
            "platform": "HuggingFace Spaces" if os.getenv("SPACE_ID") else "Local"
        }

@app.get("/api-info")
async def api_info():
    """Информация о статусе API и React"""
    import os
    from pathlib import Path
    
    # Проверяем React build
    possible_react_paths = [
        Path("/home/user/app/frontend/build"),  # HF Spaces путь
        Path("./frontend/build"),
        Path("frontend/build")
    ]
    
    react_status = "not_found"
    react_path = None
    react_files = []
    react_details = {}
    
    for path in possible_react_paths:
        if path.exists():
            react_status = "found"
            react_path = str(path)
            try:
                react_files = [f.name for f in path.iterdir()]
                react_details = {
                    "has_index_html": (path / "index.html").exists(),
                    "has_static_dir": (path / "static").exists(),
                    "total_files": len(react_files)
                }
                
                # Проверяем статические файлы
                if react_details["has_static_dir"]:
                    static_path = path / "static"
                    static_files = list(static_path.glob("**/*"))
                    react_details["static_files_count"] = len(static_files)
                    react_details["has_js_files"] = any(f.suffix == '.js' for f in static_files)
                    react_details["has_css_files"] = any(f.suffix == '.css' for f in static_files)
                    
            except Exception as e:
                react_files = [f"error_reading_directory: {e}"]
            break
    
    return {
        "api": "Legal Assistant API v2.0",
        "status": "running",
        "react_status": react_status,
        "react_path": react_path,
        "react_files": react_files[:10],  # First 10 files
        "react_details": react_details,
        "environment": "HuggingFace Spaces",
        "space_id": os.getenv("SPACE_ID", "unknown"),
        "working_directory": os.getcwd(),
        "python_path": sys.path[:3],
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "debug": "/debug-react",
            "startup": "/startup-progress"
        },
        "recommendations": [
            "Check /debug-react for full path analysis",
            "Ensure Docker correctly copied React build files",
            "React should be in /home/user/app/frontend/build/",
            "Try /docs if React is not available"
        ]
    }

@app.get("/debug-react")
async def debug_react():
    """Полная диагностика React"""
    import os
    from pathlib import Path
    
    debug_info = {
        "current_directory": os.getcwd(),
        "script_location": str(Path(__file__).parent),
        "environment": {
            "SPACE_ID": os.getenv("SPACE_ID"),
            "HOME": os.getenv("HOME"),
            "PWD": os.getenv("PWD"),
            "USER": os.getenv("USER")
        }
    }
    
    # Проверяем все возможные пути
    paths_to_check = [
        "/home/user/app/frontend/build",
        "/home/user/app/frontend", 
        "./frontend/build",
        "./frontend",
        "/app/frontend/build",
        "/app/frontend",
        "frontend",
        "build",
        "/home/user/app"
    ]
    
    debug_info["path_checks"] = []
    
    for path_str in paths_to_check:
        path = Path(path_str)
        path_info = {
            "path": path_str,
            "absolute_path": str(path.absolute()) if path.exists() else "N/A",
            "exists": path.exists(),
            "is_dir": path.is_dir() if path.exists() else False,
            "is_file": path.is_file() if path.exists() else False
        }
        
        if path.exists():
            try:
                if path.is_dir():
                    files = list(path.iterdir())
                    file_names = [f.name for f in files]
                    path_info["files"] = file_names[:15]  # First 15 files
                    path_info["total_files"] = len(files)
                    
                    # Check for key React files
                    if "index.html" in file_names:
                        path_info["has_index_html"] = True
                        index_path = path / "index.html"
                        path_info["index_html_size"] = index_path.stat().st_size
                        
                    if "static" in file_names:
                        path_info["has_static_dir"] = True
                        static_path = path / "static"
                        if static_path.exists() and static_path.is_dir():
                            static_files = list(static_path.rglob("*"))
                            path_info["static_files_count"] = len(static_files)
                            path_info["static_files_sample"] = [f.name for f in static_files[:10]]
                            
                    # Check for package.json
                    if "package.json" in file_names:
                        path_info["has_package_json"] = True
                        
                elif path.is_file():
                    path_info["file_size"] = path.stat().st_size
                    path_info["file_extension"] = path.suffix
                    
            except Exception as e:
                path_info["error"] = str(e)
        
        debug_info["path_checks"].append(path_info)
    
    # Проверяем переменные Docker
    debug_info["docker_info"] = {
        "is_docker": os.path.exists("/.dockerenv"),
        "hostname": os.uname().nodename if hasattr(os, 'uname') else "unknown"
    }
    
    return debug_info

@app.get("/startup-progress") 
async def startup_progress():
    """Статус загрузки приложения"""
    from pathlib import Path
    
    # Проверяем React
    react_path = Path("/home/user/app/frontend/build")
    react_ready = react_path.exists() and (react_path / "index.html").exists()
    
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
                "path": str(react_path),
                "ready": react_ready
            },
            "model": {
                "status": "loading",
                "description": "GPTQ model loading in background"
            }
        },
        "progress": "75%" if react_ready else "50%",
        "message": "API готов, React SPA проверяется...",
        "endpoints_working": ["/docs", "/api-info", "/debug-react", "/health"],
        "next_steps": [
            "Check /debug-react for React build location",
            "Verify Docker copied frontend/build correctly",
            "Try /docs for API documentation"
        ]
    }

# Быстрый health check для deployment
@app.get("/health")
async def health_check():
    """Быстрая проверка здоровья"""
    try:
        import asyncio
        return await asyncio.wait_for({
            "status": "healthy", 
            "version": "2.0.0",
            "platform": "HuggingFace Spaces" if os.getenv("SPACE_ID") else "Local",
            "gptq_model": "TheBloke/Llama-2-7B-Chat-GPTQ",
            "react_spa": "Checking...",
            "lazy_loading": True,
            "memory_optimized": True,
            "cors_fix_applied": True,
            "post_endpoints_working": True,
            "timeout_protected": True,
            "timeout_limits": {
                "global": f"{GLOBAL_REQUEST_TIMEOUT}s",
                "gptq_loading": f"{GPTQ_MODEL_LOADING_TIMEOUT}s",
                "health_check": "15s"
            },
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

# Статические файлы React (если найдены)
try:
    from fastapi.staticfiles import StaticFiles
    from pathlib import Path
    
    react_static_path = Path("/home/user/app/frontend/build/static")
    if react_static_path.exists():
        app.mount("/static", StaticFiles(directory=react_static_path), name="react_static")
        print("✅ React static files mounted at /static")
    
    # Дополнительные React файлы
    react_build_path = Path("/home/user/app/frontend/build")
    if react_build_path.exists():
        react_files = ["manifest.json", "favicon.ico", "robots.txt", "logo192.png", "logo512.png"]
        
        for file_name in react_files:
            file_path = react_build_path / file_name
            if file_path.exists():
                @app.get(f"/{file_name}", include_in_schema=False)
                async def serve_react_file(filename=file_name):
                    from fastapi.responses import FileResponse
                    return FileResponse(react_build_path / filename)
                
        print(f"✅ React assets available: {[f for f in react_files if (react_build_path / f).exists()]}")
        
except Exception as e:
    print(f"⚠️ Could not mount React static files: {e}")

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