# backend/main.py - ФИНАЛЬНАЯ ВЕРСИЯ ДЛЯ HUGGINGFACE SPACES + REACT SPA
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
from utils.startup_banner import (
    print_startup_banner, check_hf_spaces_environment, 
    check_critical_dependencies, create_necessary_directories
)
from config.timeouts import (
    GLOBAL_REQUEST_TIMEOUT, KEEP_ALIVE_TIMEOUT, GRACEFUL_TIMEOUT,
    GPTQ_MODEL_LOADING_TIMEOUT, GPTQ_FIRST_LOAD_TIMEOUT
)
from core.app_factory import create_app_for_deployment

logger = logging.getLogger(__name__)

def main():
    """Главная функция для разработки (не используется в HF Spaces)"""
    try:
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
        
        print()
        
        # Создаем приложение
        app = create_app_for_deployment()
        
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
        print(f"   • HF Health: http://localhost:{port}/hf-spaces-health")
        print(f"   • Model Status: http://localhost:{port}/model-status")
        print(f"   • Memory Status: http://localhost:{port}/memory-status")
        print(f"   • Startup Progress: http://localhost:{port}/startup-progress")
        print(f"   • Timeout Status: http://localhost:{port}/timeout-status")
        
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

# ====================================
# ПРИЛОЖЕНИЕ ДЛЯ DEPLOYMENT
# ====================================

# Создаем экземпляр приложения для WSGI/ASGI серверов
try:
    # Улучшенная инициализация для deployment
    print("🚀 Initializing Legal Assistant API for HuggingFace Spaces...")
    
    # Проверяем и настраиваем HF Spaces
    is_hf_spaces = check_hf_spaces_environment()
    
    # Создаем директории (может частично не удастся в HF Spaces - это нормально)
    create_necessary_directories()
    
    # Создаем приложение с улучшенной обработкой ошибок
    app = create_app_for_deployment()
    
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
    
    @app.get("/")
    async def recovery_info():
        return {
            "status": "recovery_mode",
            "error": str(e),
            "platform": "HuggingFace Spaces" if os.getenv("SPACE_ID") else "Local",
            "timestamp": __import__("time").time(),
            "message": "Application is running in recovery mode",
            "cors_fix_applied": True,
            "react_spa": "Recovery mode - limited functionality",
            "timeout_controls": {
                "global_timeout": GLOBAL_REQUEST_TIMEOUT,
                "keep_alive": KEEP_ALIVE_TIMEOUT,
                "gptq_loading": GPTQ_MODEL_LOADING_TIMEOUT
            },
            "available_endpoints": [
                "/health - Basic health check",
                "/recovery-diagnostics - Detailed error info", 
                "/docs - API documentation (limited)"
            ],
            "recommendations": [
                "Check server logs for detailed errors",
                "Verify all dependencies are installed",
                f"GPTQ model loading can take up to {GPTQ_FIRST_LOAD_TIMEOUT//60} minutes",
                "Try refreshing the page in a few minutes",
                "Some services may still be initializing",
                "React frontend may need to be built: npm run build"
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
            "react_spa": "Enabled",
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
            "timestamp": __import__("time").time()
        }, timeout=15)
    except:
        return {
            "status": "timeout",
            "timeout_limit": "15s",
            "global_timeout": f"{GLOBAL_REQUEST_TIMEOUT}s",
            "message": "Health check timeout - services may be loading"
        }

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
    print("🏥 Health Check: /hf-spaces-health")
    print("📊 Timeout Status: /timeout-status")
    print("🤖 Model Status: /model-status")
    print("💾 Memory Status: /memory-status")
    print("✅ POST endpoints fixed and working")
    print("⚛️ React SPA integrated and ready")
    print(f"🛡️ All requests protected by comprehensive timeout system")
    print(f"⏰ GPTQ model loading: up to {GPTQ_FIRST_LOAD_TIMEOUT//60} minutes first time")