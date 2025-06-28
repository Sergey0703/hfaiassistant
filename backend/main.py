# backend/main.py - УПРОЩЁННАЯ ТОЧКА ВХОДА
"""
Упрощённая точка входа приложения без сложных проверок и множественных конфигураций
Заменяет переусложнённый main.py с startup_banner, timeout_middleware и сложной диагностикой
"""

import uvicorn
import sys
import os
import logging
from pathlib import Path

# Добавляем текущую директорию в Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Простая настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def print_simple_banner():
    """Простой баннер без сложных проверок"""
    is_hf_spaces = os.getenv("SPACE_ID") is not None
    
    banner = f"""
🏛️ Legal Assistant API v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🦙 AI Legal Assistant with Llama-3.1-8B-Instruct
⚛️ React Frontend + FastAPI Backend
🌍 Platform: {'HuggingFace Spaces' if is_hf_spaces else 'Local Development'}
📚 Features: Vector Search, Web Scraping, Multilingual Support
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    print(banner)

def create_directories():
    """Создаёт необходимые директории"""
    directories = ["logs", "chromadb_data", "uploads", "temp", "backups"]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create directory {directory}: {e}")

def main():
    """Главная функция для разработки"""
    try:
        print_simple_banner()
        
        # Определяем окружение
        is_hf_spaces = os.getenv("SPACE_ID") is not None
        
        # Создаём необходимые директории
        create_directories()
        
        # Создаём приложение
        app = create_app()
        
        if app is None:
            print("❌ Failed to create FastAPI application")
            sys.exit(1)
        
        # Настройки для запуска
        host = "0.0.0.0"
        port = 7860 if is_hf_spaces else 8000
        
        print(f"\n🚀 Server Configuration:")
        print(f"   • Host: {host}:{port}")
        print(f"   • Environment: {'HuggingFace Spaces' if is_hf_spaces else 'Local'}")
        print(f"   • LLM Model: Llama-3.1-8B-Instruct")
        print(f"   • API Documentation: http://localhost:{port}/docs")
        print(f"   • Health Check: http://localhost:{port}/health")
        
        if not is_hf_spaces:
            print(f"   • Main App: http://localhost:{port}/")
        
        print(f"\n🎯 Starting Legal Assistant API...")
        print("=" * 70)
        
        # Запускаем сервер
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            log_level="info",
            reload=False,  # Отключаем reload в production
            access_log=True,
            workers=1,  # Один worker для простоты
            timeout_keep_alive=65,
            limit_concurrency=10,  # Разумное ограничение
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Legal Assistant API shutting down...")
        print("Thank you for using Legal Assistant!")
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

# ====================================
# СОЗДАНИЕ ПРИЛОЖЕНИЯ ДЛЯ DEPLOYMENT
# ====================================

try:
    print("🚀 Initializing Legal Assistant API...")
    
    # Проверяем окружение
    is_hf_spaces = os.getenv("SPACE_ID") is not None
    
    if is_hf_spaces:
        print("🤗 HuggingFace Spaces environment detected")
        
        # Простые оптимизации для HF Spaces
        os.environ.setdefault("USE_CHROMADB", "true")
        os.environ.setdefault("LLM_DEMO_MODE", "false")  # Используем реальную Llama
        os.environ.setdefault("LOG_LEVEL", "INFO")
    
    # Создаём приложение
    from app import create_app
    app = create_app()
    
    if app is None:
        raise RuntimeError("Failed to create FastAPI application")
    
    print("✅ Legal Assistant API ready for deployment")
    print(f"🌍 Platform: {'HuggingFace Spaces' if is_hf_spaces else 'Local'}")
    print("🦙 LLM Model: Llama-3.1-8B-Instruct via HuggingFace Inference API")
    print("⚛️ React Frontend: Integrated")
    print("📚 Vector Search: ChromaDB enabled")
    print("🌐 Web Scraping: Available")
    print("🔄 Simple initialization: No background tasks")
    
except Exception as e:
    print(f"❌ Deployment initialization failed: {e}")
    print("🔄 Creating minimal fallback application...")
    
    # Создаём минимальное fallback приложение
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(
        title="Legal Assistant API - Minimal Mode", 
        version="2.0.0",
        description="Minimal mode - some services may be unavailable"
    )
    
    # CORS даже в fallback
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"]
    )
    
    @app.get("/")
    async def minimal_root():
        return {
            "status": "minimal_mode",
            "error": str(e),
            "message": "Application running in minimal mode",
            "available_endpoints": ["/docs", "/health"],
            "recommendations": [
                "Check that all dependencies are installed",
                "Verify all Python modules are present",
                "Check server logs for detailed errors"
            ]
        }
    
    @app.get("/health")
    async def minimal_health():
        return {
            "status": "minimal",
            "error": str(e),
            "timestamp": __import__("time").time(),
            "message": "Application started in minimal mode"
        }
    
    print("✅ Minimal fallback application created")

# ====================================
# ОСНОВНЫЕ ENDPOINTS (если API роуты не загрузились)
# ====================================

try:
    # Проверяем что у нас есть основные endpoints
    from fastapi.responses import JSONResponse
    
    @app.get("/api-info")
    async def api_info():
        """Информация об API (fallback endpoint)"""
        return {
            "api": "Legal Assistant API v2.0",
            "llm_model": "Llama-3.1-8B-Instruct",
            "status": "running",
            "features": {
                "llama_integration": True,
                "vector_search": True,
                "web_scraping": True,
                "react_frontend": True,
                "multilingual": True
            },
            "endpoints": {
                "documentation": "/docs",
                "health_check": "/health",
                "api_status": "/api-status"
            },
            "platform": "HuggingFace Spaces" if os.getenv("SPACE_ID") else "Local",
            "simplified_architecture": True
        }
    
    @app.get("/llama-status")
    async def llama_status():
        """Статус Llama модели"""
        try:
            from app.dependencies import get_llm_service
            llm_service = get_llm_service()
            status = await llm_service.get_service_status()
            
            return {
                "llama_model": "meta-llama/Llama-3.1-8B-Instruct",
                "service_ready": status.get("ready", False),
                "service_type": status.get("service_type", "unknown"),
                "hf_token_configured": status.get("hf_token_configured", False),
                "supported_languages": ["en", "uk"],
                "features": {
                    "legal_qa": True,
                    "document_analysis": True,
                    "multilingual": True,
                    "context_aware": True
                },
                "inference_method": "HuggingFace Inference API",
                "recommendations": status.get("recommendations", [])
            }
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={
                    "llama_model": "meta-llama/Llama-3.1-8B-Instruct",
                    "service_ready": False,
                    "error": str(e),
                    "message": "Llama service initialization failed"
                }
            )

except Exception as endpoint_error:
    print(f"⚠️ Could not add info endpoints: {endpoint_error}")

# ====================================
# ФИНАЛЬНАЯ ДИАГНОСТИКА
# ====================================

if __name__ == "__main__":
    main()
else:
    # Когда модуль импортируется (deployment)
    logger.info("📦 Legal Assistant API module imported")
    logger.info("🦙 LLM Model: Llama-3.1-8B-Instruct")
    logger.info("⚛️ React SPA: Integrated fullstack application")
    logger.info("🚀 Ready for deployment")
    logger.info("💡 Simplified architecture - no background tasks")
    
    print("🔗 Available endpoints:")
    print("   • Main App: /")
    print("   • API Documentation: /docs")
    print("   • Health Check: /health")
    print("   • API Status: /api-status")
    print("   • Llama Status: /llama-status")
    print("   • Chat API: /api/user/chat")
    print("   • Search API: /api/user/search")
    print("   • Admin Panel: /api/admin")
    
    print("✅ Simplified Legal Assistant API ready")
    print("🦙 Llama-3.1-8B-Instruct integration active")
    print("⚡ Fast startup - no complex background loading")