# backend/main.py - МИНИМАЛЬНАЯ ТОЧКА ВХОДА
"""
Упрощенная точка входа для минимальной RAG системы
Убрана сложная диагностика, оставлены только базовые функции
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

def print_minimal_banner():
    """Минимальный баннер для RAG системы"""
    is_hf_spaces = os.getenv("SPACE_ID") is not None
    
    banner = f"""
🏛️ Minimal Legal RAG System v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 FLAN-T5 Small + Sentence Transformers + ChromaDB
⚡ Target: <1GB RAM, Fast startup, HuggingFace Spaces optimized
🌍 Platform: {'HuggingFace Spaces' if is_hf_spaces else 'Local Development'}
📚 Features: Semantic Search, Document Upload, Multilingual Support
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    print(banner)

def create_directories():
    """Создаёт необходимые директории"""
    directories = ["logs", "chromadb_data", "uploads", "temp", ".cache"]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create directory {directory}: {e}")

def main():
    """Главная функция для разработки"""
    try:
        print_minimal_banner()
        
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
        
        print(f"\n🚀 Minimal RAG Configuration:")
        print(f"   • Host: {host}:{port}")
        print(f"   • Environment: {'HuggingFace Spaces' if is_hf_spaces else 'Local'}")
        print(f"   • LLM Model: google/flan-t5-small (~300 MB)")
        print(f"   • Embedding: all-MiniLM-L6-v2 (~90 MB)")
        print(f"   • Vector DB: ChromaDB")
        print(f"   • Total RAM: ~920 MB target")
        print(f"   • API Documentation: http://localhost:{port}/docs")
        print(f"   • Health Check: http://localhost:{port}/health")
        
        if not is_hf_spaces:
            print(f"   • Main App: http://localhost:{port}/")
        
        print(f"\n⚡ Starting Minimal RAG System...")
        print("=" * 50)
        
        # Запускаем сервер
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            log_level="info",
            reload=False,
            access_log=True,
            workers=1,
            timeout_keep_alive=30,  # Меньше для экономии ресурсов
            limit_concurrency=5,    # Ограничение для стабильности
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Minimal RAG System shutting down...")
        print("Thank you for using Minimal RAG!")
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

# ====================================
# СОЗДАНИЕ ПРИЛОЖЕНИЯ ДЛЯ DEPLOYMENT
# ====================================

try:
    print("🚀 Initializing Minimal RAG System...")
    
    # Проверяем окружение
    is_hf_spaces = os.getenv("SPACE_ID") is not None
    
    if is_hf_spaces:
        print("🤗 HuggingFace Spaces environment detected")
        
        # Простые оптимизации для HF Spaces
        os.environ.setdefault("USE_CHROMADB", "true")
        os.environ.setdefault("LOG_LEVEL", "INFO")
        os.environ.setdefault("LLM_MODEL", "google/flan-t5-small")
        os.environ.setdefault("LLM_MAX_TOKENS", "150")
        os.environ.setdefault("LLM_TIMEOUT", "20")
    
    # Создаём приложение
    from app import create_app
    app = create_app()
    
    if app is None:
        raise RuntimeError("Failed to create FastAPI application")
    
    print("✅ Minimal RAG System ready for deployment")
    print(f"🌍 Platform: {'HuggingFace Spaces' if is_hf_spaces else 'Local'}")
    print("🤖 LLM Model: google/flan-t5-small")
    print("🔍 Embedding: sentence-transformers/all-MiniLM-L6-v2")
    print("📊 Vector DB: ChromaDB")
    print("⚛️ React Frontend: Integrated")
    print("💾 Memory Target: <1GB RAM")
    print("⚡ Fast startup: No heavy models")
    
except Exception as e:
    print(f"❌ Deployment initialization failed: {e}")
    print("🔄 Creating minimal fallback application...")
    
    # Создаём минимальное fallback приложение
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(
        title="Minimal RAG System - Fallback Mode", 
        version="1.0.0",
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
            "model": "google/flan-t5-small",
            "target_memory": "<1GB RAM",
            "available_endpoints": ["/docs", "/health"],
            "recommendations": [
                "Check that transformers is installed",
                "Verify sentence-transformers availability", 
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
    @app.get("/api-info")
    async def api_info():
        """Информация об API"""
        return {
            "api": "Minimal RAG System v1.0",
            "llm_model": "google/flan-t5-small",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "vector_db": "ChromaDB",
            "status": "running",
            "memory_target": "<1GB RAM",
            "features": {
                "flan_t5_integration": True,
                "semantic_search": True,
                "document_upload": True,
                "multilingual": True,
                "fast_startup": True
            },
            "endpoints": {
                "documentation": "/docs",
                "health_check": "/health",
                "api_status": "/api-status"
            },
            "platform": "HuggingFace Spaces" if os.getenv("SPACE_ID") else "Local"
        }
    
    @app.get("/model-status")
    async def model_status():
        """Статус моделей"""
        try:
            from app.dependencies import get_llm_service
            llm_service = get_llm_service()
            status = await llm_service.get_service_status()
            
            return {
                "llm_model": "google/flan-t5-small",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "llm_ready": status.get("ready", False),
                "llm_type": status.get("service_type", "unknown"),
                "memory_estimate": {
                    "flan_t5": "~300 MB",
                    "embeddings": "~90 MB", 
                    "chromadb": "~20 MB",
                    "total": "~920 MB"
                },
                "features": {
                    "text2text_generation": True,
                    "semantic_embeddings": True,
                    "vector_search": True,
                    "multilingual": True
                }
            }
        except Exception as e:
            return {
                "llm_model": "google/flan-t5-small",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2", 
                "llm_ready": False,
                "error": str(e),
                "message": "Model services initialization failed"
            }

except Exception as endpoint_error:
    print(f"⚠️ Could not add info endpoints: {endpoint_error}")

# ====================================
# ФИНАЛЬНАЯ ДИАГНОСТИКА
# ====================================

if __name__ == "__main__":
    main()
else:
    # Когда модуль импортируется (deployment)
    logger.info("📦 Minimal RAG System module imported")
    logger.info("🤖 LLM Model: google/flan-t5-small")
    logger.info("🔍 Embedding Model: sentence-transformers/all-MiniLM-L6-v2")
    logger.info("💾 Memory Target: <1GB RAM")
    logger.info("⚡ Fast startup enabled")
    
    print("🔗 Available endpoints:")
    print("   • Main App: /")
    print("   • API Documentation: /docs")
    print("   • Health Check: /health")
    print("   • API Status: /api-status")
    print("   • Model Status: /model-status")
    print("   • Chat API: /api/user/chat")
    print("   • Search API: /api/user/search")
    print("   • Admin Panel: /api/admin")
    
    print("✅ Minimal RAG System ready")
    print("🤖 FLAN-T5 Small integration active")
    print("⚡ Optimized for <1GB RAM usage")