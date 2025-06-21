# ====================================
# ФАЙЛ: backend/main.py (ОБНОВЛЕННАЯ ВЕРСИЯ ДЛЯ HUGGINGFACE SPACES)
# Заменить существующий файл полностью
# ====================================

"""
Legal Assistant API - Main Application Entry Point
Обновленная версия с модульной архитектурой и полной интеграцией компонентов для HuggingFace Spaces
"""

import uvicorn
import sys
import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

# Добавляем текущую директорию в Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Настройка логирования должна быть первой
try:
    from utils.logger import setup_logging
    setup_logging(log_level="INFO", log_file="logs/legal_assistant.log")
except ImportError:
    # Fallback если utils.logger недоступен
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    print("⚠️ Using basic logging setup (utils.logger not available)")

import logging
from app import create_app

logger = logging.getLogger(__name__)

def print_startup_banner():
    """Выводит красивый баннер при запуске"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    🏛️  Legal Assistant API v2.0               ║
╠══════════════════════════════════════════════════════════════╣
║  AI-Powered Legal Assistant with Document Processing         ║
║  • HuggingFace LLM Integration                               ║
║  • Document Upload & Processing                              ║
║  • Website Scraping & Content Extraction                    ║
║  • Vector Search with ChromaDB                               ║
║  • Multi-language Support (English/Ukrainian)               ║
║  • RESTful API with FastAPI                                  ║
║  • Deployed on HuggingFace Spaces                           ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_system_info():
    """Выводит информацию о системе"""
    import platform
    import sys
    
    print("📊 System Information:")
    print(f"   Platform: {platform.platform()}")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   Architecture: {platform.architecture()[0]}")
    print(f"   Working Directory: {os.getcwd()}")
    print(f"   Environment: {'HuggingFace Spaces' if os.getenv('SPACE_ID') else 'Local'}")
    print()

def check_dependencies():
    """Проверяет наличие критических зависимостей"""
    print("🔍 Checking Dependencies:")
    
    dependencies = [
        ("fastapi", "FastAPI framework"),
        ("uvicorn", "ASGI server"),
        ("pydantic", "Data validation"),
        ("transformers", "HuggingFace Transformers"),
        ("torch", "PyTorch for LLM"),
        ("aiohttp", "HTTP client (optional)"),
        ("beautifulsoup4", "HTML parsing (optional)"),
        ("chromadb", "Vector database (optional)"),
        ("sentence_transformers", "Text embeddings (optional)")
    ]
    
    missing_deps = []
    
    for dep_name, description in dependencies:
        try:
            __import__(dep_name)
            status = "✅"
        except ImportError:
            status = "❌" if dep_name in ["fastapi", "uvicorn", "pydantic", "transformers"] else "⚠️"
            if dep_name in ["fastapi", "uvicorn", "pydantic", "transformers"]:
                missing_deps.append(dep_name)
        
        print(f"   {status} {dep_name}: {description}")
    
    if missing_deps:
        print(f"\n❌ Critical dependencies missing: {', '.join(missing_deps)}")
        print("   Install with: pip install fastapi uvicorn pydantic transformers")
        return False
    
    print("✅ All critical dependencies available")
    return True

def get_services_status():
    """Получает статус сервисов"""
    print("🔧 Services Status:")
    
    try:
        from app.dependencies import get_services_status
        status = get_services_status()
        
        for service, available in status.items():
            icon = "✅" if available else "❌"
            print(f"   {icon} {service.replace('_', ' ').title()}")
            
        return status
    except Exception as e:
        print(f"   ❌ Could not check services status: {e}")
        return {}

def check_configuration():
    """Проверяет конфигурацию приложения"""
    print("⚙️ Configuration:")
    
    try:
        from app.config import settings
        
        config_items = [
            ("API Version", getattr(settings, 'VERSION', '2.0.0')),
            ("Environment", "HuggingFace Spaces" if os.getenv('SPACE_ID') else "Local"),
            ("Max File Size", f"{getattr(settings, 'MAX_FILE_SIZE', 0) // 1024 // 1024}MB"),
            ("ChromaDB Enabled", getattr(settings, 'USE_CHROMADB', False)),
            ("LLM Demo Mode", getattr(settings, 'LLM_DEMO_MODE', False)),
            ("Log Level", getattr(settings, 'LOG_LEVEL', 'INFO'))
        ]
        
        for name, value in config_items:
            print(f"   • {name}: {value}")
            
    except Exception as e:
        print(f"   ❌ Configuration check failed: {e}")

def create_directories():
    """Создает необходимые директории"""
    directories = [
        "logs",
        "simple_db", 
        "chromadb_data",
        "uploads",
        "temp",
        "backups",
        ".cache"
    ]
    
    created_dirs = []
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            if not os.path.exists(directory):
                created_dirs.append(directory)
        except Exception as e:
            logger.warning(f"Could not create directory {directory}: {e}")
    
    if created_dirs:
        print(f"📁 Created directories: {', '.join(created_dirs)}")

def main():
    """Главная функция запуска приложения для разработки"""
    try:
        # Баннер и информация о системе
        print_startup_banner()
        print_system_info()
        
        # Создаем необходимые директории
        create_directories()
        
        # Проверяем зависимости
        if not check_dependencies():
            print("\n❌ Cannot start application due to missing critical dependencies")
            sys.exit(1)
        
        print()
        
        # Проверяем конфигурацию
        check_configuration()
        print()
        
        # Проверяем сервисы
        services_status = get_services_status()
        print()
        
        # Создаем приложение
        print("🚀 Initializing FastAPI Application...")
        app = create_app_for_deployment()
        
        if app is None:
            print("❌ Failed to create FastAPI application")
            sys.exit(1)
        
        print("✅ FastAPI application created successfully")
        
        # Информация о запуске
        print("\n🌐 Server Information:")
        print("   • Host: 0.0.0.0")
        print("   • Port: 7860 (HuggingFace Spaces)")
        print("   • Docs: http://localhost:7860/docs")
        print("   • ReDoc: http://localhost:7860/redoc")
        print("   • Health: http://localhost:7860/health")
        print("   • API: http://localhost:7860/api/")
        
        # Проверяем режим разработки
        reload_mode = os.getenv("RELOAD", "false").lower() in ["true", "1", "yes"]
        log_level = os.getenv("LOG_LEVEL", "info").lower()
        
        print(f"\n⚙️ Server Configuration:")
        print(f"   • Reload: {reload_mode}")
        print(f"   • Log Level: {log_level}")
        print(f"   • Workers: 1 (HuggingFace Spaces)")
        print(f"   • Environment: {'HF Spaces' if os.getenv('SPACE_ID') else 'Local'}")
        
        print("\n🎯 Ready to serve requests!")
        print("=" * 60)
        
        # Запускаем сервер
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=7860,  # HuggingFace Spaces требует порт 7860
            log_level=log_level,
            reload=reload_mode,
            access_log=True,
            server_header=False,  # Скрываем версию сервера
            date_header=False     # Убираем заголовок Date
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Legal Assistant API shutting down...")
        print("Thank you for using Legal Assistant!")
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}", exc_info=True)
        print(f"\n❌ Fatal error during startup: {e}")
        print("Check logs for detailed error information")
        sys.exit(1)

def create_app_for_deployment():
    """Создает приложение для deployment (без uvicorn.run)"""
    try:
        # Минимальная настройка для production
        try:
            from utils.logger import setup_logging
            setup_logging(log_level="INFO")
        except ImportError:
            logging.basicConfig(level=logging.INFO)
        
        from app import create_app
        app = create_app()
        
        if app is None:
            raise RuntimeError("Failed to create FastAPI application")
        
        # ====================================
        # ИСПРАВЛЕНИЯ ДЛЯ HUGGINGFACE SPACES
        # ====================================
        
        # Явно настраиваем URLs для Swagger UI
        app.docs_url = "/docs"
        app.redoc_url = "/redoc" 
        app.openapi_url = "/openapi.json"
        
        # Настройка CORS для HuggingFace Spaces
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Разрешаем все origins для HF Spaces
            allow_methods=["*"],
            allow_headers=["*"],
            allow_credentials=True,
            expose_headers=["*"]  # Добавлено для лучшей совместимости
        )
        
        logger.info("🚀 Legal Assistant API initialized for deployment")
        logger.info("🌐 CORS and Swagger UI configured for HuggingFace Spaces")
        return app
        
    except Exception as e:
        logger.error(f"Deployment initialization failed: {e}")
        raise

# Создаем экземпляр приложения для WSGI/ASGI серверов
try:
    app = create_app_for_deployment()
except Exception as e:
    print(f"❌ Failed to create app instance: {e}")
    # Создаем заглушку для избежания ImportError
    from fastapi import FastAPI
    app = FastAPI(title="Legal Assistant API - Error", version="2.0.0")
    
    @app.get("/")
    async def error_root():
        return {"error": "Application failed to initialize", "details": str(e)}

# Дополнительные endpoint'ы для мониторинга
@app.get("/health")
async def health_check():
    """Быстрая проверка здоровья"""
    return {
        "status": "healthy", 
        "version": "2.0.0",
        "environment": "HuggingFace Spaces" if os.getenv("SPACE_ID") else "Local",
        "timestamp": __import__("time").time()
    }

@app.get("/version")
async def get_version():
    """Информация о версии"""
    return {
        "version": "2.0.0",
        "name": "Legal Assistant API",
        "description": "AI Legal Assistant with HuggingFace LLM integration",
        "environment": "HuggingFace Spaces" if os.getenv("SPACE_ID") else "Local",
        "features": [
            "HuggingFace Transformers LLM",
            "Document Processing", 
            "Vector Search",
            "Web Scraping",
            "Multi-language Support"
        ]
    }

if __name__ == "__main__":
    main()
else:
    # Если модуль импортируется, выводим краткую информацию
    logger.info("📦 Legal Assistant API module imported")
    logger.info("   Use 'python main.py' to start the development server")
    logger.info("   Or import 'app' for WSGI/ASGI deployment")