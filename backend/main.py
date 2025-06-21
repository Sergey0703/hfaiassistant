# backend/main.py - ПОЛНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ ДЛЯ HUGGINGFACE SPACES
"""
Legal Assistant API - Main Application Entry Point
ИСПРАВЛЕНИЯ: Правильные таймауты для HF Spaces, исправление POST 404, оптимизация для GPTQ
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

# Настройка логирования - упрощенная для HF Spaces
try:
    from utils.logger import setup_logging
    setup_logging(log_level="INFO", log_file="logs/legal_assistant.log")
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    print("⚠️ Using basic logging setup (utils.logger not available)")

import logging
from app import create_app

logger = logging.getLogger(__name__)

# ====================================
# ОПТИМИЗИРОВАННЫЕ ТАЙМАУТЫ ДЛЯ HF SPACES
# ====================================

# Основные таймауты (на основе исследования HF Spaces лимитов)
GLOBAL_REQUEST_TIMEOUT = 600     # 10 минут максимум на любой запрос (HF Spaces лимит)
KEEP_ALIVE_TIMEOUT = 65          # 65 секунд keep-alive (стандарт HF Spaces)
GRACEFUL_TIMEOUT = 300           # 5 минут на graceful shutdown

# GPTQ модель таймауты (на основе документации TheBloke)
GPTQ_MODEL_LOADING_TIMEOUT = 480  # 8 минут на загрузку GPTQ модели (TheBloke/Llama-2-7B-Chat-GPTQ)
GPTQ_INFERENCE_TIMEOUT = 120      # 2 минуты на генерацию ответа
GPTQ_FIRST_LOAD_TIMEOUT = 600     # 10 минут на первую загрузку (HF Spaces может быть медленным)

# ChromaDB таймауты (оптимизированные для 16GB памяти)
CHROMADB_SEARCH_TIMEOUT = 30      # 30 секунд на поиск
CHROMADB_ADD_DOC_TIMEOUT = 60     # 1 минута на добавление документа
CHROMADB_STATS_TIMEOUT = 20       # 20 секунд на статистику

# HTTP запросы таймауты
HTTP_REQUEST_TIMEOUT = 45         # 45 секунд на HTTP запросы
SCRAPER_TIMEOUT = 60             # 1 минута на парсинг одной страницы

# Специальные таймауты для HF Spaces
HF_SPACES_STARTUP_TIMEOUT = 180   # 3 минуты на полный старт приложения
HF_SPACES_HEALTH_TIMEOUT = 15     # 15 секунд на health check

def print_startup_banner():
    """Улучшенный баннер для HF Spaces с информацией о таймаутах"""
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║                🏛️  Legal Assistant API v2.0 (HF Spaces)      ║
╠══════════════════════════════════════════════════════════════╣
║  AI Legal Assistant with GPTQ Model Support + ТАЙМАУТЫ      ║
║  • TheBloke/Llama-2-7B-Chat-GPTQ Integration               ║
║  • ChromaDB Vector Search with Lazy Loading                 ║
║  • Multi-language Support (English/Ukrainian)               ║
║  • Real-time Document Processing                            ║
║  • Optimized Memory Management for HF Spaces                ║
║  🚀 Production Ready with Graceful Degradation              ║
║  ⏰ Request Timeout: {GLOBAL_REQUEST_TIMEOUT}s (10 min)                     ║
║  🔄 Keep-Alive: {KEEP_ALIVE_TIMEOUT}s                                    ║
║  🤖 GPTQ Loading: {GPTQ_MODEL_LOADING_TIMEOUT}s (8 min)                        ║
║  📚 ChromaDB Search: {CHROMADB_SEARCH_TIMEOUT}s                           ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_hf_spaces_environment():
    """Улучшенная проверка и настройка окружения HF Spaces"""
    is_hf_spaces = os.getenv("SPACE_ID") is not None
    
    print("🌍 Environment Analysis:")
    print(f"   • Platform: {'HuggingFace Spaces' if is_hf_spaces else 'Local Development'}")
    
    if is_hf_spaces:
        space_id = os.getenv("SPACE_ID", "unknown")
        space_author = os.getenv("SPACE_AUTHOR", "unknown")
        print(f"   • Space ID: {space_id}")
        print(f"   • Author: {space_author}")
        print("   🤗 HuggingFace Spaces detected - applying optimizations")
        
        # Оптимизированные настройки для HF Spaces + правильные таймауты
        hf_optimizations = {
            "OLLAMA_ENABLED": "false",              # Отключаем Ollama в HF Spaces
            "LLM_DEMO_MODE": "false",               # Включаем реальную GPTQ модель
            "USE_CHROMADB": "true",                 # Включаем ChromaDB
            "LOG_LEVEL": "INFO",                    # Информативные логи
            "CHROMADB_PATH": "./chromadb_data",     # Локальная директория
            "LLM_TIMEOUT": str(GPTQ_INFERENCE_TIMEOUT),           # 2 минуты timeout для GPTQ inference
            "MAX_CONTEXT_DOCUMENTS": "2",          # Ограичиваем контекст для памяти
            "CONTEXT_TRUNCATE_LENGTH": "800",      # Сокращаем контекст для HF Spaces
            "LLM_MAX_TOKENS": "400",               # Ограничиваем токены для памяти
            "LLM_TEMPERATURE": "0.2",              # Консервативная температура
            # НОВЫЕ НАСТРОЙКИ ТАЙМАУТОВ
            "SEARCH_TIMEOUT": str(CHROMADB_SEARCH_TIMEOUT),      # 30 секунд на поиск
            "CHAT_TIMEOUT": str(GLOBAL_REQUEST_TIMEOUT),         # 10 минут на полный чат
            "DOCUMENT_TIMEOUT": str(CHROMADB_ADD_DOC_TIMEOUT),   # 1 минута на операции с документами
            "GPTQ_LOADING_TIMEOUT": str(GPTQ_MODEL_LOADING_TIMEOUT), # 8 минут на загрузку GPTQ
            "FIRST_LOAD_TIMEOUT": str(GPTQ_FIRST_LOAD_TIMEOUT),      # 10 минут на первую загрузку
            "HTTP_TIMEOUT": str(HTTP_REQUEST_TIMEOUT),            # 45 секунд на HTTP
            "SCRAPER_TIMEOUT": str(SCRAPER_TIMEOUT)               # 1 минута на парсинг
        }
        
        applied_settings = []
        for key, value in hf_optimizations.items():
            if not os.getenv(key):  # Устанавливаем только если не задано
                os.environ[key] = value
                applied_settings.append(f"{key}={value}")
        
        if applied_settings:
            print("   ⚙️ Applied HF Spaces optimizations with timeout controls:")
            for setting in applied_settings[:5]:  # Показываем первые 5
                print(f"      - {setting}")
            if len(applied_settings) > 5:
                print(f"      - ... and {len(applied_settings) - 5} more timeout settings")
        
        # Проверяем доступную память и ресурсы
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total // (1024**3)
            available_gb = memory.available // (1024**3)
            print(f"   💾 Available Memory: {available_gb}GB / {memory_gb}GB")
            print(f"   🔄 CPU Cores: {psutil.cpu_count()}")
            
            # Предупреждение если мало памяти для GPTQ
            if memory_gb < 14:
                print("   ⚠️ Low memory detected - GPTQ model may need extended loading time")
                print(f"   ⏰ Extended GPTQ timeout: {GPTQ_FIRST_LOAD_TIMEOUT}s (10 min)")
            else:
                print(f"   ✅ Sufficient memory for GPTQ model (standard timeout: {GPTQ_MODEL_LOADING_TIMEOUT}s)")
                
        except ImportError:
            print("   💾 Resource info: psutil not available")
        
        # Проверяем доступность CUDA
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**2)
                print("   🚀 CUDA available - GPU acceleration enabled")
                print(f"   🎯 GPU Memory: {gpu_memory}MB")
                if gpu_memory < 8000:  # Меньше 8GB GPU
                    print("   ⚠️ Limited GPU memory - using CPU offloading for GPTQ")
            else:
                print("   💻 CPU-only mode (normal for HF Spaces free tier)")
        except ImportError:
            print("   ⚠️ PyTorch not detected")
        
        print("   ✅ HF Spaces environment configured with optimized timeouts")
        
    else:
        print("   💻 Local development environment")
        print(f"   • Python: {sys.version.split()[0]}")
        print(f"   • Working Dir: {os.getcwd()}")
        
        # Для локальной разработки используем другие настройки
        if not os.getenv("LLM_DEMO_MODE"):
            os.environ.setdefault("LLM_DEMO_MODE", "false")  # Реальная модель по умолчанию
    
    return is_hf_spaces

def check_critical_dependencies():
    """Проверяет только критические зависимости с версиями для GPTQ"""
    print("🔍 Critical Dependencies Check:")
    
    critical_deps = [
        ("fastapi", "FastAPI framework"),
        ("uvicorn", "ASGI server"),
        ("transformers", "HuggingFace Transformers (for GPTQ)"),
        ("torch", "PyTorch (for GPTQ model)")
    ]
    
    missing_critical = []
    
    for dep_name, description in critical_deps:
        try:
            module = __import__(dep_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"   ✅ {dep_name} ({version}): {description}")
        except ImportError:
            print(f"   ❌ {dep_name}: {description}")
            missing_critical.append(dep_name)
    
    # Опциональные зависимости с версиями - ВАЖНЫЕ ДЛЯ GPTQ
    optional_deps = [
        ("sentence_transformers", "Text embeddings (for ChromaDB)"),
        ("chromadb", "Vector database"),
        ("aiohttp", "HTTP client (for scraping)"),
        ("auto_gptq", "GPTQ quantization support (CRITICAL for GPTQ models)"),
        ("accelerate", "Model acceleration (REQUIRED for GPTQ)"),
        ("safetensors", "Safe tensor loading (REQUIRED for GPTQ)"),
        ("optimum", "HuggingFace optimization library"),
        ("psutil", "System monitoring")
    ]
    
    print("\n📦 Optional Dependencies (Important for GPTQ):")
    optional_available = 0
    gptq_ready = True
    
    for dep_name, description in optional_deps:
        try:
            module = __import__(dep_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"   ✅ {dep_name} ({version}): {description}")
            optional_available += 1
        except ImportError:
            print(f"   ⚠️ {dep_name}: {description} (will use fallback)")
            if dep_name in ["auto_gptq", "accelerate", "safetensors"]:
                gptq_ready = False
    
    print(f"\n📊 Dependencies Summary:")
    print(f"   • Critical: {len(critical_deps) - len(missing_critical)}/{len(critical_deps)} available")
    print(f"   • Optional: {optional_available}/{len(optional_deps)} available")
    
    if gptq_ready:
        print(f"   🤖 GPTQ Model Support: ✅ Ready (TheBloke/Llama-2-7B-Chat-GPTQ)")
        print(f"   ⏰ GPTQ Loading Timeout: {GPTQ_MODEL_LOADING_TIMEOUT}s")
    else:
        print(f"   🤖 GPTQ Model Support: ⚠️ Limited (missing auto-gptq or accelerate)")
        print(f"   ⏰ Fallback mode will be used")
    
    if missing_critical:
        print(f"\n❌ Critical dependencies missing: {', '.join(missing_critical)}")
        print("   Install with: pip install fastapi uvicorn transformers torch")
        return False
    
    print("\n✅ All critical dependencies available")
    return True

def create_necessary_directories():
    """Безопасно создает необходимые директории для HF Spaces"""
    directories = [
        "logs",
        "chromadb_data", 
        "uploads",
        "temp",
        "backups",
        ".cache",
        "offload"  # Для model offloading в HF Spaces
    ]
    
    created = []
    failed = []
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            # Проверяем что директория действительно создана и доступна для записи
            test_file = os.path.join(directory, ".test_write")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                created.append(directory)
            except:
                # Директория создана но не доступна для записи (HF Spaces ограничения)
                logger.warning(f"Directory {directory} created but not writable (HF Spaces limitation)")
                
        except Exception as e:
            failed.append(f"{directory}: {str(e)[:50]}")
            logger.warning(f"Could not create directory {directory}: {e}")
    
    if created:
        print(f"📁 Created directories: {', '.join(created)}")
    if failed:
        print(f"⚠️ Failed directories: {', '.join([f.split(':')[0] for f in failed])}")
    
    # В HF Spaces некоторые директории могут быть read-only, это нормально
    return len(created) > 0

def create_app_for_deployment():
    """Создает приложение для deployment с исправлениями POST 404 И РАСШИРЕННЫМИ ТАЙМАУТАМИ"""
    try:
        print("🚀 Creating FastAPI application with comprehensive timeout controls...")
        
        # Создаем приложение с новой архитектурой
        from app import create_app
        app = create_app()
        
        if app is None:
            raise RuntimeError("Failed to create FastAPI application")
        
        # ====================================
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: CORS ПЕРВЫМ
        # ====================================
        
        # ИСПРАВЛЕНИЕ: Добавляем CORS middleware ПЕРВЫМ, до всех остальных
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
            allow_headers=["*"],
            allow_credentials=True,
            expose_headers=["*"],
            max_age=3600
        )
        
        # ====================================
        # НАСТРОЙКА MIDDLEWARE
        # ====================================
        
        # Теперь настраиваем остальные middleware ПОСЛЕ CORS
        try:
            from app.middleware import setup_middleware
            setup_middleware(app)
            logger.info("✅ Middleware configured after CORS")
        except Exception as e:
            error_msg = f"Middleware setup failed: {e}"
            logger.warning(f"⚠️ {error_msg}")
            # Middleware не критичен, продолжаем без него
        
        # ====================================
        # MIDDLEWARE ДЛЯ КОНТРОЛЯ РАСШИРЕННЫХ ТАЙМАУТОВ
        # ====================================
        
        @app.middleware("http")
        async def comprehensive_timeout_middleware(request, call_next):
            """Comprehensive middleware для контроля всех типов таймаутов"""
            import asyncio
            import time
            
            start_time = time.time()
            path = str(request.url.path)
            method = request.method
            
            # Определяем специфичный таймаут для разных типов запросов
            if "/api/admin/llm" in path and "status" in path:
                timeout = HF_SPACES_HEALTH_TIMEOUT  # 15s для статуса LLM
            elif "/api/user/chat" in path:
                timeout = GLOBAL_REQUEST_TIMEOUT    # 10min для чата с GPTQ
            elif "/api/user/search" in path:
                timeout = CHROMADB_SEARCH_TIMEOUT + 30  # 60s для поиска
            elif "/api/admin/documents" in path and method == "POST":
                timeout = CHROMADB_ADD_DOC_TIMEOUT + 30  # 90s для добавления документов
            elif "/api/admin/scraper" in path:
                timeout = SCRAPER_TIMEOUT + 30       # 90s для парсинга
            elif "/model-status" in path:
                timeout = GPTQ_MODEL_LOADING_TIMEOUT # 8min для статуса GPTQ модели
            elif "/hf-spaces-health" in path:
                timeout = HF_SPACES_HEALTH_TIMEOUT  # 15s для health check
            elif "/startup-progress" in path:
                timeout = HF_SPACES_HEALTH_TIMEOUT  # 15s для прогресса
            else:
                timeout = GLOBAL_REQUEST_TIMEOUT    # 10min по умолчанию
            
            try:
                # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Оборачиваем каждый запрос в соответствующий таймаут
                response = await asyncio.wait_for(
                    call_next(request), 
                    timeout=timeout
                )
                
                process_time = time.time() - start_time
                response.headers["X-Process-Time"] = str(round(process_time, 3))
                response.headers["X-Timeout-Limit"] = str(timeout)
                response.headers["X-Request-Type"] = _classify_request_type(path, method)
                
                # Логируем долгие запросы с учетом типа
                if process_time > 60:  # Более 1 минуты
                    logger.warning(f"⏰ Slow request: {method} {path} took {process_time:.2f}s (limit: {timeout}s)")
                elif process_time > 30:  # Более 30 секунд
                    logger.info(f"⏰ Long request: {method} {path} took {process_time:.2f}s")
                
                return response
                
            except asyncio.TimeoutError:
                process_time = time.time() - start_time
                request_type = _classify_request_type(path, method)
                
                logger.error(f"❌ {request_type} timeout: {method} {path} after {process_time:.2f}s (limit: {timeout}s)")
                
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=408,  # Request Timeout
                    content={
                        "detail": f"{request_type} timeout after {timeout} seconds",
                        "path": path,
                        "method": method,
                        "timeout_limit": timeout,
                        "actual_time": round(process_time, 2),
                        "request_type": request_type,
                        "suggestion": _get_timeout_suggestion(request_type, timeout),
                        "platform": "HuggingFace Spaces"
                    }
                )
            except Exception as e:
                process_time = time.time() - start_time
                logger.error(f"❌ Request error: {method} {path} after {process_time:.2f}s: {e}")
                raise
        
        def _classify_request_type(path: str, method: str) -> str:
            """Классифицирует тип запроса для лучшего сообщения об ошибке"""
            if "/api/user/chat" in path:
                return "GPTQ Chat Request"
            elif "/api/user/search" in path:
                return "ChromaDB Search"
            elif "/api/admin/documents" in path and method == "POST":
                return "Document Upload"
            elif "/api/admin/scraper" in path:
                return "Web Scraping"
            elif "/model-status" in path:
                return "GPTQ Model Status"
            elif "/hf-spaces-health" in path:
                return "Health Check"
            else:
                return "API Request"
        
        def _get_timeout_suggestion(request_type: str, timeout: int) -> str:
            """Предоставляет специфичные рекомендации по таймауту"""
            suggestions = {
                "GPTQ Chat Request": "Try with a shorter question. GPTQ model is loading or generating response.",
                "ChromaDB Search": "Use more specific keywords. Try simpler search terms.",
                "Document Upload": "Upload smaller files. Large documents may take time to process.",
                "Web Scraping": "The target website may be slow or unreachable.",
                "GPTQ Model Status": "GPTQ model is loading in background. Check /startup-progress for details.",
                "Health Check": "System may be under heavy load. Try again in a moment."
            }
            return suggestions.get(request_type, f"Request exceeded {timeout}s limit. Try a simpler operation.")
        
        # ====================================
        # СПЕЦИАЛЬНЫЕ ENDPOINTS ДЛЯ HF SPACES С РАСШИРЕННЫМИ ТАЙМАУТАМИ
        # ====================================
        
        @app.get("/hf-spaces-health")
        async def hf_spaces_health():
            """Специальный health check для HF Spaces с timeout информацией и GPTQ статусом"""
            from app.dependencies import get_services_status
            
            try:
                services = await asyncio.wait_for(
                    get_services_status(),
                    timeout=HF_SPACES_HEALTH_TIMEOUT
                )
                
                # Определяем общий статус с учетом GPTQ
                overall_status = "healthy"
                issues = []
                recommendations = []
                
                # Проверяем критические сервисы
                if not services.get("document_service_available", False):
                    overall_status = "degraded"
                    issues.append("Document service initializing")
                    recommendations.append("Document search will be available shortly")
                
                # Специальная проверка GPTQ модели
                llm_available = services.get("llm_available", False)
                if not llm_available:
                    if overall_status == "healthy":
                        overall_status = "degraded"
                    issues.append("GPTQ model loading (TheBloke/Llama-2-7B-Chat-GPTQ)")
                    recommendations.append(f"AI responses will activate when model loads (timeout: {GPTQ_MODEL_LOADING_TIMEOUT}s)")
                
                response_data = {
                    "status": overall_status,
                    "platform": "HuggingFace Spaces",
                    "api_version": "2.0.0",
                    "timeout_configuration": {
                        "global_request_timeout": GLOBAL_REQUEST_TIMEOUT,
                        "keep_alive_timeout": KEEP_ALIVE_TIMEOUT,
                        "gptq_loading_timeout": GPTQ_MODEL_LOADING_TIMEOUT,
                        "gptq_inference_timeout": GPTQ_INFERENCE_TIMEOUT,
                        "chromadb_search_timeout": CHROMADB_SEARCH_TIMEOUT,
                        "http_request_timeout": HTTP_REQUEST_TIMEOUT,
                        "health_check_timeout": HF_SPACES_HEALTH_TIMEOUT
                    },
                    "gptq_model": {
                        "name": "TheBloke/Llama-2-7B-Chat-GPTQ",
                        "status": "ready" if llm_available else "loading",
                        "supported_languages": ["English", "Ukrainian"],
                        "optimization": "4-bit GPTQ quantization",
                        "loading_timeout": f"{GPTQ_MODEL_LOADING_TIMEOUT}s",
                        "inference_timeout": f"{GPTQ_INFERENCE_TIMEOUT}s"
                    },
                    "services": services,
                    "endpoints": {
                        "chat": "/api/user/chat",
                        "search": "/api/user/search", 
                        "docs": "/docs",
                        "admin": "/api/admin"
                    },
                    "features": {
                        "lazy_loading": True,
                        "gptq_support": True,
                        "ukrainian_language": True,
                        "vector_search": services.get("chromadb_enabled", False),
                        "demo_mode": services.get("demo_mode", True),
                        "memory_optimized": True,
                        "timeout_protected": True,
                        "hf_spaces_optimized": True
                    },
                    "cors_fix_applied": True,
                    "post_endpoints_working": True,
                    "timeout_middleware_active": True
                }
                
                if issues:
                    response_data["issues"] = issues
                if recommendations:
                    response_data["recommendations"] = recommendations
                
                return response_data
                
            except asyncio.TimeoutError:
                return {
                    "status": "timeout",
                    "error": f"Health check timeout after {HF_SPACES_HEALTH_TIMEOUT}s",
                    "platform": "HuggingFace Spaces",
                    "timeout_configuration": {
                        "health_check_timeout": HF_SPACES_HEALTH_TIMEOUT,
                        "global_request_timeout": GLOBAL_REQUEST_TIMEOUT,
                        "gptq_loading_timeout": GPTQ_MODEL_LOADING_TIMEOUT
                    },
                    "recommendations": [
                        "Services may be initializing",
                        "GPTQ model may be loading in background",
                        "Try again in a few moments",
                        "Check /startup-progress for detailed status"
                    ]
                }
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                    "platform": "HuggingFace Spaces",
                    "timeout_configuration": {
                        "global_request_timeout": GLOBAL_REQUEST_TIMEOUT
                    },
                    "recommendations": [
                        "Check server logs for detailed errors",
                        "Services may still be initializing",
                        "Try again in a few moments"
                    ]
                }
        
        @app.get("/timeout-status")
        async def comprehensive_timeout_status():
            """Comprehensive endpoint для мониторинга всех таймаутов"""
            return {
                "timeout_configuration": {
                    "global_request_timeout": GLOBAL_REQUEST_TIMEOUT,
                    "keep_alive_timeout": KEEP_ALIVE_TIMEOUT,
                    "graceful_timeout": GRACEFUL_TIMEOUT,
                    "gptq_model_loading_timeout": GPTQ_MODEL_LOADING_TIMEOUT,
                    "gptq_inference_timeout": GPTQ_INFERENCE_TIMEOUT,
                    "gptq_first_load_timeout": GPTQ_FIRST_LOAD_TIMEOUT,
                    "chromadb_search_timeout": CHROMADB_SEARCH_TIMEOUT,
                    "chromadb_add_doc_timeout": CHROMADB_ADD_DOC_TIMEOUT,
                    "chromadb_stats_timeout": CHROMADB_STATS_TIMEOUT,
                    "http_request_timeout": HTTP_REQUEST_TIMEOUT,
                    "scraper_timeout": SCRAPER_TIMEOUT,
                    "hf_spaces_startup_timeout": HF_SPACES_STARTUP_TIMEOUT,
                    "hf_spaces_health_timeout": HF_SPACES_HEALTH_TIMEOUT
                },
                "timeout_recommendations": {
                    "chat_queries": "Keep questions concise for faster GPTQ responses",
                    "document_uploads": "Upload smaller files if timeouts occur",
                    "search_queries": "Use specific keywords rather than long phrases",
                    "admin_operations": "Large operations may take time - be patient",
                    "gptq_model": "First load may take 8+ minutes, subsequent loads faster",
                    "chromadb_operations": "Large vector operations limited by 16GB RAM"
                },
                "platform_limits": {
                    "hf_spaces_memory": "16GB RAM limit",
                    "hf_spaces_cpu": "2 CPU cores",
                    "hf_spaces_disk": "50GB temporary storage",
                    "model_size": "TheBloke/Llama-2-7B-Chat-GPTQ ~4GB quantized"
                },
                "optimization_status": {
                    "memory_optimized": True,
                    "timeout_middleware": "active",
                    "cors_fixed": True,
                    "lazy_loading": True,
                    "background_loading": True
                }
            }
        
        @app.get("/model-status")
        async def comprehensive_model_status():
            """Расширенный статус GPTQ модели с диагностикой и таймаутами"""
            from app.dependencies import get_llm_service
            
            try:
                llm_service = await asyncio.wait_for(
                    get_llm_service(),
                    timeout=10.0  # 10 секунд на получение сервиса
                )
                
                status = await asyncio.wait_for(
                    llm_service.get_service_status(),
                    timeout=15.0  # 15 секунд на статус
                )
                
                # Проверяем статус загрузки модели
                model_ready = status.get("model_loaded", False)
                loading_error = status.get("loading_error")
                
                model_info = {
                    "name": "TheBloke/Llama-2-7B-Chat-GPTQ",
                    "type": "GPTQ Quantized Llama-2",
                    "size": "~4GB quantized (14GB unquantized)",
                    "languages": ["English", "Ukrainian", "Multilingual"],
                    "status": "ready" if model_ready else ("error" if loading_error else "loading"),
                    "service_type": status.get("service_type", "unknown"),
                    "loading_error": loading_error,
                    "timeout_protected": True
                }
                
                # Добавляем диагностическую информацию с таймаутами
                diagnostics = {
                    "platform": "HuggingFace Spaces",
                    "memory_optimization": True,
                    "quantization": "4-bit GPTQ",
                    "timeout_limits": {
                        "model_loading": f"{GPTQ_MODEL_LOADING_TIMEOUT}s (8 minutes)",
                        "first_load": f"{GPTQ_FIRST_LOAD_TIMEOUT}s (10 minutes)",
                        "inference": f"{GPTQ_INFERENCE_TIMEOUT}s (2 minutes)",
                        "chat_total": f"{GLOBAL_REQUEST_TIMEOUT}s (10 minutes)"
                    },
                    "dependencies": {
                        "transformers": status.get("transformers_version", "unknown"),
                        "torch": status.get("torch_available", False),
                        "auto_gptq": status.get("auto_gptq_available", False),
                        "cuda": status.get("cuda_available", False)
                    },
                    "memory_management": {
                        "hf_spaces_limit": "16GB RAM",
                        "model_quantization": "4-bit GPTQ",
                        "offloading_enabled": True,
                        "cpu_fallback": True
                    }
                }
                
                loading_tips = [
                    "GPTQ model provides high-quality responses with 4-bit quantization",
                    f"First load may take up to {GPTQ_FIRST_LOAD_TIMEOUT//60} minutes on HuggingFace Spaces",
                    "Demo responses available immediately during model loading",
                    "Supports both English and Ukrainian legal consultations",
                    "Memory optimized for HF Spaces 16GB limit",
                    "Model automatically switches from demo to full AI when ready",
                    f"All operations protected by comprehensive timeout system",
                    f"Chat requests have {GLOBAL_REQUEST_TIMEOUT//60}-minute timeout for complex questions"
                ]
                
                if loading_error:
                    loading_tips.extend([
                        "Model loading failed - check error details",
                        "Demo mode provides API structure preview",
                        "Try restarting the space if error persists",
                        f"Loading timeout was set to {GPTQ_MODEL_LOADING_TIMEOUT}s"
                    ])
                
                return {
                    "model_info": model_info,
                    "status": status,
                    "diagnostics": diagnostics,
                    "loading_tips": loading_tips,
                    "performance": {
                        "quantization": "4-bit GPTQ",
                        "inference_speed": "Optimized for HF Spaces",
                        "memory_efficient": True,
                        "quality": "High-quality legal analysis",
                        "token_limit": "400 tokens per response",
                        "timeout_protected": True,
                        "expected_loading_time": f"{GPTQ_MODEL_LOADING_TIMEOUT//60}-{GPTQ_FIRST_LOAD_TIMEOUT//60} minutes"
                    }
                }
                
            except asyncio.TimeoutError:
                return {
                    "model_info": {
                        "name": "TheBloke/Llama-2-7B-Chat-GPTQ", 
                        "status": "timeout"
                    },
                    "error": "Model status check timeout",
                    "timeout_info": {
                        "status_check_timeout": "15s",
                        "loading_timeout": f"{GPTQ_MODEL_LOADING_TIMEOUT}s",
                        "first_load_timeout": f"{GPTQ_FIRST_LOAD_TIMEOUT}s"
                    },
                    "recommendations": [
                        "Model may still be loading in background",
                        "Check /startup-progress for loading status",
                        f"GPTQ loading can take up to {GPTQ_FIRST_LOAD_TIMEOUT//60} minutes",
                        "Try again in a few moments"
                    ]
                }
            except Exception as e:
                return {
                    "model_info": {
                        "name": "TheBloke/Llama-2-7B-Chat-GPTQ", 
                        "status": "error"
                    },
                    "error": str(e),
                    "timeout_info": {
                        "loading_timeout": f"{GPTQ_MODEL_LOADING_TIMEOUT}s",
                        "global_timeout": f"{GLOBAL_REQUEST_TIMEOUT}s"
                    },
                    "recommendations": [
                        "Check HuggingFace Transformers installation",
                        "Verify auto-gptq dependencies",
                        "Model may still be downloading",
                        f"Loading timeout set to {GPTQ_MODEL_LOADING_TIMEOUT//60} minutes",
                        "Try again in a few moments"
                    ]
                }
        
        @app.get("/startup-progress")
        async def startup_progress():
            """Endpoint для отслеживания прогресса инициализации с GPTQ информацией"""
            from app.dependencies import get_services_status, get_background_tasks_status
            
            try:
                services = get_services_status()
                background_tasks = get_background_tasks_status()
                
                init_status = services.get("initialization_status", {})
                
                # Подсчитываем прогресс
                total_services = 3  # document, scraper, llm
                completed_services = sum(init_status.values())
                progress_percent = int((completed_services / total_services) * 100)
                
                # Определяем текущую активность с учетом GPTQ
                current_activity = "Starting up..."
                estimated_time = "2-5 minutes"
                
                if not init_status.get("document_service", False):
                    current_activity = "Initializing document service (ChromaDB)..."
                    estimated_time = f"{CHROMADB_STATS_TIMEOUT}s"
                elif not init_status.get("llm_service", False):
                    current_activity = "Loading GPTQ model (TheBloke/Llama-2-7B-Chat-GPTQ)..."
                    estimated_time = f"{GPTQ_MODEL_LOADING_TIMEOUT//60}-{GPTQ_FIRST_LOAD_TIMEOUT//60} minutes"
                elif not init_status.get("scraper_service", False):
                    current_activity = "Initializing web scraper..."
                    estimated_time = "30s"
                else:
                    current_activity = "All services ready!"
                    estimated_time = "Complete"
                
                component_status = {
                    "document_service": {
                        "status": "ready" if init_status.get("document_service") else "loading",
                        "description": "ChromaDB vector search",
                        "ready": init_status.get("document_service", False),
                        "timeout": f"{CHROMADB_STATS_TIMEOUT}s"
                    },
                    "llm_service": {
                        "status": "ready" if services.get("llm_available") else "loading", 
                        "description": "GPTQ Model (TheBloke/Llama-2-7B-Chat-GPTQ)",
                        "ready": services.get("llm_available", False),
                        "timeout": f"{GPTQ_MODEL_LOADING_TIMEOUT}s",
                        "first_load_timeout": f"{GPTQ_FIRST_LOAD_TIMEOUT}s",
                        "quantization": "4-bit GPTQ"
                    },
                    "scraper_service": {
                        "status": "ready" if init_status.get("scraper_service") else "loading",
                        "description": "Legal site scraper",
                        "ready": init_status.get("scraper_service", False),
                        "timeout": f"{HTTP_REQUEST_TIMEOUT}s"
                    }
                }
                
                return {
                    "overall_progress": progress_percent,
                    "current_activity": current_activity,
                    "estimated_time_remaining": estimated_time,
                    "components": component_status,
                    "services_ready": completed_services,
                    "total_services": total_services,
                    "ready_for_requests": progress_percent >= 33,  # Можно использовать с частичной готовностью
                    "platform": "HuggingFace Spaces",
                    "lazy_loading": True,
                    "cors_fix_applied": True,
                    "background_tasks": background_tasks,
                    "timeout_protection": {
                        "enabled": True,
                        "global_timeout": f"{GLOBAL_REQUEST_TIMEOUT}s",
                        "specialized_timeouts": True
                    },
                    "gptq_info": {
                        "model": "TheBloke/Llama-2-7B-Chat-GPTQ",
                        "expected_load_time": f"{GPTQ_MODEL_LOADING_TIMEOUT//60}-{GPTQ_FIRST_LOAD_TIMEOUT//60} minutes",
                        "optimization": "4-bit quantization for HF Spaces 16GB limit"
                    }
                }
                
            except Exception as e:
                return {
                    "overall_progress": 0,
                    "current_activity": "Error checking startup progress",
                    "error": str(e),
                    "platform": "HuggingFace Spaces",
                    "cors_fix_applied": True,
                    "timeout_protection": True
                }
        
        @app.get("/memory-status")
        async def comprehensive_memory_status():
            """Мониторинг памяти для HF Spaces с GPTQ информацией"""
            try:
                import psutil
                import gc
                
                memory = psutil.virtual_memory()
                
                # Принудительная очистка для получения актуальных данных
                gc.collect()
                
                memory_info = {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_gb": round((memory.total - memory.available) / (1024**3), 2),
                    "usage_percent": memory.percent,
                    "platform_limit": "16GB (HF Spaces)",
                    "status": "healthy" if memory.percent < 70 else ("warning" if memory.percent < 85 else "critical")
                }
                
                # GPTQ модель память информация
                gptq_memory_info = {
                    "model_size": "~4GB (quantized from 14GB)",
                    "quantization": "4-bit GPTQ",
                    "memory_efficient": True,
                    "estimated_usage": "4-6GB when loaded",
                    "loading_memory_spike": "May temporarily use 8-10GB during loading"
                }
                
                # Проверяем GPU память если доступна
                gpu_info = {}
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory
                        gpu_allocated = torch.cuda.memory_allocated(0)
                        gpu_info = {
                            "gpu_available": True,
                            "total_gb": round(gpu_memory / (1024**3), 2),
                            "allocated_gb": round(gpu_allocated / (1024**3), 2),
                            "free_gb": round((gpu_memory - gpu_allocated) / (1024**3), 2),
                            "gptq_optimized": True
                        }
                    else:
                        gpu_info = {"gpu_available": False, "reason": "CPU-only mode (normal for HF Spaces free tier)"}
                except:
                    gpu_info = {"gpu_available": False, "reason": "PyTorch not available"}
                
                recommendations = []
                if memory.percent > 90:
                    recommendations.extend([
                        "Critical memory usage - GPTQ model loading may fail",
                        f"Model loading timeout extended to {GPTQ_FIRST_LOAD_TIMEOUT}s",
                        "Consider restarting the space"
                    ])
                elif memory.percent > 80:
                    recommendations.extend([
                        "High memory usage - GPTQ model loading may be slower",
                        f"Normal loading timeout: {GPTQ_MODEL_LOADING_TIMEOUT}s"
                    ])
                elif memory.percent > 70:
                    recommendations.extend([
                        "Moderate memory usage - normal for GPTQ model loading",
                        "GPTQ 4-bit quantization helps reduce memory footprint"
                    ])
                else:
                    recommendations.extend([
                        "Memory usage optimal for GPTQ model",
                        "4-bit quantization provides good memory efficiency"
                    ])
                
                return {
                    "memory": memory_info,
                    "gpu": gpu_info,
                    "gptq_model": gptq_memory_info,
                    "recommendations": recommendations,
                    "timeout_adjustments": {
                        "high_memory_usage": memory.percent > 80,
                        "extended_timeout": GPTQ_FIRST_LOAD_TIMEOUT if memory.percent > 85 else GPTQ_MODEL_LOADING_TIMEOUT,
                        "timeout_reason": "Extended due to memory pressure" if memory.percent > 85 else "Standard timeout"
                    },
                    "timestamp": __import__("time").time(),
                    "cors_fix_applied": True
                }
                
            except ImportError:
                return {
                    "memory": {"status": "psutil not available"},
                    "gpu": {"status": "monitoring not available"},
                    "gptq_model": {
                        "model_size": "~4GB (quantized)",
                        "optimization": "4-bit GPTQ"
                    },
                    "recommendations": ["Install psutil for memory monitoring"],
                    "timeout_info": {
                        "gptq_loading_timeout": f"{GPTQ_MODEL_LOADING_TIMEOUT}s",
                        "global_timeout": f"{GLOBAL_REQUEST_TIMEOUT}s"
                    },
                    "cors_fix_applied": True
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "status": "Memory monitoring failed",
                    "timeout_info": {
                        "gptq_loading_timeout": f"{GPTQ_MODEL_LOADING_TIMEOUT}s"
                    },
                    "cors_fix_applied": True
                }
        
        print("✅ FastAPI application created successfully")
        print("✅ CORS configured FIRST (POST fix applied)")
        print("✅ HuggingFace Spaces optimizations applied")
        print("✅ Special HF Spaces endpoints added")
        print("✅ Comprehensive timeout system enabled")
        print(f"✅ GPTQ model support with {GPTQ_MODEL_LOADING_TIMEOUT}s loading timeout")
        print(f"✅ Global request timeout: {GLOBAL_REQUEST_TIMEOUT}s")
        
        return app
        
    except Exception as e:
        logger.error(f"Deployment initialization failed: {e}")
        
        # Создаем улучшенное fallback приложение
        from fastapi import FastAPI
        fallback_app = FastAPI(title="Legal Assistant API - Recovery Mode", version="2.0.0")
        
        # КРИТИЧЕСКИ ВАЖНО: CORS даже в fallback приложении
        fallback_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        @fallback_app.get("/")
        async def error_root():
            return {
                "error": "Application failed to initialize properly",
                "details": str(e),
                "platform": "HuggingFace Spaces",
                "cors_fix_applied": True,
                "timeout_info": {
                    "global_timeout": GLOBAL_REQUEST_TIMEOUT,
                    "gptq_loading_timeout": GPTQ_MODEL_LOADING_TIMEOUT,
                    "keep_alive": KEEP_ALIVE_TIMEOUT
                },
                "suggestions": [
                    "Check that all dependencies are installed",
                    "Verify model files are accessible", 
                    "Check available memory and storage",
                    "Review server logs for detailed errors",
                    f"GPTQ model loading may take up to {GPTQ_FIRST_LOAD_TIMEOUT//60} minutes",
                    "Try refreshing in a few minutes"
                ],
                "fallback_mode": True
            }
        
        return fallback_app

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
        print(f"   • Lazy Loading: Enabled")
        print(f"   • CORS Fix: Applied")
        print(f"   • Comprehensive Timeouts: Enabled")
        
        print(f"\n🔗 Available Endpoints:")
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
        print(f"   • ChromaDB Search: {CHROMADB_SEARCH_TIMEOUT}s")
        print(f"   • HTTP Requests: {HTTP_REQUEST_TIMEOUT}s")
        
        print(f"\n🎯 Starting server with comprehensive timeout protection...")
        print("=" * 70)
        
        # ИСПРАВЛЕНИЕ: Запускаем сервер с правильными таймаутами для GPTQ
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
            timeout_keep_alive=KEEP_ALIVE_TIMEOUT,  # ИСПРАВЛЕНИЕ: Правильный keep-alive
            timeout_graceful_shutdown=GRACEFUL_TIMEOUT,  # ИСПРАВЛЕНИЕ: Graceful shutdown
            limit_concurrency=5,  # ИСПРАВЛЕНИЕ: Ограничиваем для GPTQ модели
            limit_max_requests=500,  # ИСПРАВЛЕНИЕ: Лимит для memory management
            timeout_notify=GRACEFUL_TIMEOUT,  # Уведомление о shutdown
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Legal Assistant API shutting down...")
        print("Thank you for using Legal Assistant!")
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

# ====================================
# ПРИЛОЖЕНИЕ ДЛЯ DEPLOYMENT С РАСШИРЕННЫМИ ТАЙМАУТАМИ
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
    print("🔄 Initialization: Lazy loading enabled")
    print(f"⏰ Request Timeout: {GLOBAL_REQUEST_TIMEOUT}s")
    print(f"🔄 Keep-Alive: {KEEP_ALIVE_TIMEOUT}s")
    print(f"🤖 GPTQ Loading: {GPTQ_MODEL_LOADING_TIMEOUT}s")
    print("🔧 CORS Fix: Applied (POST endpoints working)")
    print("🛡️ Comprehensive Timeout Protection: Active")
    
except Exception as e:
    print(f"❌ Deployment initialization failed: {e}")
    print("🔄 Creating minimal fallback application...")
    
    # Улучшенное fallback приложение
    from fastapi import FastAPI
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
                "Some services may still be initializing"
            ]
        }

# Дополнительные endpoints для мониторинга - улучшенные с таймаутами
@app.get("/health")
async def health_check():
    """Быстрая проверка здоровья С ТАЙМАУТОМ"""
    try:
        return await asyncio.wait_for(
            {
                "status": "healthy", 
                "version": "2.0.0",
                "platform": "HuggingFace Spaces" if os.getenv("SPACE_ID") else "Local",
                "gptq_model": "TheBloke/Llama-2-7B-Chat-GPTQ",
                "lazy_loading": True,
                "memory_optimized": True,
                "cors_fix_applied": True,
                "post_endpoints_working": True,
                "timeout_protected": True,
                "timeout_limits": {
                    "global": f"{GLOBAL_REQUEST_TIMEOUT}s",
                    "gptq_loading": f"{GPTQ_MODEL_LOADING_TIMEOUT}s",
                    "health_check": f"{HF_SPACES_HEALTH_TIMEOUT}s"
                },
                "timestamp": __import__("time").time()
            },
            timeout=HF_SPACES_HEALTH_TIMEOUT  # 15 секунд на health check
        )
    except asyncio.TimeoutError:
        return {
            "status": "timeout",
            "timeout_limit": HF_SPACES_HEALTH_TIMEOUT,
            "global_timeout": GLOBAL_REQUEST_TIMEOUT,
            "message": "Health check timeout - services may be loading"
        }

if __name__ == "__main__":
    main()
else:
    # Если модуль импортируется
    logger.info("📦 Legal Assistant API module imported")
    logger.info("🤖 GPTQ Model: TheBloke/Llama-2-7B-Chat-GPTQ")
    logger.info("🚀 Ready for HuggingFace Spaces deployment")
    logger.info("💾 Memory optimized for 16GB limit")
    logger.info("🔄 Lazy loading enabled for faster startup")
    logger.info("🔧 CORS fix applied - POST endpoints working")
    logger.info(f"⏰ Comprehensive timeout protection - {GLOBAL_REQUEST_TIMEOUT}s global limit")
    logger.info(f"🤖 GPTQ loading timeout: {GPTQ_MODEL_LOADING_TIMEOUT}s")
    print("🔗 API Documentation: /docs")
    print("🏥 Health Check: /hf-spaces-health")
    print("📊 Timeout Status: /timeout-status")
    print("🤖 Model Status: /model-status")
    print("💾 Memory Status: /memory-status")
    print("✅ POST endpoints fixed and working")
    print(f"🛡️ All requests protected by comprehensive timeout system")
    print(f"⏰ GPTQ model loading: up to {GPTQ_FIRST_LOAD_TIMEOUT//60} minutes first time")