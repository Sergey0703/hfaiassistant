# backend/utils/startup_banner.py
"""
Утилиты для стартапа приложения: баннер, проверка окружения, зависимостей
"""

import os
import sys
import logging
from pathlib import Path
from config.timeouts import (
    GLOBAL_REQUEST_TIMEOUT, KEEP_ALIVE_TIMEOUT, GPTQ_MODEL_LOADING_TIMEOUT,
    CHROMADB_SEARCH_TIMEOUT, GPTQ_FIRST_LOAD_TIMEOUT
)

logger = logging.getLogger(__name__)

def print_startup_banner():
    """Улучшенный баннер для HF Spaces с информацией о таймаутах и React"""
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║                🏛️  Legal Assistant API v2.0 (HF Spaces)      ║
╠══════════════════════════════════════════════════════════════╣
║  AI Legal Assistant with GPTQ Model + React Frontend        ║
║  • TheBloke/Llama-2-7B-Chat-GPTQ Integration               ║
║  • React SPA with FastAPI Backend                           ║
║  • ChromaDB Vector Search with Lazy Loading                 ║
║  • Multi-language Support (English/Ukrainian)               ║
║  • Real-time Document Processing                            ║
║  • Optimized Memory Management for HF Spaces                ║
║  🚀 Production Ready with Graceful Degradation              ║
║  ⏰ Request Timeout: {GLOBAL_REQUEST_TIMEOUT}s (10 min)                     ║
║  🔄 Keep-Alive: {KEEP_ALIVE_TIMEOUT}s                                    ║
║  🤖 GPTQ Loading: {GPTQ_MODEL_LOADING_TIMEOUT}s (8 min)                        ║
║  📚 ChromaDB Search: {CHROMADB_SEARCH_TIMEOUT}s                           ║
║  ⚛️ React SPA: Enabled                                      ║
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
            "LLM_TIMEOUT": str(120),                # 2 минуты timeout для GPTQ inference
            "MAX_CONTEXT_DOCUMENTS": "2",          # Ограичиваем контекст для памяти
            "CONTEXT_TRUNCATE_LENGTH": "800",      # Сокращаем контекст для HF Spaces
            "LLM_MAX_TOKENS": "400",               # Ограничиваем токены для памяти
            "LLM_TEMPERATURE": "0.2",              # Консервативная температура
            "SERVE_REACT": "true",                 # Включаем React SPA
        }
        
        applied_settings = []
        for key, value in hf_optimizations.items():
            if not os.getenv(key):  # Устанавливаем только если не задано
                os.environ[key] = value
                applied_settings.append(f"{key}={value}")
        
        if applied_settings:
            print("   ⚙️ Applied HF Spaces optimizations:")
            for setting in applied_settings[:5]:  # Показываем первые 5
                print(f"      - {setting}")
            if len(applied_settings) > 5:
                print(f"      - ... and {len(applied_settings) - 5} more settings")
        
        # Проверяем доступную память и ресурсы
        _check_system_resources()
        
        print("   ✅ HF Spaces environment configured with optimized timeouts")
        
    else:
        print("   💻 Local development environment")
        print(f"   • Python: {sys.version.split()[0]}")
        print(f"   • Working Dir: {os.getcwd()}")
        
        # Для локальной разработки используем другие настройки
        if not os.getenv("LLM_DEMO_MODE"):
            os.environ.setdefault("LLM_DEMO_MODE", "false")  # Реальная модель по умолчанию
        if not os.getenv("SERVE_REACT"):
            os.environ.setdefault("SERVE_REACT", "true")     # React по умолчанию
    
    return is_hf_spaces

def _check_system_resources():
    """Проверяет системные ресурсы"""
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

def check_critical_dependencies():
    """Проверяет только критические зависимости с версиями для GPTQ + React"""
    print("🔍 Critical Dependencies Check:")
    
    critical_deps = [
        ("fastapi", "FastAPI framework"),
        ("uvicorn", "ASGI server"),
        ("transformers", "HuggingFace Transformers (for GPTQ)"),
        ("torch", "PyTorch (for GPTQ model)")
    ]
    
    missing_critical = []
    
    for dep_name, description in critical_deps.items():
        try:
            module = __import__(dep_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"   ✅ {dep_name} ({version}): {description}")
        except ImportError:
            print(f"   ❌ {dep_name}: {description}")
            missing_critical.append(dep_name)
    
    # Опциональные зависимости с версиями - ВАЖНЫЕ ДЛЯ GPTQ + React
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
    
    # Проверяем React build
    print("\n⚛️ React Frontend Check:")
    react_build_path = Path(__file__).parent.parent.parent / "frontend" / "build"
    if react_build_path.exists():
        print(f"   ✅ React build found: {react_build_path}")
        index_html = react_build_path / "index.html"
        if index_html.exists():
            print("   ✅ React index.html ready")
        else:
            print("   ⚠️ React index.html missing")
    else:
        print(f"   ⚠️ React build not found: {react_build_path}")
        print("   💡 Run: cd frontend && npm run build")
    
    print(f"\n📊 Dependencies Summary:")
    print(f"   • Critical: {len(critical_deps) - len(missing_critical)}/{len(critical_deps)} available")
    print(f"   • Optional: {optional_available}/{len(optional_deps)} available")
    print(f"   • React: {'✅ Ready' if react_build_path.exists() else '⚠️ Not built'}")
    
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
    """Безопасно создает необходимые директории для HF Spaces + React"""
    directories = [
        "logs",
        "chromadb_data", 
        "uploads",
        "temp",
        "backups",
        ".cache",
        "offload",  # Для model offloading в HF Spaces
        "frontend/build"  # На случай если React не собран
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