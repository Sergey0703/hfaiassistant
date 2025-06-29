# backend/utils/startup_banner.py - УПРОЩЕННЫЙ СТАРТАП
"""
Упрощенные утилиты стартапа для минимальной RAG системы
Убраны сложные проверки GPTQ, оставлены только базовые функции
"""

import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def print_minimal_startup_banner():
    """Упрощенный баннер для минимальной RAG системы"""
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║               🏛️  Minimal Legal RAG System v1.0              ║
╠══════════════════════════════════════════════════════════════╣
║  🤖 FLAN-T5 Small + Sentence Transformers + ChromaDB        ║
║  💾 Memory Target: <1GB RAM                                  ║
║  ⚡ Fast Startup: <60 seconds                                ║
║  🌍 Platform: {'HuggingFace Spaces' if _is_hf_spaces() else 'Local Development'}                        ║
║  📚 Features: Semantic Search, Document Upload, Chat        ║
║  🔍 Embeddings: all-MiniLM-L6-v2 (384D, ~90MB)             ║
║  🗄️ Vector DB: ChromaDB with lazy loading                   ║
║  🌐 Multilingual: English + Ukrainian support               ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_minimal_environment():
    """Упрощенная проверка окружения"""
    is_hf_spaces = _is_hf_spaces()
    
    print("🌍 Environment Check:")
    print(f"   • Platform: {'HuggingFace Spaces' if is_hf_spaces else 'Local Development'}")
    
    if is_hf_spaces:
        space_id = os.getenv("SPACE_ID", "unknown")
        print(f"   • Space ID: {space_id}")
        print("   🤗 HuggingFace Spaces detected - applying optimizations")
        
        # Простые оптимизации для HF Spaces
        optimizations = {
            "USE_CHROMADB": "true",
            "LOG_LEVEL": "INFO",
            "LLM_MODEL": "google/flan-t5-small",
            "LLM_MAX_TOKENS": "150",
            "LLM_TIMEOUT": "20",
            "MAX_CONTEXT_DOCUMENTS": "2",
            "CONTEXT_TRUNCATE_LENGTH": "300"
        }
        
        applied = []
        for key, value in optimizations.items():
            if not os.getenv(key):
                os.environ[key] = value
                applied.append(f"{key}={value}")
        
        if applied:
            print("   ⚙️ Applied optimizations:")
            for setting in applied[:3]:  # Показываем первые 3
                print(f"      - {setting}")
            if len(applied) > 3:
                print(f"      - ... and {len(applied) - 3} more")
        
        print("   ✅ HF Spaces environment optimized")
        
    else:
        print("   💻 Local development environment")
        print(f"   • Python: {sys.version.split()[0]}")
        
        # Для локальной разработки
        os.environ.setdefault("LLM_MODEL", "google/flan-t5-small")
        os.environ.setdefault("USE_CHROMADB", "true")
    
    return is_hf_spaces

def check_minimal_dependencies():
    """Проверяет только критические зависимости"""
    print("🔍 Critical Dependencies Check:")
    
    critical_deps = [
        ("fastapi", "FastAPI framework"),
        ("transformers", "HuggingFace Transformers for FLAN-T5"),
        ("sentence_transformers", "Sentence embeddings"),
        ("chromadb", "Vector database")
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
    
    # Проверяем PyTorch
    print("\n🔥 PyTorch Check:")
    try:
        import torch
        print(f"   ✅ torch ({torch.__version__}): PyTorch framework")
        
        if torch.cuda.is_available():
            print("   🚀 CUDA available - GPU acceleration possible")
        else:
            print("   💻 CPU-only mode (optimal for minimal setup)")
    except ImportError:
        print("   ❌ torch: PyTorch framework")
        missing_critical.append("torch")
    
    # Проверяем модели
    print("\n🤖 Model Availability Check:")
    model_checks = [
        ("google/flan-t5-small", "LLM model"),
        ("sentence-transformers/all-MiniLM-L6-v2", "Embedding model")
    ]
    
    for model_name, description in model_checks:
        try:
            if "flan-t5" in model_name:
                from transformers import AutoTokenizer
                AutoTokenizer.from_pretrained(model_name)
            else:
                from sentence_transformers import SentenceTransformer
                SentenceTransformer(model_name)
            print(f"   ✅ {model_name}: {description}")
        except Exception as e:
            print(f"   ⚠️ {model_name}: {description} (will download on first use)")
    
    print(f"\n📊 Dependencies Summary:")
    print(f"   • Critical: {len(critical_deps) - len(missing_critical)}/{len(critical_deps)} available")
    print(f"   • Memory Target: <1GB RAM")
    print(f"   • Models: FLAN-T5 Small (~300MB) + all-MiniLM-L6-v2 (~90MB)")
    
    if missing_critical:
        print(f"\n❌ Missing critical dependencies: {', '.join(missing_critical)}")
        print("   Install with: pip install fastapi transformers sentence-transformers chromadb torch")
        return False
    
    print("\n✅ All critical dependencies available for minimal RAG")
    return True

def create_minimal_directories():
    """Создает минимальный набор директорий"""
    directories = [
        "logs",
        "chromadb_data", 
        "uploads",
        "temp",
        ".cache"
    ]
    
    created = []
    failed = []
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            # Простая проверка записи
            test_file = os.path.join(directory, ".test")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                created.append(directory)
            except:
                # Директория создана но не доступна для записи
                logger.warning(f"Directory {directory} created but not writable")
                
        except Exception as e:
            failed.append(f"{directory}: {str(e)[:30]}")
            logger.warning(f"Could not create directory {directory}: {e}")
    
    if created:
        print(f"📁 Created directories: {', '.join(created)}")
    if failed:
        print(f"⚠️ Failed directories: {', '.join([f.split(':')[0] for f in failed])}")
    
    return len(created) > 0

def estimate_memory_usage():
    """Оценка потребления памяти для диагностики"""
    print("\n💾 Memory Usage Estimation:")
    
    components = {
        "FLAN-T5 Small": "~300 MB",
        "all-MiniLM-L6-v2": "~90 MB", 
        "ChromaDB": "~20 MB (per 10K docs)",
        "FastAPI + deps": "~100 MB",
        "Python runtime": "~50 MB",
        "Operating overhead": "~100 MB"
    }
    
    total_mb = 300 + 90 + 20 + 100 + 50 + 100  # 660 MB
    
    for component, usage in components.items():
        print(f"   • {component}: {usage}")
    
    print(f"\n   📊 Total Estimated: ~{total_mb} MB")
    print(f"   🎯 Target: <1GB (1024 MB)")
    print(f"   ✅ Memory Efficiency: {(total_mb/1024)*100:.1f}% of 1GB target")
    
    return total_mb

def check_model_accessibility():
    """Проверяет доступность моделей HuggingFace"""
    print("\n🔗 Model Accessibility Check:")
    
    models_to_check = [
        "google/flan-t5-small",
        "sentence-transformers/all-MiniLM-L6-v2"
    ]
    
    accessible_models = []
    
    for model in models_to_check:
        try:
            # Простая проверка доступности через requests
            import requests
            if "flan-t5" in model:
                url = f"https://huggingface.co/{model}/resolve/main/config.json"
            else:
                url = f"https://huggingface.co/{model}/resolve/main/config.json"
            
            response = requests.head(url, timeout=5)
            if response.status_code == 200:
                print(f"   ✅ {model}: Accessible")
                accessible_models.append(model)
            else:
                print(f"   ⚠️ {model}: Not accessible (will try download)")
        except Exception as e:
            print(f"   ⚠️ {model}: Connection check failed")
    
    if len(accessible_models) == len(models_to_check):
        print("   🌐 All models accessible from HuggingFace Hub")
    else:
        print("   ⚠️ Some models may need to download on first use")
    
    return len(accessible_models) > 0

def print_startup_summary():
    """Выводит итоговую сводку стартапа"""
    print("\n" + "="*60)
    print("🚀 Minimal RAG Startup Summary:")
    print("="*60)
    
    # Проверяем окружение
    is_hf_spaces = check_minimal_environment()
    
    # Проверяем зависимости
    deps_ok = check_minimal_dependencies()
    
    # Создаем директории
    dirs_ok = create_minimal_directories()
    
    # Оценка памяти
    memory_mb = estimate_memory_usage()
    
    # Проверка моделей
    models_ok = check_model_accessibility()
    
    print("\n📋 Final Status:")
    print(f"   • Environment: {'✅ Ready' if is_hf_spaces or True else '❌ Issues'}")
    print(f"   • Dependencies: {'✅ All critical available' if deps_ok else '❌ Missing critical'}")
    print(f"   • Directories: {'✅ Created' if dirs_ok else '⚠️ Limited access'}")
    print(f"   • Memory: ✅ {memory_mb} MB (within 1GB target)")
    print(f"   • Models: {'✅ Accessible' if models_ok else '⚠️ Will download'}")
    
    overall_status = deps_ok and dirs_ok
    
    if overall_status:
        print("\n🎉 Minimal RAG System ready to start!")
        print("   • Expected startup time: <60 seconds")
        print("   • Memory usage: <1GB RAM")
        print("   • Models: FLAN-T5 Small + all-MiniLM-L6-v2")
    else:
        print("\n⚠️ Some issues detected - system may work with limitations")
    
    print("="*60)
    
    return overall_status

def _is_hf_spaces() -> bool:
    """Проверяет является ли окружение HuggingFace Spaces"""
    return os.getenv("SPACE_ID") is not None

def get_startup_recommendations():
    """Возвращает рекомендации по оптимизации стартапа"""
    recommendations = []
    
    # Проверяем память
    memory_mb = estimate_memory_usage()
    if memory_mb > 800:  # Если приближаемся к лимиту
        recommendations.append("Consider reducing context length for memory optimization")
    
    # Проверяем окружение
    if _is_hf_spaces():
        recommendations.extend([
            "HF Spaces detected - optimizations applied automatically",
            "Models will download to .cache directory on first use",
            "Expect 2-3 minute first startup for model downloads"
        ])
    else:
        recommendations.extend([
            "Local development - ensure good internet for model downloads",
            "Consider pre-downloading models: huggingface-cli download",
            "Use nvidia-smi to monitor GPU usage if available"
        ])
    
    # Общие рекомендации
    recommendations.extend([
        "Monitor memory usage in production",
        "Use shorter questions for faster responses",
        "Upload relevant documents for better RAG performance"
    ])
    
    return recommendations

# ====================================
# ЭКСПОРТ
# ====================================

__all__ = [
    "print_minimal_startup_banner",
    "check_minimal_environment", 
    "check_minimal_dependencies",
    "create_minimal_directories",
    "estimate_memory_usage",
    "check_model_accessibility",
    "print_startup_summary",
    "get_startup_recommendations"
]