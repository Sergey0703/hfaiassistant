# backend/app/config.py - УПРОЩЁННАЯ КОНФИГУРАЦИЯ (ПОЛНАЯ ВЕРСИЯ)
"""
Упрощённая конфигурация без множественных настроек LLM и сложных валидаций
Заменяет переусложнённый config.py с Ollama, GPTQ и множественными таймаутами
"""

import os
from typing import List

# ====================================
# ОСНОВНЫЕ НАСТРОЙКИ ПРИЛОЖЕНИЯ
# ====================================

class Settings:
    """Упрощённый класс настроек"""
    
    # Основные настройки API
    API_V1_PREFIX: str = "/api"
    PROJECT_NAME: str = "Legal Assistant API"
    VERSION: str = "2.0.0"
    DESCRIPTION: str = "AI Legal Assistant with Llama integration"
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"  # Для HuggingFace Spaces
    ]
    
    # ====================================
    # LLAMA LLM НАСТРОЙКИ (УПРОЩЁННЫЕ)
    # ====================================
    
    # Основные модели
    LLM_PRIMARY_MODEL: str = "meta-llama/Llama-3.1-8B-Instruct"
    LLM_FAST_MODEL: str = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Параметры генерации
    LLM_MAX_TOKENS: int = 200
    LLM_TEMPERATURE: float = 0.3  # Консервативно для юридических вопросов
    LLM_TIMEOUT: int = 30  # Единый таймаут
    
    # HuggingFace настройки
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    
    # ====================================
    # БАЗА ДАННЫХ И ПОИСК
    # ====================================
    
    # ChromaDB
    USE_CHROMADB: bool = os.getenv("USE_CHROMADB", "true").lower() == "true"
    CHROMADB_PATH: str = os.getenv("CHROMADB_PATH", "./chromadb_data")
    
    # Эмбеддинги
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Поиск
    DEFAULT_SEARCH_LIMIT: int = 5
    MAX_SEARCH_LIMIT: int = 20
    MAX_CONTEXT_DOCUMENTS: int = 2  # Упрощено для производительности
    CONTEXT_TRUNCATE_LENGTH: int = 500  # Сокращено для скорости
    
    # ====================================
    # ФАЙЛЫ И ЗАГРУЗКИ
    # ====================================
    
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: List[str] = [".txt", ".pdf", ".docx", ".md"]
    
    # ====================================
    # ВЕБ-СКРАПИНГ
    # ====================================
    
    SCRAPING_DELAY: float = 1.5
    SCRAPING_TIMEOUT: int = 15
    MAX_URLS_PER_REQUEST: int = 10  # Уменьшено для стабильности
    
    # ====================================
    # ЛОГИРОВАНИЕ И МОНИТОРИНГ
    # ====================================
    
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # ====================================
    # СПЕЦИАЛЬНЫЕ РЕЖИМЫ
    # ====================================
    
    # Демо режим (если LLM недоступен)
    LLM_DEMO_MODE: bool = os.getenv("LLM_DEMO_MODE", "false").lower() == "true"
    
    # HuggingFace Spaces
    HUGGINGFACE_SPACES: bool = os.getenv("SPACE_ID") is not None
    
    # Языки
    SUPPORTED_LANGUAGES: List[str] = ["en", "uk"]
    DEFAULT_LANGUAGE: str = "en"

# Создаём глобальный экземпляр
settings = Settings()

# ====================================
# КОНСТАНТЫ И СПРАВОЧНИКИ
# ====================================

# API метаданные
API_METADATA = {
    "title": settings.PROJECT_NAME,
    "version": settings.VERSION,
    "description": settings.DESCRIPTION,
    "contact": {
        "name": "Legal Assistant Team",
        "email": "support@legalassistant.com"
    }
}

# API теги для документации
API_TAGS = [
    {
        "name": "User Chat",
        "description": "User chat endpoints for legal assistance"
    },
    {
        "name": "User Search", 
        "description": "Document search endpoints for users"
    },
    {
        "name": "Admin Documents",
        "description": "Document management for administrators"
    },
    {
        "name": "Admin Scraper",
        "description": "Web scraping for administrators"
    },
    {
        "name": "Admin Stats",
        "description": "Statistics and analytics"
    },
    {
        "name": "Admin LLM",
        "description": "LLM management and monitoring"
    },
    {
        "name": "System",
        "description": "System health and information"
    }
]

# Категории документов
DOCUMENT_CATEGORIES = [
    "general",
    "legislation", 
    "jurisprudence",
    "government",
    "civil_rights",
    "scraped",
    "ukraine_legal",
    "ireland_legal"
]

# Предустановленные юридические сайты
UKRAINE_LEGAL_URLS = [
    "https://zakon.rada.gov.ua/laws/main",
    "https://court.gov.ua/",
    "https://minjust.gov.ua/",
    "https://ccu.gov.ua/"
]

IRELAND_LEGAL_URLS = [
    "https://www.irishstatutebook.ie/",
    "https://www.courts.ie/",
    "https://www.citizensinformation.ie/en/",
    "https://www.justice.ie/"
]

# ====================================
# ФУНКЦИИ КОНФИГУРАЦИИ
# ====================================

def get_llm_config() -> dict:
    """Возвращает конфигурацию LLM (упрощённую)"""
    return {
        "primary_model": settings.LLM_PRIMARY_MODEL,
        "fast_model": settings.LLM_FAST_MODEL,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "temperature": settings.LLM_TEMPERATURE,
        "timeout": settings.LLM_TIMEOUT,
        "hf_token_configured": bool(settings.HF_TOKEN),
        "demo_mode": settings.LLM_DEMO_MODE,
        "supported_languages": settings.SUPPORTED_LANGUAGES
    }

def get_database_config() -> dict:
    """Возвращает конфигурацию базы данных"""
    return {
        "use_chromadb": settings.USE_CHROMADB,
        "chromadb_path": settings.CHROMADB_PATH,
        "embedding_model": settings.EMBEDDING_MODEL,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "max_context_documents": settings.MAX_CONTEXT_DOCUMENTS
    }

def get_api_config() -> dict:
    """Возвращает конфигурацию API"""
    return {
        "project_name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "description": settings.DESCRIPTION,
        "cors_origins": settings.CORS_ORIGINS,
        "api_prefix": settings.API_V1_PREFIX
    }

def validate_config() -> dict:
    """Простая валидация конфигурации"""
    issues = []
    warnings = []
    
    # Проверяем HF токен
    if not settings.HF_TOKEN and not settings.LLM_DEMO_MODE:
        warnings.append("HF_TOKEN not set - using public inference (rate limited)")
    
    # Проверяем пути
    if settings.USE_CHROMADB:
        try:
            os.makedirs(settings.CHROMADB_PATH, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create ChromaDB directory: {e}")
    
    # Проверяем настройки LLM
    if settings.LLM_MAX_TOKENS < 50:
        warnings.append("LLM_MAX_TOKENS is very low (< 50)")
    
    if settings.LLM_TEMPERATURE < 0 or settings.LLM_TEMPERATURE > 1:
        issues.append("LLM_TEMPERATURE must be between 0 and 1")
    
    # Проверяем размер файлов
    if settings.MAX_FILE_SIZE > 50 * 1024 * 1024:  # 50MB
        warnings.append("MAX_FILE_SIZE is very large - may cause memory issues")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "config_summary": {
            "llm_model": settings.LLM_PRIMARY_MODEL,
            "chromadb_enabled": settings.USE_CHROMADB,
            "hf_spaces": settings.HUGGINGFACE_SPACES,
            "demo_mode": settings.LLM_DEMO_MODE
        }
    }

def get_environment_info() -> dict:
    """Информация об окружении"""
    return {
        "huggingface_spaces": settings.HUGGINGFACE_SPACES,
        "space_id": os.getenv("SPACE_ID"),
        "demo_mode": settings.LLM_DEMO_MODE,
        "log_level": settings.LOG_LEVEL,
        "chromadb_enabled": settings.USE_CHROMADB,
        "supported_languages": settings.SUPPORTED_LANGUAGES,
        "environment_variables": {
            "HF_TOKEN": "configured" if settings.HF_TOKEN else "not_set",
            "USE_CHROMADB": os.getenv("USE_CHROMADB", "true"),
            "LLM_DEMO_MODE": os.getenv("LLM_DEMO_MODE", "false"),
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
            "SPACE_ID": "configured" if os.getenv("SPACE_ID") else "not_set"
        }
    }

def get_full_config_summary() -> dict:
    """Полная сводка конфигурации"""
    validation = validate_config()
    
    return {
        "validation": validation,
        "llm": get_llm_config(),
        "database": get_database_config(),
        "api": get_api_config(),
        "environment": get_environment_info(),
        "categories": DOCUMENT_CATEGORIES,
        "predefined_sites": {
            "ukraine": len(UKRAINE_LEGAL_URLS),
            "ireland": len(IRELAND_LEGAL_URLS)
        }
    }

# ====================================
# ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ СОВМЕСТИМОСТИ С АДМИН API
# ====================================

def validate_llm_config() -> dict:
    """Валидация LLM конфигурации (для совместимости с админ API)"""
    issues = []
    warnings = []
    
    # Проверяем HF токен
    if not settings.HF_TOKEN:
        warnings.append("HF_TOKEN not set - using public inference (rate limited)")
    
    # Проверяем модели
    if not settings.LLM_PRIMARY_MODEL:
        issues.append("LLM_PRIMARY_MODEL not configured")
    
    if not settings.LLM_FAST_MODEL:
        warnings.append("LLM_FAST_MODEL not configured - will use primary model")
    
    # Проверяем параметры
    if settings.LLM_TEMPERATURE < 0 or settings.LLM_TEMPERATURE > 1:
        issues.append("LLM_TEMPERATURE must be between 0 and 1")
    
    if settings.LLM_MAX_TOKENS < 10:
        issues.append("LLM_MAX_TOKENS too low (minimum 10)")
    
    if settings.LLM_TIMEOUT < 5:
        warnings.append("LLM_TIMEOUT very low - may cause request failures")
    
    # Проверяем доступность моделей
    try:
        from huggingface_hub import InferenceClient
        huggingface_available = True
    except ImportError:
        huggingface_available = False
        issues.append("huggingface_hub not installed - LLM service will not work")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "config": {
            "primary_model": settings.LLM_PRIMARY_MODEL,
            "fast_model": settings.LLM_FAST_MODEL,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "temperature": settings.LLM_TEMPERATURE,
            "timeout": settings.LLM_TIMEOUT,
            "hf_token_configured": bool(settings.HF_TOKEN),
            "demo_mode": settings.LLM_DEMO_MODE,
            "huggingface_available": huggingface_available,
            "supported_languages": settings.SUPPORTED_LANGUAGES
        },
        "recommendations": _get_llm_recommendations(issues, warnings, huggingface_available)
    }

def _get_llm_recommendations(issues: List[str], warnings: List[str], huggingface_available: bool) -> List[str]:
    """Генерирует рекомендации по LLM конфигурации"""
    recommendations = []
    
    if not huggingface_available:
        recommendations.append("Install huggingface_hub: pip install huggingface_hub")
    
    if not settings.HF_TOKEN:
        recommendations.append("Set HF_TOKEN environment variable for better rate limits")
        recommendations.append("Get token at: https://huggingface.co/settings/tokens")
    
    if settings.LLM_TEMPERATURE > 0.5:
        recommendations.append("Consider lower temperature (0.2-0.3) for more consistent legal advice")
    
    if settings.LLM_MAX_TOKENS > 500:
        recommendations.append("Consider lower max_tokens for faster responses")
    
    if len(issues) == 0 and len(warnings) == 0:
        recommendations.append("LLM configuration is optimal")
    
    return recommendations

# ====================================
# ЭКСПОРТ
# ====================================

__all__ = [
    # Основной класс и экземпляр
    "Settings",
    "settings",
    
    # Константы
    "API_METADATA",
    "API_TAGS", 
    "DOCUMENT_CATEGORIES",
    "UKRAINE_LEGAL_URLS",
    "IRELAND_LEGAL_URLS",
    
    # Функции конфигурации
    "get_llm_config",
    "get_database_config",
    "get_api_config",
    "validate_config",
    "get_environment_info",
    "get_full_config_summary",
    
    # Совместимость с админ API
    "validate_llm_config"
]