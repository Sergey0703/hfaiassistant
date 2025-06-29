# backend/app/config.py - ПОЛНАЯ КОНФИГУРАЦИЯ С ПРИОРИТЕТОМ ENV VARIABLES
"""
Конфигурация с правильной обработкой переменных окружения HuggingFace Spaces
Environment variables имеют ПРИОРИТЕТ над значениями по умолчанию в коде
"""

import os
from typing import List

# ====================================
# ОСНОВНЫЕ НАСТРОЙКИ ПРИЛОЖЕНИЯ
# ====================================

class Settings:
    """Класс настроек с приоритетом переменных окружения"""
    
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
    # LLM НАСТРОЙКИ С ПРИОРИТЕТОМ ENV VARIABLES
    # ====================================
    
    # КРИТИЧНО: Environment variables перезаписывают эти значения!
    # Что вы видите в HF Spaces settings имеет приоритет
    
    # Проверенные рабочие модели (fallback если не задано в ENV)
    _DEFAULT_PRIMARY_MODEL = "google/flan-t5-base"  # Надежная
    _DEFAULT_FAST_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    _BACKUP_MODELS = [
    "meta-llama/Llama-2-7b-chat-hf",
    "distilgpt2", 
    "gpt2"
]
    
    # Приоритет: ENV -> Код
    @property
    def LLM_PRIMARY_MODEL(self) -> str:
        # Проверяем разные варианты названий переменных
        model = (
            os.getenv("LLM_MODEL") or           # Ваша переменная в HF Spaces
            os.getenv("LLM_PRIMARY_MODEL") or
            os.getenv("HUGGINGFACE_MODEL") or
            self._DEFAULT_PRIMARY_MODEL
        )
        print(f"🤖 Using LLM model: {model}")
        return model
    
    @property 
    def LLM_FAST_MODEL(self) -> str:
        return os.getenv("LLM_FAST_MODEL", self._DEFAULT_FAST_MODEL)
    
    @property
    def LLM_BACKUP_MODELS(self) -> List[str]:
        # Добавляем резервные модели если основная не работает
        backup_env = os.getenv("LLM_BACKUP_MODELS")
        if backup_env:
            return backup_env.split(",")
        return self._BACKUP_MODELS
    
    # Параметры генерации (тоже можно настроить через ENV)
    @property
    def LLM_MAX_TOKENS(self) -> int:
        return int(os.getenv("LLM_MAX_TOKENS", "200"))
    
    @property
    def LLM_TEMPERATURE(self) -> float:
        return float(os.getenv("LLM_TEMPERATURE", "0.3"))
    
    @property
    def LLM_TIMEOUT(self) -> int:
        return int(os.getenv("LLM_TIMEOUT", "30"))
    
    # HuggingFace настройки
    @property
    def HF_TOKEN(self) -> str:
        return os.getenv("HF_TOKEN", "")
    
    @property
    def LLM_DEMO_MODE(self) -> bool:
        return os.getenv("LLM_DEMO_MODE", "false").lower() == "true"
    
    # ====================================
    # БАЗА ДАННЫХ И ПОИСК
    # ====================================
    
    @property
    def USE_CHROMADB(self) -> bool:
        return os.getenv("USE_CHROMADB", "true").lower() == "true"
    
    @property 
    def CHROMADB_PATH(self) -> str:
        return os.getenv("CHROMADB_PATH", "./chromadb_data")
    
    # Эмбеддинги
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Поиск
    DEFAULT_SEARCH_LIMIT: int = 5
    MAX_SEARCH_LIMIT: int = 20
    MAX_CONTEXT_DOCUMENTS: int = 2  # Оптимизировано для HF Spaces
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
    MAX_URLS_PER_REQUEST: int = 10
    
    # ====================================
    # ЛОГИРОВАНИЕ И МОНИТОРИНГ
    # ====================================
    
    @property
    def LOG_LEVEL(self) -> str:
        return os.getenv("LOG_LEVEL", "INFO")
    
    # ====================================
    # СПЕЦИАЛЬНЫЕ РЕЖИМЫ
    # ====================================
    
    @property
    def HUGGINGFACE_SPACES(self) -> bool:
        return os.getenv("SPACE_ID") is not None
    
    # Языки
    SUPPORTED_LANGUAGES: List[str] = ["en", "uk"]
    DEFAULT_LANGUAGE: str = "en"
    
    def get_active_model_info(self) -> dict:
        """Возвращает информацию об активной модели"""
        return {
            "primary_model": self.LLM_PRIMARY_MODEL,
            "fast_model": self.LLM_FAST_MODEL, 
            "backup_models": self.LLM_BACKUP_MODELS,
            "source": "ENV variable" if os.getenv("LLM_MODEL") else "default config",
            "demo_mode": self.LLM_DEMO_MODE,
            "hf_token_configured": bool(self.HF_TOKEN),
            "parameters": {
                "max_tokens": self.LLM_MAX_TOKENS,
                "temperature": self.LLM_TEMPERATURE,
                "timeout": self.LLM_TIMEOUT
            }
        }
    
    def validate_model_availability(self) -> dict:
        """Проверяет доступность настроенной модели"""
        validation = {
            "primary_model_set": bool(self.LLM_PRIMARY_MODEL),
            "hf_token_available": bool(self.HF_TOKEN),
            "known_working_models": [
                "microsoft/phi-2",
                "google/flan-t5-base", 
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "microsoft/DialoGPT-small"
            ],
            "current_model": self.LLM_PRIMARY_MODEL,
            "is_known_working": self.LLM_PRIMARY_MODEL in [
                "microsoft/phi-2",
                "google/flan-t5-base",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                "microsoft/DialoGPT-small",
                "distilgpt2",
                "gpt2"
            ]
        }
        
        return validation

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
    """Возвращает конфигурацию LLM"""
    return settings.get_active_model_info()

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
    """Валидация конфигурации с проверкой модели"""
    issues = []
    warnings = []
    
    # Проверяем модель
    model_validation = settings.validate_model_availability()
    if not model_validation["is_known_working"]:
        warnings.append(f"Model '{settings.LLM_PRIMARY_MODEL}' may not work - consider tested alternatives")
    
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
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "model_info": model_validation,
        "config_summary": {
            "active_model": settings.LLM_PRIMARY_MODEL,
            "model_source": "ENV variable" if os.getenv("LLM_MODEL") else "default",
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
        "active_model": settings.LLM_PRIMARY_MODEL,
        "model_source": "ENV" if os.getenv("LLM_MODEL") else "CONFIG",
        "environment_variables": {
            "HF_TOKEN": "configured" if settings.HF_TOKEN else "not_set",
            "LLM_MODEL": os.getenv("LLM_MODEL", "not_set"),
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

def validate_llm_config() -> dict:
    """Валидация LLM конфигурации (для совместимости с админ API)"""
    return validate_config()

# ====================================
# МОДЕЛЬ НЕ РАБОТАЕТ? РЕКОМЕНДАЦИИ
# ====================================

def get_model_recommendations() -> dict:
    """Рекомендации по моделям на основе опыта"""
    return {
        "tested_working": {
            "microsoft/phi-2": "Лучшее качество, стабильная",
            "google/flan-t5-base": "Самая надежная, быстрая",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "Очень быстрая",
            "microsoft/DialoGPT-small": "Для чатов, меньше medium",
            "distilgpt2": "Простая и быстрая",
            "gpt2": "Последний fallback"
        },
        "not_recommended": {
            "microsoft/DialoGPT-medium": "Часто не работает в HF API",
            "microsoft/DialoGPT-large": "Слишком большая для HF free tier"
        },
        "current_setup": {
            "your_model": settings.LLM_PRIMARY_MODEL,
            "source": "ENV variable" if os.getenv("LLM_MODEL") else "config default",
            "recommendation": "Попробуйте microsoft/phi-2 или google/flan-t5-base если не работает"
        }
    }

# ====================================
# АВТОДИАГНОСТИКА ПРИ ЗАПУСКЕ
# ====================================

def print_config_summary():
    """Выводит сводку конфигурации при запуске"""
    model_info = settings.get_active_model_info()
    print("\n🤖 LLM Configuration:")
    print(f"   Primary Model: {model_info['primary_model']}")
    print(f"   Source: {model_info['source']}")
    print(f"   Demo Mode: {model_info['demo_mode']}")
    print(f"   HF Token: {'✅ Set' if model_info['hf_token_configured'] else '❌ Not set'}")
    
    if not model_info['hf_token_configured']:
        print("   💡 Tip: Set HF_TOKEN for better rate limits")
    
    if model_info['primary_model'] == "microsoft/DialoGPT-medium":
        print("   ⚠️ Warning: DialoGPT-medium often fails - try microsoft/phi-2")

# Автоматический вызов при импорте (только в dev режиме)
if os.getenv("DEBUG") or os.getenv("SHOW_CONFIG_SUMMARY"):
    print_config_summary()

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
    "validate_llm_config",
    "get_model_recommendations",
    "print_config_summary"
]