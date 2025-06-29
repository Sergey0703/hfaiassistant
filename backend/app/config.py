# backend/app/config.py - МИНИМАЛЬНАЯ КОНФИГУРАЦИЯ RAG СИСТЕМЫ
"""
Упрощенная конфигурация для минимальной RAG системы
Target: ~920 MB RAM, HuggingFace Spaces совместимость
"""

import os
from typing import List

class Settings:
    """Минимальные настройки для RAG системы"""
    
    # ====================================
    # ОСНОВНЫЕ НАСТРОЙКИ
    # ====================================
    API_V1_PREFIX: str = "/api"
    PROJECT_NAME: str = "Minimal Legal RAG"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Minimal RAG system with FLAN-T5 and sentence-transformers"
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    
    # ====================================
    # LLM НАСТРОЙКИ - FLAN-T5 SMALL
    # ====================================
    
    @property
    def LLM_MODEL(self) -> str:
        return os.getenv("LLM_MODEL", "google/flan-t5-small")
    
    @property
    def LLM_MAX_TOKENS(self) -> int:
        return int(os.getenv("LLM_MAX_TOKENS", "150"))  # Меньше для T5
    
    @property
    def LLM_TEMPERATURE(self) -> float:
        return float(os.getenv("LLM_TEMPERATURE", "0.3"))
    
    @property
    def LLM_TIMEOUT(self) -> int:
        return int(os.getenv("LLM_TIMEOUT", "20"))  # Быстрая модель
    
    @property
    def HF_TOKEN(self) -> str:
        return os.getenv("HF_TOKEN", "")
    
    # ====================================
    # EMBEDDING НАСТРОЙКИ
    # ====================================
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 500  # Меньше чанки
    CHUNK_OVERLAP: int = 50
    
    # ====================================
    # ПОИСК И КОНТЕКСТ
    # ====================================
    DEFAULT_SEARCH_LIMIT: int = 3
    MAX_SEARCH_LIMIT: int = 5
    MAX_CONTEXT_DOCUMENTS: int = 2
    CONTEXT_TRUNCATE_LENGTH: int = 300  # Короткий контекст для T5
    
    # ====================================
    # БАЗА ДАННЫХ
    # ====================================
    @property
    def USE_CHROMADB(self) -> bool:
        return os.getenv("USE_CHROMADB", "true").lower() == "true"
    
    @property 
    def CHROMADB_PATH(self) -> str:
        return os.getenv("CHROMADB_PATH", "./chromadb_data")
    
    # ====================================
    # ФАЙЛЫ И ЗАГРУЗКИ
    # ====================================
    MAX_FILE_SIZE: int = 5 * 1024 * 1024  # 5MB
    ALLOWED_FILE_TYPES: List[str] = [".txt", ".pdf", ".md"]
    
    # ====================================
    # ВЕБ-СКРАПИНГ
    # ====================================
    SCRAPING_DELAY: float = 1.0
    SCRAPING_TIMEOUT: int = 10
    MAX_URLS_PER_REQUEST: int = 5
    
    # ====================================
    # ЛОГИРОВАНИЕ
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

# Создаём глобальный экземпляр
settings = Settings()

# ====================================
# КОНСТАНТЫ
# ====================================

# API метаданные
API_METADATA = {
    "title": settings.PROJECT_NAME,
    "version": settings.VERSION,
    "description": settings.DESCRIPTION,
}

# API теги
API_TAGS = [
    {"name": "User Chat", "description": "Chat with legal assistant"},
    {"name": "User Search", "description": "Search legal documents"},
    {"name": "Admin Documents", "description": "Document management"},
    {"name": "System", "description": "System health and info"}
]

# Категории документов
DOCUMENT_CATEGORIES = [
    "general",
    "legislation", 
    "jurisprudence",
    "government",
    "civil_rights"
]

# Предустановленные сайты (упрощенно)
UKRAINE_LEGAL_URLS = [
    "https://zakon.rada.gov.ua/laws/main"
]

IRELAND_LEGAL_URLS = [
    "https://www.irishstatutebook.ie/"
]

# ====================================
# ФУНКЦИИ КОНФИГУРАЦИИ
# ====================================

def get_llm_config() -> dict:
    """Возвращает конфигурацию LLM"""
    return {
        "model": settings.LLM_MODEL,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "temperature": settings.LLM_TEMPERATURE,
        "timeout": settings.LLM_TIMEOUT,
        "hf_token_configured": bool(settings.HF_TOKEN),
        "model_type": "text2text-generation"
    }

def get_database_config() -> dict:
    """Возвращает конфигурацию базы данных"""
    return {
        "use_chromadb": settings.USE_CHROMADB,
        "chromadb_path": settings.CHROMADB_PATH,
        "embedding_model": settings.EMBEDDING_MODEL,
        "chunk_size": settings.CHUNK_SIZE
    }

def validate_config() -> dict:
    """Простая валидация конфигурации"""
    issues = []
    warnings = []
    
    # Проверяем основные параметры
    if settings.LLM_MAX_TOKENS < 10:
        issues.append("LLM_MAX_TOKENS too low")
    
    if not (0 <= settings.LLM_TEMPERATURE <= 1):
        issues.append("LLM_TEMPERATURE must be 0-1")
    
    if not settings.HF_TOKEN:
        warnings.append("HF_TOKEN not set - may have rate limits")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "model_info": get_llm_config(),
        "memory_estimate": "~920 MB total"
    }

def get_environment_info() -> dict:
    """Информация об окружении"""
    return {
        "huggingface_spaces": settings.HUGGINGFACE_SPACES,
        "model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "memory_target": "< 1GB RAM",
        "chromadb_enabled": settings.USE_CHROMADB
    }

# ====================================
# ЭКСПОРТ
# ====================================

__all__ = [
    "Settings", "settings",
    "API_METADATA", "API_TAGS", "DOCUMENT_CATEGORIES",
    "UKRAINE_LEGAL_URLS", "IRELAND_LEGAL_URLS",
    "get_llm_config", "get_database_config", 
    "validate_config", "get_environment_info"
]