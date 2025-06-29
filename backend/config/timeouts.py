# backend/config/timeouts.py - УПРОЩЕННЫЕ ТАЙМАУТЫ
"""
Оптимизированные таймауты для минимальной RAG системы
Убраны GPTQ специфичные настройки, оставлены только для FLAN-T5
"""

# ====================================
# БАЗОВЫЕ ТАЙМАУТЫ ДЛЯ МИНИМАЛЬНОЙ RAG СИСТЕМЫ
# ====================================

# Основные таймауты (оптимизированы под быстрые модели)
GLOBAL_REQUEST_TIMEOUT = 60      # 1 минута максимум на любой запрос
KEEP_ALIVE_TIMEOUT = 30          # 30 секунд keep-alive
GRACEFUL_TIMEOUT = 60            # 1 минута на graceful shutdown

# FLAN-T5 Small таймауты (быстрая модель)
FLAN_T5_LOADING_TIMEOUT = 120    # 2 минуты на загрузку модели
FLAN_T5_INFERENCE_TIMEOUT = 20   # 20 секунд на генерацию ответа
FLAN_T5_FIRST_LOAD_TIMEOUT = 180 # 3 минуты на первую загрузку

# Sentence Transformers таймауты
EMBEDDING_LOADING_TIMEOUT = 60   # 1 минута на загрузку embeddings
EMBEDDING_INFERENCE_TIMEOUT = 5  # 5 секунд на создание embedding

# ChromaDB таймауты (оптимизированы для быстрой работы)
CHROMADB_SEARCH_TIMEOUT = 5      # 5 секунд на поиск
CHROMADB_ADD_DOC_TIMEOUT = 30    # 30 секунд на добавление документа
CHROMADB_STATS_TIMEOUT = 10      # 10 секунд на статистику

# HTTP запросы таймауты (уменьшены)
HTTP_REQUEST_TIMEOUT = 20        # 20 секунд на HTTP запросы
SCRAPER_TIMEOUT = 30            # 30 секунд на парсинг страницы

# HuggingFace Spaces оптимизации
HF_SPACES_STARTUP_TIMEOUT = 90   # 1.5 минуты на полный старт
HF_SPACES_HEALTH_TIMEOUT = 10    # 10 секунд на health check

# ====================================
# CHAT API ТАЙМАУТЫ (НОВЫЕ)
# ====================================

CHAT_SEARCH_TIMEOUT = 5          # 5 секунд на поиск документов
CHAT_LLM_TIMEOUT = 20           # 20 секунд на FLAN-T5 генерацию
CHAT_TOTAL_TIMEOUT = 30         # 30 секунд общий таймаут чата

# ====================================
# ФУНКЦИИ ДЛЯ ПОЛУЧЕНИЯ ТАЙМАУТОВ
# ====================================

def get_request_timeout(endpoint_type: str) -> int:
    """Возвращает таймаут для конкретного типа endpoint"""
    timeouts = {
        "chat": CHAT_TOTAL_TIMEOUT,
        "search": CHROMADB_SEARCH_TIMEOUT + 10,
        "upload": CHROMADB_ADD_DOC_TIMEOUT + 10,
        "scraper": SCRAPER_TIMEOUT + 10,
        "model_load": FLAN_T5_LOADING_TIMEOUT,
        "health": HF_SPACES_HEALTH_TIMEOUT,
        "embedding": EMBEDDING_INFERENCE_TIMEOUT + 5,
        "default": GLOBAL_REQUEST_TIMEOUT
    }
    return timeouts.get(endpoint_type, timeouts["default"])

def get_model_timeout(model_type: str) -> int:
    """Возвращает таймаут для загрузки модели"""
    model_timeouts = {
        "flan-t5": FLAN_T5_LOADING_TIMEOUT,
        "flan-t5-first": FLAN_T5_FIRST_LOAD_TIMEOUT,
        "embedding": EMBEDDING_LOADING_TIMEOUT,
        "default": FLAN_T5_LOADING_TIMEOUT
    }
    return model_timeouts.get(model_type, model_timeouts["default"])

def get_inference_timeout(model_type: str) -> int:
    """Возвращает таймаут для инференса модели"""
    inference_timeouts = {
        "flan-t5": FLAN_T5_INFERENCE_TIMEOUT,
        "embedding": EMBEDDING_INFERENCE_TIMEOUT,
        "default": FLAN_T5_INFERENCE_TIMEOUT
    }
    return inference_timeouts.get(model_type, inference_timeouts["default"])

def get_timeout_config() -> dict:
    """Возвращает полную конфигурацию таймаутов"""
    return {
        # Основные таймауты
        "global_request_timeout": GLOBAL_REQUEST_TIMEOUT,
        "keep_alive_timeout": KEEP_ALIVE_TIMEOUT,
        "graceful_timeout": GRACEFUL_TIMEOUT,
        
        # Модели
        "flan_t5_loading_timeout": FLAN_T5_LOADING_TIMEOUT,
        "flan_t5_inference_timeout": FLAN_T5_INFERENCE_TIMEOUT,
        "flan_t5_first_load_timeout": FLAN_T5_FIRST_LOAD_TIMEOUT,
        "embedding_loading_timeout": EMBEDDING_LOADING_TIMEOUT,
        "embedding_inference_timeout": EMBEDDING_INFERENCE_TIMEOUT,
        
        # База данных
        "chromadb_search_timeout": CHROMADB_SEARCH_TIMEOUT,
        "chromadb_add_doc_timeout": CHROMADB_ADD_DOC_TIMEOUT,
        "chromadb_stats_timeout": CHROMADB_STATS_TIMEOUT,
        
        # HTTP и скрапинг
        "http_request_timeout": HTTP_REQUEST_TIMEOUT,
        "scraper_timeout": SCRAPER_TIMEOUT,
        
        # HF Spaces
        "hf_spaces_startup_timeout": HF_SPACES_STARTUP_TIMEOUT,
        "hf_spaces_health_timeout": HF_SPACES_HEALTH_TIMEOUT,
        
        # Chat API
        "chat_search_timeout": CHAT_SEARCH_TIMEOUT,
        "chat_llm_timeout": CHAT_LLM_TIMEOUT,
        "chat_total_timeout": CHAT_TOTAL_TIMEOUT
    }

def get_optimized_timeouts_for_memory(memory_mb: int) -> dict:
    """Возвращает оптимизированные таймауты в зависимости от доступной памяти"""
    base_config = get_timeout_config()
    
    if memory_mb < 1024:  # Меньше 1GB - агрессивная оптимизация
        multiplier = 0.7
        base_config.update({
            "chat_total_timeout": int(CHAT_TOTAL_TIMEOUT * multiplier),
            "flan_t5_inference_timeout": int(FLAN_T5_INFERENCE_TIMEOUT * multiplier),
            "chromadb_search_timeout": int(CHROMADB_SEARCH_TIMEOUT * multiplier)
        })
    elif memory_mb < 2048:  # 1-2GB - умеренная оптимизация
        multiplier = 0.8
        base_config.update({
            "chat_total_timeout": int(CHAT_TOTAL_TIMEOUT * multiplier)
        })
    # Для >2GB оставляем стандартные таймауты
    
    return base_config

def is_timeout_aggressive() -> bool:
    """Проверяет используются ли агрессивные таймауты"""
    return CHAT_TOTAL_TIMEOUT <= 30 and FLAN_T5_INFERENCE_TIMEOUT <= 20

def get_timeout_recommendations() -> list:
    """Возвращает рекомендации по таймаутам"""
    recommendations = []
    
    if is_timeout_aggressive():
        recommendations.append("Using aggressive timeouts optimized for <1GB RAM")
        recommendations.append("Consider increasing timeouts if experiencing frequent timeouts")
    
    recommendations.extend([
        f"Chat timeout: {CHAT_TOTAL_TIMEOUT}s (search: {CHAT_SEARCH_TIMEOUT}s + LLM: {CHAT_LLM_TIMEOUT}s)",
        f"Model loading: {FLAN_T5_LOADING_TIMEOUT}s (first time: {FLAN_T5_FIRST_LOAD_TIMEOUT}s)",
        f"Search timeout: {CHROMADB_SEARCH_TIMEOUT}s for vector similarity",
        "Timeouts optimized for FLAN-T5 Small fast inference"
    ])
    
    return recommendations

# ====================================
# КОНСТАНТЫ ДЛЯ BACKWARD COMPATIBILITY
# ====================================

# Для совместимости со старым кодом, если используется
GPTQ_MODEL_LOADING_TIMEOUT = FLAN_T5_LOADING_TIMEOUT  # Теперь указывает на FLAN-T5
GPTQ_INFERENCE_TIMEOUT = FLAN_T5_INFERENCE_TIMEOUT
GPTQ_FIRST_LOAD_TIMEOUT = FLAN_T5_FIRST_LOAD_TIMEOUT

# ====================================
# ЭКСПОРТ
# ====================================

__all__ = [
    # Основные константы
    "GLOBAL_REQUEST_TIMEOUT",
    "KEEP_ALIVE_TIMEOUT", 
    "GRACEFUL_TIMEOUT",
    
    # FLAN-T5 таймауты
    "FLAN_T5_LOADING_TIMEOUT",
    "FLAN_T5_INFERENCE_TIMEOUT", 
    "FLAN_T5_FIRST_LOAD_TIMEOUT",
    
    # Embedding таймауты
    "EMBEDDING_LOADING_TIMEOUT",
    "EMBEDDING_INFERENCE_TIMEOUT",
    
    # ChromaDB таймауты
    "CHROMADB_SEARCH_TIMEOUT",
    "CHROMADB_ADD_DOC_TIMEOUT",
    "CHROMADB_STATS_TIMEOUT",
    
    # Chat API таймауты
    "CHAT_SEARCH_TIMEOUT",
    "CHAT_LLM_TIMEOUT", 
    "CHAT_TOTAL_TIMEOUT",
    
    # Функции
    "get_request_timeout",
    "get_model_timeout",
    "get_inference_timeout", 
    "get_timeout_config",
    "get_optimized_timeouts_for_memory",
    "is_timeout_aggressive",
    "get_timeout_recommendations"
]