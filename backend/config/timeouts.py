# backend/config/timeouts.py
"""
Конфигурация таймаутов для HuggingFace Spaces и Legal Assistant API
"""

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

# ====================================
# ФУНКЦИИ ДЛЯ ПОЛУЧЕНИЯ ТАЙМАУТОВ
# ====================================

def get_request_timeout(endpoint_type: str) -> int:
    """Возвращает таймаут для конкретного типа endpoint"""
    timeouts = {
        "chat": GLOBAL_REQUEST_TIMEOUT,
        "search": CHROMADB_SEARCH_TIMEOUT + 30,
        "upload": CHROMADB_ADD_DOC_TIMEOUT + 30,
        "scraper": SCRAPER_TIMEOUT + 30,
        "model_status": GPTQ_MODEL_LOADING_TIMEOUT,
        "health": HF_SPACES_HEALTH_TIMEOUT,
        "default": GLOBAL_REQUEST_TIMEOUT
    }
    return timeouts.get(endpoint_type, timeouts["default"])

def get_timeout_config() -> dict:
    """Возвращает полную конфигурацию таймаутов"""
    return {
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
    }