# backend/models/responses.py - ПОЛНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ
"""
Полные Pydantic модели ответов для Legal Assistant API
Все модели исправлены под Pydantic v2 с поддержкой protected_namespaces
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

# ====================================
# ОСНОВНЫЕ МОДЕЛИ ОТВЕТОВ
# ====================================

class ChatResponse(BaseModel):
    """Модель ответа чата"""
    response: str = Field(..., description="Ответ FLAN-T5 ассистента")
    sources: Optional[List[str]] = Field(None, description="Источники информации")

class SearchResult(BaseModel):
    """Модель результата поиска"""
    content: str = Field(..., description="Содержимое документа")
    filename: str = Field(..., description="Имя файла")
    document_id: str = Field(..., description="ID документа")
    relevance_score: float = Field(..., ge=0, le=1, description="Оценка релевантности")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Метаданные")

class SearchResponse(BaseModel):
    """Модель ответа поиска"""
    query: str = Field(..., description="Поисковый запрос")
    results: List[SearchResult] = Field(..., description="Результаты поиска")
    total_found: int = Field(..., ge=0, description="Количество найденных результатов")
    search_metadata: Dict[str, Any] = Field(default_factory=dict, description="Метаданные поиска")

# ====================================
# ДОКУМЕНТЫ
# ====================================

class DocumentInfo(BaseModel):
    """Информация о документе"""
    id: Union[str, int] = Field(..., description="ID документа")
    filename: str = Field(..., description="Имя файла")
    category: str = Field(..., description="Категория документа")
    source: str = Field(..., description="Источник документа")
    original_url: str = Field(..., description="Оригинальный URL")
    content: str = Field(..., description="Содержимое документа")
    size: int = Field(..., ge=0, description="Размер в байтах")
    word_count: int = Field(..., ge=0, description="Количество слов")
    chunks_count: int = Field(..., ge=0, description="Количество чанков")
    added_at: float = Field(..., description="Время добавления")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Метаданные")

class DocumentsResponse(BaseModel):
    """Модель ответа со списком документов"""
    documents: List[DocumentInfo] = Field(..., description="Список документов")
    total: int = Field(..., ge=0, description="Общее количество документов")
    message: Optional[str] = Field(None, description="Дополнительное сообщение")
    database_type: Optional[str] = Field(None, description="Тип базы данных")

class DocumentUploadResponse(BaseModel):
    """Модель ответа загрузки документа"""
    message: str = Field(..., description="Сообщение о результате")
    filename: str = Field(..., description="Имя загруженного файла")
    category: str = Field(..., description="Категория документа")
    size: int = Field(..., ge=0, description="Размер файла")
    file_type: Optional[str] = Field(None, description="Тип файла")

class DocumentDeleteResponse(BaseModel):
    """Модель ответа удаления документа"""
    message: str = Field(..., description="Сообщение о результате")
    deleted_id: Union[str, int] = Field(..., description="ID удаленного документа")
    deleted_count: Optional[int] = Field(None, description="Количество удаленных документов")
    remaining_documents: Optional[int] = Field(None, description="Оставшиеся документы")
    database_type: Optional[str] = Field(None, description="Тип базы данных")

# ====================================
# SCRAPING MODELS (ИСПРАВЛЕННЫЕ)
# ====================================

class ScrapeResult(BaseModel):
    """Результат парсинга одного URL"""
    url: str = Field(..., description="URL который парсился")
    title: str = Field(..., description="Заголовок страницы")
    success: bool = Field(..., description="Успешность парсинга")
    content_length: int = Field(..., ge=0, description="Длина контента")
    error: Optional[str] = Field(None, description="Ошибка парсинга")

class ScrapeResponse(BaseModel):
    """Ответ парсинга сайтов"""
    message: str = Field(..., description="Сообщение о результате")
    results: List[ScrapeResult] = Field(..., description="Результаты парсинга")
    summary: Dict[str, Any] = Field(default_factory=dict, description="Сводка по парсингу")

class PredefinedSitesResponse(BaseModel):
    """Ответ со списком предустановленных сайтов"""
    ukraine: List[str] = Field(..., description="Украинские юридические сайты")
    ireland: List[str] = Field(..., description="Ирландские юридические сайты") 
    total: Dict[str, int] = Field(..., description="Статистика по сайтам")

# ====================================
# ИСТОРИЯ ЧАТОВ
# ====================================

class ChatHistoryItem(BaseModel):
    """Элемент истории чата"""
    message: str = Field(..., description="Сообщение пользователя")
    response: str = Field(..., description="Ответ ассистента")
    language: str = Field(..., description="Язык")
    sources: Optional[List[str]] = Field(None, description="Источники")
    timestamp: Optional[float] = Field(None, description="Время сообщения")

class ChatHistoryResponse(BaseModel):
    """Ответ истории чатов"""
    history: List[ChatHistoryItem] = Field(..., description="История чатов")
    total_messages: int = Field(..., ge=0, description="Общее количество сообщений")

# ====================================
# СТАТИСТИКА И СТАТУС (ИСПРАВЛЕННЫЕ)
# ====================================

class AdminStats(BaseModel):
    """Статистика админ панели - исправленная версия"""
    
    # ИСПРАВЛЕНИЕ: Добавляем model_config для отключения protected namespaces
    model_config = {"protected_namespaces": ()}
    
    total_documents: int = Field(..., ge=0, description="Общее количество документов")
    total_chats: int = Field(..., ge=0, description="Общее количество чатов")
    categories: List[str] = Field(..., description="Список категорий")
    services_status: Dict[str, Any] = Field(..., description="Статус сервисов")
    
    # Опциональные поля
    vector_db_info: Optional[Dict[str, Any]] = Field(None, description="Информация о векторной БД")
    vector_db_error: Optional[str] = Field(None, description="Ошибка векторной БД")
    
    # Дополнительная информация
    initialization_summary: Optional[Dict[str, Any]] = Field(None, description="Сводка инициализации")
    system_info: Optional[Dict[str, Any]] = Field(None, description="Системная информация")
    recommendations: List[str] = Field(default_factory=list, description="Рекомендации")

class HealthCheckResponse(BaseModel):
    """Модель ответа проверки здоровья"""
    status: str = Field(..., description="Статус системы")
    timestamp: float = Field(..., description="Время проверки")
    services: Dict[str, Any] = Field(..., description="Статус сервисов")
    models: Optional[Dict[str, str]] = Field(None, description="Информация о моделях")
    memory_target: Optional[str] = Field(None, description="Целевое потребление памяти")

# ====================================
# ОБЩИЕ ОТВЕТЫ
# ====================================

class SuccessResponse(BaseModel):
    """Модель успешного ответа"""
    message: str = Field(..., description="Сообщение об успехе")
    data: Optional[Dict[str, Any]] = Field(None, description="Дополнительные данные")

class ErrorResponse(BaseModel):
    """Модель ответа с ошибкой"""
    detail: str = Field(..., description="Описание ошибки")
    error_code: Optional[str] = Field(None, description="Код ошибки")
    timestamp: datetime = Field(default_factory=datetime.now, description="Время ошибки")

class NotificationResponse(BaseModel):
    """Модель уведомления"""
    message: str = Field(..., description="Текст уведомления")
    type: str = Field(..., pattern="^(success|error|info|warning)$", description="Тип уведомления")

# ====================================
# RAG СПЕЦИФИЧНЫЕ ОТВЕТЫ (ИСПРАВЛЕННЫЕ)
# ====================================

class ModelInfoResponse(BaseModel):
    """Информация о моделях RAG системы"""
    
    # ИСПРАВЛЕНИЕ: Добавляем model_config для отключения protected namespaces  
    model_config = {"protected_namespaces": ()}
    
    llm: Dict[str, Any] = Field(..., description="Информация о LLM")
    embedding: Dict[str, Any] = Field(..., description="Информация об embedding модели")
    vector_db: Dict[str, Any] = Field(..., description="Информация о векторной БД")
    memory_usage: Dict[str, str] = Field(..., description="Потребление памяти")

class SystemStatusResponse(BaseModel):
    """Статус минимальной RAG системы"""
    
    # ИСПРАВЛЕНИЕ: Добавляем model_config
    model_config = {"protected_namespaces": ()}
    
    system: str = Field(default="Minimal RAG", description="Название системы")
    version: str = Field(default="1.0.0", description="Версия")
    status: str = Field(..., description="Статус системы")
    models: Dict[str, str] = Field(..., description="Используемые модели")
    memory: Dict[str, str] = Field(..., description="Информация о памяти")
    features: List[str] = Field(..., description="Доступные функции")
    platform: str = Field(..., description="Платформа развертывания")

# ====================================
# FLAN-T5 СПЕЦИФИЧНЫЕ ОТВЕТЫ (ИСПРАВЛЕННЫЕ)
# ====================================

class T5GenerationResponse(BaseModel):
    """Ответ генерации FLAN-T5"""
    
    # ИСПРАВЛЕНИЕ: Добавляем model_config
    model_config = {"protected_namespaces": ()}
    
    content: str = Field(..., description="Сгенерированный контент")
    model: str = Field(default="google/flan-t5-small", description="Используемая модель")
    tokens_used: int = Field(..., ge=0, description="Количество использованных токенов")
    response_time: float = Field(..., ge=0, description="Время генерации")
    success: bool = Field(..., description="Успешность генерации")
    error: Optional[str] = Field(None, description="Ошибка генерации")

class EmbeddingResponse(BaseModel):
    """Ответ генерации эмбеддингов"""
    
    # ИСПРАВЛЕНИЕ: Добавляем model_config
    model_config = {"protected_namespaces": ()}
    
    embeddings: List[List[float]] = Field(..., description="Векторные представления")
    model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Модель")
    dimensions: int = Field(default=384, description="Размерность векторов")
    processing_time: float = Field(..., ge=0, description="Время обработки")

# ====================================
# ДИАГНОСТИКА И МОНИТОРИНГ (ИСПРАВЛЕННЫЕ)
# ====================================

class MemoryDiagnostics(BaseModel):
    """Диагностика памяти - исправленная версия"""
    
    # ИСПРАВЛЕНИЕ: Добавляем model_config
    model_config = {"protected_namespaces": ()}
    
    total_memory_mb: float = Field(..., description="Общее потребление памяти в MB")
    llm_memory_mb: Dict[str, float] = Field(..., description="Память по моделям")  # ИСПРАВЛЕНО: было model_memory_mb
    available_memory_mb: float = Field(..., description="Доступная память")
    memory_efficiency: str = Field(..., description="Эффективность использования памяти")
    recommendations: List[str] = Field(..., description="Рекомендации по оптимизации")

class PerformanceMetrics(BaseModel):
    """Метрики производительности"""
    
    # ИСПРАВЛЕНИЕ: Добавляем model_config на всякий случай
    model_config = {"protected_namespaces": ()}
    
    avg_search_time: float = Field(..., description="Среднее время поиска")
    avg_generation_time: float = Field(..., description="Среднее время генерации")
    avg_total_response_time: float = Field(..., description="Среднее время полного ответа")
    throughput_requests_per_minute: float = Field(..., description="Пропускная способность")
    success_rate: float = Field(..., ge=0, le=100, description="Процент успешных запросов")

# ====================================
# ДОПОЛНИТЕЛЬНЫЕ ДИАГНОСТИЧЕСКИЕ МОДЕЛИ
# ====================================

class ServiceHealthStatus(BaseModel):
    """Статус здоровья сервиса"""
    
    model_config = {"protected_namespaces": ()}
    
    service_name: str = Field(..., description="Название сервиса")
    status: str = Field(..., description="Статус (healthy/degraded/unhealthy)")
    last_check: float = Field(..., description="Время последней проверки")
    response_time: Optional[float] = Field(None, description="Время ответа")
    error_message: Optional[str] = Field(None, description="Сообщение об ошибке")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Дополнительные данные")

class SystemDiagnostics(BaseModel):
    """Системная диагностика"""
    
    model_config = {"protected_namespaces": ()}
    
    timestamp: float = Field(..., description="Время диагностики")
    overall_status: str = Field(..., description="Общий статус системы")
    services: List[ServiceHealthStatus] = Field(..., description="Статус сервисов")
    memory_info: MemoryDiagnostics = Field(..., description="Информация о памяти")
    performance: PerformanceMetrics = Field(..., description="Метрики производительности")
    recommendations: List[str] = Field(default_factory=list, description="Рекомендации")

# ====================================
# ЭКСПОРТ
# ====================================

# Основные модели для обязательного экспорта
__all__ = [
    # Основные ответы
    "ChatResponse",
    "SearchResponse", 
    "SearchResult",
    
    # Документы
    "DocumentInfo",
    "DocumentsResponse",
    "DocumentUploadResponse",
    "DocumentDeleteResponse",
    
    # Scraping (ДОБАВЛЕНО)
    "ScrapeResult",
    "ScrapeResponse", 
    "PredefinedSitesResponse",
    
    # История и статистика
    "ChatHistoryItem",
    "ChatHistoryResponse",
    "AdminStats",  # ИСПРАВЛЕНО: было MinimalStats
    "HealthCheckResponse",
    
    # Общие ответы
    "SuccessResponse",
    "ErrorResponse",
    "NotificationResponse",
    
    # RAG специфичные
    "ModelInfoResponse",
    "SystemStatusResponse",
    "T5GenerationResponse", 
    "EmbeddingResponse",
    
    # Диагностика
    "MemoryDiagnostics",
    "PerformanceMetrics",
    "ServiceHealthStatus",
    "SystemDiagnostics"
]