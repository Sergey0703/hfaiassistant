# backend/models/responses.py - УПРОЩЕННЫЕ МОДЕЛИ ОТВЕТОВ
"""
Упрощенные Pydantic модели для минимальной RAG системы
Убраны избыточные поля, оставлены только необходимые
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
    """Упрощенная информация о документе"""
    id: Union[str, int] = Field(..., description="ID документа")
    filename: str = Field(..., description="Имя файла")
    category: str = Field(..., description="Категория документа")
    content: str = Field(..., description="Содержимое документа")
    size: int = Field(..., ge=0, description="Размер в байтах")
    added_at: float = Field(..., description="Время добавления")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Метаданные")

class DocumentsResponse(BaseModel):
    """Модель ответа со списком документов"""
    documents: List[DocumentInfo] = Field(..., description="Список документов")
    total: int = Field(..., ge=0, description="Общее количество документов")
    message: Optional[str] = Field(None, description="Дополнительное сообщение")

class DocumentUploadResponse(BaseModel):
    """Модель ответа загрузки документа"""
    message: str = Field(..., description="Сообщение о результате")
    filename: str = Field(..., description="Имя загруженного файла")
    category: str = Field(..., description="Категория документа")
    size: int = Field(..., ge=0, description="Размер файла")

class DocumentDeleteResponse(BaseModel):
    """Модель ответа удаления документа"""
    message: str = Field(..., description="Сообщение о результате")
    deleted_id: Union[str, int] = Field(..., description="ID удаленного документа")

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
# СТАТИСТИКА И СТАТУС
# ====================================

class MinimalStats(BaseModel):
    """Упрощенная статистика системы"""
    total_documents: int = Field(..., ge=0, description="Общее количество документов")
    total_chats: int = Field(..., ge=0, description="Общее количество чатов")
    categories: List[str] = Field(..., description="Список категорий")
    services_status: Dict[str, Any] = Field(..., description="Статус сервисов")
    
    # Информация о моделях
    model_info: Optional[Dict[str, Any]] = Field(None, description="Информация о моделях")
    memory_info: Optional[Dict[str, Any]] = Field(None, description="Информация о памяти")

class HealthCheckResponse(BaseModel):
    """Модель ответа проверки здоровья"""
    status: str = Field(..., description="Статус системы")
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
# СПЕЦИАЛИЗИРОВАННЫЕ ОТВЕТЫ ДЛЯ RAG
# ====================================

class ModelInfoResponse(BaseModel):
    """Информация о моделях RAG системы"""
    llm: Dict[str, Any] = Field(..., description="Информация о LLM")
    embedding: Dict[str, Any] = Field(..., description="Информация об embedding модели")
    vector_db: Dict[str, Any] = Field(..., description="Информация о векторной БД")
    memory_usage: Dict[str, str] = Field(..., description="Потребление памяти")

class SystemStatusResponse(BaseModel):
    """Статус минимальной RAG системы"""
    system: str = Field(default="Minimal RAG", description="Название системы")
    version: str = Field(default="1.0.0", description="Версия")
    status: str = Field(..., description="Статус системы")
    models: Dict[str, str] = Field(..., description="Используемые модели")
    memory: Dict[str, str] = Field(..., description="Информация о памяти")
    features: List[str] = Field(..., description="Доступные функции")
    platform: str = Field(..., description="Платформа развертывания")

# ====================================
# FLAN-T5 СПЕЦИФИЧНЫЕ ОТВЕТЫ
# ====================================

class T5GenerationResponse(BaseModel):
    """Ответ генерации FLAN-T5"""
    content: str = Field(..., description="Сгенерированный контент")
    model: str = Field(default="google/flan-t5-small", description="Используемая модель")
    tokens_used: int = Field(..., ge=0, description="Количество использованных токенов")
    response_time: float = Field(..., ge=0, description="Время генерации")
    success: bool = Field(..., description="Успешность генерации")
    error: Optional[str] = Field(None, description="Ошибка генерации")

class EmbeddingResponse(BaseModel):
    """Ответ генерации эмбеддингов"""
    embeddings: List[List[float]] = Field(..., description="Векторные представления")
    model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Модель")
    dimensions: int = Field(default=384, description="Размерность векторов")
    processing_time: float = Field(..., ge=0, description="Время обработки")

# ====================================
# ДИАГНОСТИКА И МОНИТОРИНГ
# ====================================

class MemoryDiagnostics(BaseModel):
    """Диагностика памяти"""
    total_memory_mb: float = Field(..., description="Общее потребление памяти в MB")
    model_memory_mb: Dict[str, float] = Field(..., description="Память по моделям")
    available_memory_mb: float = Field(..., description="Доступная память")
    memory_efficiency: str = Field(..., description="Эффективность использования памяти")
    recommendations: List[str] = Field(..., description="Рекомендации по оптимизации")

class PerformanceMetrics(BaseModel):
    """Метрики производительности"""
    avg_search_time: float = Field(..., description="Среднее время поиска")
    avg_generation_time: float = Field(..., description="Среднее время генерации")
    avg_total_response_time: float = Field(..., description="Среднее время полного ответа")
    throughput_requests_per_minute: float = Field(..., description="Пропускная способность")
    success_rate: float = Field(..., ge=0, le=100, description="Процент успешных запросов")

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
    
    # История и статистика
    "ChatHistoryItem",
    "ChatHistoryResponse",
    "MinimalStats",
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
    "PerformanceMetrics"
]