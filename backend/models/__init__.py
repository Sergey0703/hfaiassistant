# ====================================
# ФАЙЛ: backend/models/__init__.py (ПОЛНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ)
# ====================================

"""
Models Package - Pydantic модели для Legal Assistant API
Полная версия с исправлениями импортов и protected_namespaces
"""

import logging
from typing import Dict, Any, List, Type, get_type_hints
from pydantic import BaseModel
import inspect
import time

logger = logging.getLogger(__name__)

# Версия пакета моделей
MODELS_VERSION = "2.1.0"

# Реестр всех моделей
class ModelsRegistry:
    """Реестр всех Pydantic моделей в системе"""
    
    def __init__(self):
        self.request_models = {}
        self.response_models = {}
        self.internal_models = {}
        self.loading_errors = []
    
    def register_model(self, model_class: Type[BaseModel], category: str, name: str = None):
        """Регистрирует модель в реестре"""
        try:
            model_name = name or model_class.__name__
            
            model_info = {
                "class": model_class,
                "name": model_name,
                "fields": list(model_class.model_fields.keys()) if hasattr(model_class, 'model_fields') else [],
                "field_count": len(model_class.model_fields) if hasattr(model_class, 'model_fields') else 0,
                "docstring": model_class.__doc__,
                "module": model_class.__module__,
                "schema": None,  # Будем генерировать по запросу
                "has_protected_config": hasattr(model_class, 'model_config') and 
                                      getattr(model_class.model_config, 'protected_namespaces', None) == ()
            }
            
            if category == "request":
                self.request_models[model_name] = model_info
            elif category == "response":
                self.response_models[model_name] = model_info
            elif category == "internal":
                self.internal_models[model_name] = model_info
            
            logger.debug(f"📝 Registered {category} model: {model_name}")
            
        except Exception as e:
            error_msg = f"Failed to register model {model_class.__name__}: {e}"
            self.loading_errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
    
    def get_model(self, name: str, category: str = None) -> Dict[str, Any]:
        """Получает информацию о модели"""
        if category:
            if category == "request":
                return self.request_models.get(name)
            elif category == "response":
                return self.response_models.get(name)
            elif category == "internal":
                return self.internal_models.get(name)
        else:
            # Поиск во всех категориях
            for models_dict in [self.request_models, self.response_models, self.internal_models]:
                if name in models_dict:
                    return models_dict[name]
        return None
    
    def get_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Возвращает все зарегистрированные модели"""
        return {
            "request": self.request_models,
            "response": self.response_models,
            "internal": self.internal_models
        }
    
    def get_models_summary(self) -> Dict[str, Any]:
        """Возвращает сводку по моделям"""
        total_models = (
            len(self.request_models) + 
            len(self.response_models) + 
            len(self.internal_models)
        )
        
        return {
            "total_models": total_models,
            "request_models": len(self.request_models),
            "response_models": len(self.response_models),
            "internal_models": len(self.internal_models),
            "loading_errors": len(self.loading_errors),
            "errors": self.loading_errors
        }
    
    def generate_schema(self, model_name: str, category: str = None) -> Dict[str, Any]:
        """Генерирует JSON схему для модели"""
        model_info = self.get_model(model_name, category)
        if not model_info:
            return None
        
        try:
            model_class = model_info["class"]
            schema = model_class.model_json_schema()  # Pydantic v2
            
            # Кэшируем схему
            model_info["schema"] = schema
            
            return schema
        except Exception as e:
            logger.error(f"Failed to generate schema for {model_name}: {e}")
            return None

# Глобальный реестр моделей
models_registry = ModelsRegistry()

def load_request_models():
    """Загружает модели запросов"""
    try:
        logger.info("📥 Loading request models...")
        
        from models.requests import (
            ChatMessage,
            SearchRequest,
            DocumentUpload,
            URLScrapeRequest,
            BulkScrapeRequest,
            DocumentUpdate,
            PredefinedScrapeRequest,
            ChatHistoryRequest,
            FileUploadForm
        )
        
        # Регистрируем модели запросов
        request_models = [
            ChatMessage,
            SearchRequest,
            DocumentUpload,
            URLScrapeRequest,
            BulkScrapeRequest,
            DocumentUpdate,
            PredefinedScrapeRequest,
            ChatHistoryRequest,
            FileUploadForm
        ]
        
        for model in request_models:
            models_registry.register_model(model, "request")
        
        logger.info(f"✅ Loaded {len(request_models)} request models")
        return True
        
    except ImportError as e:
        error_msg = f"Failed to import request models: {e}"
        models_registry.loading_errors.append(error_msg)
        logger.error(f"❌ {error_msg}")
        return False
    except Exception as e:
        error_msg = f"Error loading request models: {e}"
        models_registry.loading_errors.append(error_msg)
        logger.error(f"❌ {error_msg}")
        return False

def load_response_models():
    """Загружает модели ответов"""
    try:
        logger.info("📤 Loading response models...")
        
        from models.responses import (
            ChatResponse,
            SearchResponse,
            SearchResult,
            DocumentsResponse,
            DocumentInfo,
            DocumentUploadResponse,
            DocumentDeleteResponse,
            ScrapeResponse,              # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ
            ScrapeResult,               # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ
            PredefinedSitesResponse,    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ
            AdminStats,
            ChatHistoryItem,
            ChatHistoryResponse,
            HealthCheckResponse,
            ErrorResponse,
            SuccessResponse,
            NotificationResponse,
            ModelInfoResponse,
            SystemStatusResponse,
            T5GenerationResponse,
            EmbeddingResponse,
            MemoryDiagnostics,
            PerformanceMetrics,
            ServiceHealthStatus,
            SystemDiagnostics
        )
        
        # Регистрируем модели ответов
        response_models = [
            ChatResponse,
            SearchResponse,
            SearchResult,
            DocumentsResponse,
            DocumentInfo,
            DocumentUploadResponse,
            DocumentDeleteResponse,
            ScrapeResponse,              # ДОБАВЛЕНО
            ScrapeResult,               # ДОБАВЛЕНО
            PredefinedSitesResponse,    # ДОБАВЛЕНО
            AdminStats,
            ChatHistoryItem,
            ChatHistoryResponse,
            HealthCheckResponse,
            ErrorResponse,
            SuccessResponse,
            NotificationResponse,
            ModelInfoResponse,
            SystemStatusResponse,
            T5GenerationResponse,
            EmbeddingResponse,
            MemoryDiagnostics,
            PerformanceMetrics,
            ServiceHealthStatus,
            SystemDiagnostics
        ]
        
        for model in response_models:
            models_registry.register_model(model, "response")
        
        logger.info(f"✅ Loaded {len(response_models)} response models")
        return True
        
    except ImportError as e:
        error_msg = f"Failed to import response models: {e}"
        models_registry.loading_errors.append(error_msg)
        logger.error(f"❌ {error_msg}")
        return False
    except Exception as e:
        error_msg = f"Error loading response models: {e}"
        models_registry.loading_errors.append(error_msg)
        logger.error(f"❌ {error_msg}")
        return False

def load_internal_models():
    """Загружает внутренние модели (если есть)"""
    try:
        logger.debug("🔧 Loading internal models...")
        
        # Пока что внутренних моделей нет, но структура готова
        # Здесь можно добавить внутренние модели для кэширования, конфигурации и т.д.
        
        logger.debug("ℹ️ No internal models to load")
        return True
        
    except Exception as e:
        error_msg = f"Error loading internal models: {e}"
        models_registry.loading_errors.append(error_msg)
        logger.error(f"❌ {error_msg}")
        return False

def initialize_models():
    """Инициализирует все модели"""
    logger.info("🚀 Initializing models package...")
    
    # Загружаем все типы моделей
    request_success = load_request_models()
    response_success = load_response_models()
    internal_success = load_internal_models()
    
    # Получаем сводку
    summary = models_registry.get_models_summary()
    
    logger.info(f"📊 Models initialization completed:")
    logger.info(f"   Total models: {summary['total_models']}")
    logger.info(f"   Request models: {summary['request_models']}")
    logger.info(f"   Response models: {summary['response_models']}")
    logger.info(f"   Internal models: {summary['internal_models']}")
    
    if summary['loading_errors']:
        logger.warning(f"⚠️ {summary['loading_errors']} loading errors occurred")
        for error in models_registry.loading_errors:
            logger.warning(f"   - {error}")
    else:
        logger.info("✅ All models loaded successfully")
    
    # Проверяем critical models
    critical_models = ["ChatResponse", "SearchResponse", "AdminStats", "ScrapeResponse"]
    missing_critical = []
    
    for model_name in critical_models:
        if not models_registry.get_model(model_name, "response"):
            missing_critical.append(model_name)
    
    if missing_critical:
        logger.error(f"❌ Missing critical models: {missing_critical}")
        return False
    
    return request_success and response_success and internal_success

def get_models_info() -> Dict[str, Any]:
    """Возвращает информацию о всех моделях"""
    summary = models_registry.get_models_summary()
    all_models = models_registry.get_all_models()
    
    # Формируем детальную информацию
    models_detail = {}
    for category, models in all_models.items():
        models_detail[category] = {}
        for name, info in models.items():
            models_detail[category][name] = {
                "name": info["name"],
                "fields": info["fields"],
                "field_count": info["field_count"],
                "docstring": info["docstring"],
                "module": info["module"],
                "has_schema": info["schema"] is not None,
                "has_protected_config": info.get("has_protected_config", False)
            }
    
    return {
        "version": MODELS_VERSION,
        "summary": summary,
        "models": models_detail,
        "timestamp": time.time(),
        "pydantic_version": "v2",
        "protected_namespaces_fixed": True
    }

def get_model_schema(model_name: str, category: str = None) -> Dict[str, Any]:
    """Возвращает JSON схему для модели"""
    return models_registry.generate_schema(model_name, category)

def validate_model_data(model_name: str, data: Dict[str, Any], category: str = None) -> Dict[str, Any]:
    """Валидирует данные против модели"""
    model_info = models_registry.get_model(model_name, category)
    if not model_info:
        return {
            "valid": False,
            "error": f"Model '{model_name}' not found",
            "data": None
        }
    
    try:
        model_class = model_info["class"]
        validated_instance = model_class(**data)
        
        return {
            "valid": True,
            "error": None,
            "data": validated_instance.model_dump()  # Pydantic v2
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "data": None
        }

def get_model_examples() -> Dict[str, Any]:
    """Возвращает примеры данных для моделей"""
    examples = {}
    
    # Примеры для основных моделей
    model_examples = {
        "ChatMessage": {
            "message": "What are the requirements for Irish citizenship?",
            "language": "en"
        },
        "SearchRequest": {
            "query": "citizenship requirements",
            "category": "ireland_legal",
            "limit": 5
        },
        "DocumentUpload": {
            "filename": "irish_citizenship_guide.txt",
            "content": "This document contains information about Irish citizenship...",
            "category": "ireland_legal"
        },
        "URLScrapeRequest": {
            "url": "https://www.citizensinformation.ie/en/moving-country/irish-citizenship/",
            "category": "ireland_legal"
        },
        "AdminStats": {
            "total_documents": 42,
            "total_chats": 128,
            "categories": ["general", "ireland_legal", "ukraine_legal"],
            "services_status": {
                "document_service": True,
                "llm_service": True,
                "vector_db": True
            }
        },
        "ScrapeResponse": {
            "message": "Successfully scraped 3 URLs",
            "results": [
                {
                    "url": "https://example.com",
                    "title": "Example Page",
                    "success": True,
                    "content_length": 1500,
                    "error": None
                }
            ],
            "summary": {
                "total_processed": 3,
                "successful": 3,
                "failed": 0
            }
        }
    }
    
    for model_name, example_data in model_examples.items():
        # Пытаемся найти модель и валидировать пример
        model_info = models_registry.get_model(model_name)
        if model_info:
            validation_result = validate_model_data(model_name, example_data)
            examples[model_name] = {
                "example": example_data,
                "valid": validation_result["valid"],
                "error": validation_result.get("error"),
                "category": "response" if model_name in ["AdminStats", "ScrapeResponse"] else "request"
            }
    
    return examples

def diagnose_models() -> Dict[str, Any]:
    """Диагностика состояния моделей"""
    logger.info("🔍 Running models package diagnostics...")
    
    diagnostics = {
        "timestamp": time.time(),
        "status": "unknown",
        "summary": {},
        "model_validation": {},
        "schema_generation": {},
        "protected_namespaces_check": {},
        "issues": [],
        "recommendations": []
    }
    
    try:
        # Получаем сводку
        summary = models_registry.get_models_summary()
        diagnostics["summary"] = summary
        
        # Тестируем валидацию ключевых моделей
        critical_models = ["ChatMessage", "SearchRequest", "AdminStats", "ScrapeResponse"]
        for model_name in critical_models:
            model_info = models_registry.get_model(model_name)
            if model_info:
                try:
                    model_class = model_info["class"]
                    
                    # Получаем информацию о полях
                    fields_info = {}
                    if hasattr(model_class, 'model_fields'):
                        fields_info = {
                            "total_fields": len(model_class.model_fields),
                            "required_fields": [
                                name for name, field in model_class.model_fields.items()
                                if field.is_required()
                            ]
                        }
                    
                    diagnostics["model_validation"][model_name] = {
                        "available": True,
                        "fields_info": fields_info,
                        "has_protected_config": model_info.get("has_protected_config", False)
                    }
                    
                except Exception as e:
                    diagnostics["model_validation"][model_name] = {
                        "available": False,
                        "error": str(e)
                    }
                    diagnostics["issues"].append(f"Model {model_name} validation failed: {e}")
            else:
                diagnostics["issues"].append(f"Critical model {model_name} not found")
        
        # Тестируем генерацию схем
        for model_name in critical_models:
            try:
                schema = get_model_schema(model_name)
                diagnostics["schema_generation"][model_name] = {
                    "success": schema is not None,
                    "properties_count": len(schema.get("properties", {})) if schema else 0
                }
            except Exception as e:
                diagnostics["schema_generation"][model_name] = {
                    "success": False,
                    "error": str(e)
                }
                diagnostics["issues"].append(f"Schema generation for {model_name} failed: {e}")
        
        # Проверяем protected_namespaces конфигурацию
        models_with_protected_config = []
        for category, models in models_registry.get_all_models().items():
            for name, info in models.items():
                if info.get("has_protected_config", False):
                    models_with_protected_config.append(name)
        
        diagnostics["protected_namespaces_check"] = {
            "models_with_config": models_with_protected_config,
            "total_protected": len(models_with_protected_config),
            "expected_models": ["AdminStats", "ModelInfoResponse", "T5GenerationResponse", "MemoryDiagnostics"]
        }
        
        # Определяем общий статус
        if not diagnostics["issues"] and summary["loading_errors"] == 0:
            diagnostics["status"] = "healthy"
            diagnostics["recommendations"].append("All models package functioning correctly")
        elif len(diagnostics["issues"]) < 3 and summary["loading_errors"] < 2:
            diagnostics["status"] = "warning"
            diagnostics["recommendations"].append("Minor issues detected, most functionality available")
        else:
            diagnostics["status"] = "error"
            diagnostics["recommendations"].append("Multiple issues detected, check model definitions")
        
        # Добавляем рекомендации
        if summary["loading_errors"] > 0:
            diagnostics["recommendations"].append("Check model import statements and dependencies")
        
        if any("validation" in issue.lower() for issue in diagnostics["issues"]):
            diagnostics["recommendations"].append("Review model field definitions and requirements")
        
        if len(models_with_protected_config) < 4:
            diagnostics["recommendations"].append("Some models may still have protected_namespaces warnings")
        
        logger.info(f"🏥 Models diagnostics completed: {diagnostics['status']}")
        
        return diagnostics
        
    except Exception as e:
        logger.error(f"Models diagnostics failed: {e}")
        diagnostics.update({
            "status": "error",
            "error": str(e),
            "issues": [f"Diagnostics failure: {e}"]
        })
        return diagnostics

# Автоматическая инициализация при импорте
try:
    initialization_success = initialize_models()
    if not initialization_success:
        logger.warning("⚠️ Models package initialized with some issues")
except Exception as e:
    logger.error(f"❌ Models package initialization failed: {e}")

# Экспорт основных компонентов
__all__ = [
    # Метаданные
    'MODELS_VERSION',
    
    # Классы
    'ModelsRegistry',
    'models_registry',
    
    # Функции инициализации
    'initialize_models',
    'load_request_models',
    'load_response_models', 
    'load_internal_models',
    
    # Информационные функции
    'get_models_info',
    'get_model_schema',
    'validate_model_data',
    'get_model_examples',
    'diagnose_models',
]

# Условный экспорт моделей (если они загружены успешно)
try:
    from models.requests import *
    from models.responses import *
    
    # Добавляем имена моделей в __all__
    __all__.extend([
        # Request models
        "ChatMessage", "SearchRequest", "DocumentUpload", "URLScrapeRequest",
        "BulkScrapeRequest", "DocumentUpdate", "PredefinedScrapeRequest",
        "ChatHistoryRequest", "FileUploadForm",
        
        # Response models  
        "ChatResponse", "SearchResponse", "SearchResult", "DocumentsResponse",
        "DocumentInfo", "DocumentUploadResponse", "DocumentDeleteResponse",
        "ScrapeResponse", "ScrapeResult", "PredefinedSitesResponse",   # КРИТИЧЕСКИЕ МОДЕЛИ
        "AdminStats", "ChatHistoryItem", "ChatHistoryResponse", 
        "HealthCheckResponse", "ErrorResponse", "SuccessResponse", 
        "NotificationResponse", "ModelInfoResponse", "SystemStatusResponse",
        "T5GenerationResponse", "EmbeddingResponse", "MemoryDiagnostics",
        "PerformanceMetrics", "ServiceHealthStatus", "SystemDiagnostics"
    ])
    
    logger.debug("✅ Models exported successfully")
    
except ImportError as e:
    logger.warning(f"⚠️ Some models could not be exported: {e}")

logger.debug(f"📦 Models package loaded with {len(__all__)} exported items")