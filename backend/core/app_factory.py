# backend/core/app_factory.py
"""
Фабрика для создания FastAPI приложения с полной конфигурацией
"""

import asyncio
import time
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.timeouts import (
    GLOBAL_REQUEST_TIMEOUT, GPTQ_MODEL_LOADING_TIMEOUT, 
    GPTQ_FIRST_LOAD_TIMEOUT, HF_SPACES_HEALTH_TIMEOUT
)
from config.react_spa import setup_react_spa, setup_api_info_endpoint, is_react_built
from middleware.timeout_middleware import setup_all_timeout_middleware

logger = logging.getLogger(__name__)

def create_app_for_deployment():
    """Создает приложение для deployment с исправлениями POST 404 И РАСШИРЕННЫМИ ТАЙМАУТАМИ + React SPA"""
    try:
        print("🚀 Creating FastAPI application with comprehensive timeout controls and React SPA...")
        
        # Создаем приложение с новой архитектурой
        from app import create_app
        app = create_app()
        
        if app is None:
            raise RuntimeError("Failed to create FastAPI application")
        
        # ====================================
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: CORS ПЕРВЫМ
        # ====================================
        
        # ИСПРАВЛЕНИЕ: Добавляем CORS middleware ПЕРВЫМ, до всех остальных
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
            allow_headers=["*"],
            allow_credentials=True,
            expose_headers=["*"],
            max_age=3600
        )
        
        # ====================================
        # НАСТРОЙКА MIDDLEWARE
        # ====================================
        
        # Теперь настраиваем остальные middleware ПОСЛЕ CORS
        try:
            from app.middleware import setup_middleware
            setup_middleware(app)
            logger.info("✅ Application middleware configured after CORS")
        except Exception as e:
            error_msg = f"Application middleware setup failed: {e}"
            logger.warning(f"⚠️ {error_msg}")
            # Middleware не критичен, продолжаем без него
        
        # ====================================
        # TIMEOUT MIDDLEWARE - COMPREHENSIVE
        # ====================================
        
        setup_all_timeout_middleware(app)
        
        # ====================================
        # REACT SPA CONFIGURATION
        # ====================================
        
        # Настраиваем React SPA
        react_configured = setup_react_spa(app)
        
        # Если React не настроен, добавляем API info endpoint
        if not react_configured:
            setup_api_info_endpoint(app)
        
        # ====================================
        # СПЕЦИАЛЬНЫЕ ENDPOINTS ДЛЯ HF SPACES
        # ====================================
        
        _setup_hf_spaces_endpoints(app)
        
        print("✅ FastAPI application created successfully")
        print("✅ CORS configured FIRST (POST fix applied)")
        print("✅ HuggingFace Spaces optimizations applied")
        print("✅ Special HF Spaces endpoints added")
        print("✅ Comprehensive timeout system enabled")
        print(f"✅ GPTQ model support with {GPTQ_MODEL_LOADING_TIMEOUT}s loading timeout")
        print(f"✅ Global request timeout: {GLOBAL_REQUEST_TIMEOUT}s")
        print(f"✅ React SPA: {'Configured' if react_configured else 'Build required'}")
        
        return app
        
    except Exception as e:
        logger.error(f"Deployment initialization failed: {e}")
        
        # Создаем улучшенное fallback приложение
        return _create_fallback_app(e)

def _setup_hf_spaces_endpoints(app: FastAPI):
    """Настраивает специальные endpoints для HuggingFace Spaces"""
    
    @app.get("/hf-spaces-health")
    async def hf_spaces_health():
        """Специальный health check для HF Spaces с timeout информацией и GPTQ статусом"""
        from app.dependencies import get_services_status
        
        try:
            services = await asyncio.wait_for(
                get_services_status(),
                timeout=HF_SPACES_HEALTH_TIMEOUT
            )
            
            # Определяем общий статус с учетом GPTQ
            overall_status = "healthy"
            issues = []
            recommendations = []
            
            # Проверяем критические сервисы
            if not services.get("document_service_available", False):
                overall_status = "degraded"
                issues.append("Document service initializing")
                recommendations.append("Document search will be available shortly")
            
            # Специальная проверка GPTQ модели
            llm_available = services.get("llm_available", False)
            if not llm_available:
                if overall_status == "healthy":
                    overall_status = "degraded"
                issues.append("GPTQ model loading (TheBloke/Llama-2-7B-Chat-GPTQ)")
                recommendations.append(f"AI responses will activate when model loads (timeout: {GPTQ_MODEL_LOADING_TIMEOUT}s)")
            
            # Проверяем React SPA
            react_status = is_react_built()
            
            response_data = {
                "status": overall_status,
                "platform": "HuggingFace Spaces",
                "api_version": "2.0.0",
                "react_spa": {
                    "enabled": react_status,
                    "status": "ready" if react_status else "build_required"
                },
                "timeout_configuration": {
                    "global_request_timeout": GLOBAL_REQUEST_TIMEOUT,
                    "keep_alive_timeout": 65,
                    "gptq_loading_timeout": GPTQ_MODEL_LOADING_TIMEOUT,
                    "gptq_inference_timeout": 120,
                    "chromadb_search_timeout": 30,
                    "http_request_timeout": 45,
                    "health_check_timeout": HF_SPACES_HEALTH_TIMEOUT
                },
                "gptq_model": {
                    "name": "TheBloke/Llama-2-7B-Chat-GPTQ",
                    "status": "ready" if llm_available else "loading",
                    "supported_languages": ["English", "Ukrainian"],
                    "optimization": "4-bit GPTQ quantization",
                    "loading_timeout": f"{GPTQ_MODEL_LOADING_TIMEOUT}s",
                    "inference_timeout": "120s"
                },
                "services": services,
                "endpoints": {
                    "react_app": "/" if react_status else "/api-info",
                    "chat": "/api/user/chat",
                    "search": "/api/user/search", 
                    "docs": "/docs",
                    "admin": "/api/admin"
                },
                "features": {
                    "lazy_loading": True,
                    "gptq_support": True,
                    "ukrainian_language": True,
                    "vector_search": services.get("chromadb_enabled", False),
                    "demo_mode": services.get("demo_mode", True),
                    "memory_optimized": True,
                    "timeout_protected": True,
                    "hf_spaces_optimized": True,
                    "react_spa": react_status,
                    "cors_enabled": True
                },
                "cors_fix_applied": True,
                "post_endpoints_working": True,
                "timeout_middleware_active": True
            }
            
            if issues:
                response_data["issues"] = issues
            if recommendations:
                response_data["recommendations"] = recommendations
            
            return response_data
            
        except asyncio.TimeoutError:
            return {
                "status": "timeout",
                "error": f"Health check timeout after {HF_SPACES_HEALTH_TIMEOUT}s",
                "platform": "HuggingFace Spaces",
                "react_spa": {"enabled": is_react_built()},
                "timeout_configuration": {
                    "health_check_timeout": HF_SPACES_HEALTH_TIMEOUT,
                    "global_request_timeout": GLOBAL_REQUEST_TIMEOUT,
                    "gptq_loading_timeout": GPTQ_MODEL_LOADING_TIMEOUT
                },
                "recommendations": [
                    "Services may be initializing",
                    "GPTQ model may be loading in background",
                    "Try again in a few moments",
                    "Check /startup-progress for detailed status"
                ]
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "platform": "HuggingFace Spaces",
                "react_spa": {"enabled": is_react_built()},
                "timeout_configuration": {
                    "global_request_timeout": GLOBAL_REQUEST_TIMEOUT
                },
                "recommendations": [
                    "Check server logs for detailed errors",
                    "Services may still be initializing",
                    "Try again in a few moments"
                ]
            }
    
    @app.get("/model-status")
    async def comprehensive_model_status():
        """Расширенный статус GPTQ модели с диагностикой и таймаутами"""
        from app.dependencies import get_llm_service
        
        try:
            llm_service = await asyncio.wait_for(
                get_llm_service(),
                timeout=10.0  # 10 секунд на получение сервиса
            )
            
            status = await asyncio.wait_for(
                llm_service.get_service_status(),
                timeout=15.0  # 15 секунд на статус
            )
            
            # Проверяем статус загрузки модели
            model_ready = status.get("model_loaded", False)
            loading_error = status.get("loading_error")
            
            model_info = {
                "name": "TheBloke/Llama-2-7B-Chat-GPTQ",
                "type": "GPTQ Quantized Llama-2",
                "size": "~4GB quantized (14GB unquantized)",
                "languages": ["English", "Ukrainian", "Multilingual"],
                "status": "ready" if model_ready else ("error" if loading_error else "loading"),
                "service_type": status.get("service_type", "unknown"),
                "loading_error": loading_error,
                "timeout_protected": True
            }
            
            # Добавляем диагностическую информацию с таймаутами
            diagnostics = {
                "platform": "HuggingFace Spaces",
                "memory_optimization": True,
                "quantization": "4-bit GPTQ",
                "react_spa_enabled": is_react_built(),
                "timeout_limits": {
                    "model_loading": f"{GPTQ_MODEL_LOADING_TIMEOUT}s (8 minutes)",
                    "first_load": f"{GPTQ_FIRST_LOAD_TIMEOUT}s (10 minutes)",
                    "inference": "120s (2 minutes)",
                    "chat_total": f"{GLOBAL_REQUEST_TIMEOUT}s (10 minutes)"
                },
                "dependencies": {
                    "transformers": status.get("transformers_version", "unknown"),
                    "torch": status.get("torch_available", False),
                    "auto_gptq": status.get("auto_gptq_available", False),
                    "cuda": status.get("cuda_available", False)
                },
                "memory_management": {
                    "hf_spaces_limit": "16GB RAM",
                    "model_quantization": "4-bit GPTQ",
                    "offloading_enabled": True,
                    "cpu_fallback": True
                }
            }
            
            return {
                "model_info": model_info,
                "status": status,
                "diagnostics": diagnostics,
                "react_spa": {"enabled": is_react_built()},
                "performance": {
                    "quantization": "4-bit GPTQ",
                    "inference_speed": "Optimized for HF Spaces",
                    "memory_efficient": True,
                    "quality": "High-quality legal analysis",
                    "token_limit": "400 tokens per response",
                    "timeout_protected": True,
                    "expected_loading_time": f"{GPTQ_MODEL_LOADING_TIMEOUT//60}-{GPTQ_FIRST_LOAD_TIMEOUT//60} minutes"
                }
            }
            
        except asyncio.TimeoutError:
            return {
                "model_info": {
                    "name": "TheBloke/Llama-2-7B-Chat-GPTQ", 
                    "status": "timeout"
                },
                "error": "Model status check timeout",
                "react_spa": {"enabled": is_react_built()},
                "timeout_info": {
                    "status_check_timeout": "15s",
                    "loading_timeout": f"{GPTQ_MODEL_LOADING_TIMEOUT}s",
                    "first_load_timeout": f"{GPTQ_FIRST_LOAD_TIMEOUT}s"
                },
                "recommendations": [
                    "Model may still be loading in background",
                    "Check /startup-progress for loading status",
                    f"GPTQ loading can take up to {GPTQ_FIRST_LOAD_TIMEOUT//60} minutes",
                    "Try again in a few moments"
                ]
            }
        except Exception as e:
            return {
                "model_info": {
                    "name": "TheBloke/Llama-2-7B-Chat-GPTQ", 
                    "status": "error"
                },
                "error": str(e),
                "react_spa": {"enabled": is_react_built()},
                "timeout_info": {
                    "loading_timeout": f"{GPTQ_MODEL_LOADING_TIMEOUT}s",
                    "global_timeout": f"{GLOBAL_REQUEST_TIMEOUT}s"
                },
                "recommendations": [
                    "Check HuggingFace Transformers installation",
                    "Verify auto-gptq dependencies",
                    "Model may still be downloading",
                    f"Loading timeout set to {GPTQ_MODEL_LOADING_TIMEOUT//60} minutes",
                    "Try again in a few moments"
                ]
            }
    
    @app.get("/startup-progress")
    async def startup_progress():
        """Endpoint для отслеживания прогресса инициализации с GPTQ информацией"""
        from app.dependencies import get_services_status, get_background_tasks_status
        
        try:
            services = get_services_status()
            background_tasks = get_background_tasks_status()
            
            init_status = services.get("initialization_status", {})
            
            # Подсчитываем прогресс
            total_services = 3  # document, scraper, llm
            completed_services = sum(init_status.values())
            progress_percent = int((completed_services / total_services) * 100)
            
            # Определяем текущую активность с учетом GPTQ
            current_activity = "Starting up..."
            estimated_time = "2-5 minutes"
            
            if not init_status.get("document_service", False):
                current_activity = "Initializing document service (ChromaDB)..."
                estimated_time = "30s"
            elif not init_status.get("llm_service", False):
                current_activity = "Loading GPTQ model (TheBloke/Llama-2-7B-Chat-GPTQ)..."
                estimated_time = f"{GPTQ_MODEL_LOADING_TIMEOUT//60}-{GPTQ_FIRST_LOAD_TIMEOUT//60} minutes"
            elif not init_status.get("scraper_service", False):
                current_activity = "Initializing web scraper..."
                estimated_time = "30s"
            else:
                current_activity = "All services ready!"
                estimated_time = "Complete"
            
            component_status = {
                "document_service": {
                    "status": "ready" if init_status.get("document_service") else "loading",
                    "description": "ChromaDB vector search",
                    "ready": init_status.get("document_service", False),
                    "timeout": "30s"
                },
                "llm_service": {
                    "status": "ready" if services.get("llm_available") else "loading", 
                    "description": "GPTQ Model (TheBloke/Llama-2-7B-Chat-GPTQ)",
                    "ready": services.get("llm_available", False),
                    "timeout": f"{GPTQ_MODEL_LOADING_TIMEOUT}s",
                    "first_load_timeout": f"{GPTQ_FIRST_LOAD_TIMEOUT}s",
                    "quantization": "4-bit GPTQ"
                },
                "scraper_service": {
                    "status": "ready" if init_status.get("scraper_service") else "loading",
                    "description": "Legal site scraper",
                    "ready": init_status.get("scraper_service", False),
                    "timeout": "45s"
                },
                "react_spa": {
                    "status": "ready" if is_react_built() else "build_required",
                    "description": "React frontend application",
                    "ready": is_react_built(),
                    "build_required": not is_react_built()
                }
            }
            
            return {
                "overall_progress": progress_percent,
                "current_activity": current_activity,
                "estimated_time_remaining": estimated_time,
                "components": component_status,
                "services_ready": completed_services,
                "total_services": total_services,
                "ready_for_requests": progress_percent >= 33,  # Можно использовать с частичной готовностью
                "platform": "HuggingFace Spaces",
                "lazy_loading": True,
                "cors_fix_applied": True,
                "background_tasks": background_tasks,
                "react_spa": {
                    "enabled": is_react_built(),
                    "endpoint": "/" if is_react_built() else "/api-info"
                },
                "timeout_protection": {
                    "enabled": True,
                    "global_timeout": f"{GLOBAL_REQUEST_TIMEOUT}s",
                    "specialized_timeouts": True
                },
                "gptq_info": {
                    "model": "TheBloke/Llama-2-7B-Chat-GPTQ",
                    "expected_load_time": f"{GPTQ_MODEL_LOADING_TIMEOUT//60}-{GPTQ_FIRST_LOAD_TIMEOUT//60} minutes",
                    "optimization": "4-bit quantization for HF Spaces 16GB limit"
                }
            }
            
        except Exception as e:
            return {
                "overall_progress": 0,
                "current_activity": "Error checking startup progress",
                "error": str(e),
                "platform": "HuggingFace Spaces",
                "cors_fix_applied": True,
                "timeout_protection": True,
                "react_spa": {"enabled": is_react_built()}
            }

def _create_fallback_app(error: Exception) -> FastAPI:
    """Создает улучшенное fallback приложение"""
    
    fallback_app = FastAPI(title="Legal Assistant API - Recovery Mode", version="2.0.0")
    
    # КРИТИЧЕСКИ ВАЖНО: CORS даже в fallback приложении
    fallback_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"]
    )
    
    @fallback_app.get("/")
    async def error_root():
        return {
            "error": "Application failed to initialize properly",
            "details": str(error),
            "platform": "HuggingFace Spaces",
            "cors_fix_applied": True,
            "react_spa": {"enabled": is_react_built()},
            "timeout_info": {
                "global_timeout": GLOBAL_REQUEST_TIMEOUT,
                "gptq_loading_timeout": GPTQ_MODEL_LOADING_TIMEOUT,
                "keep_alive": 65
            },
            "suggestions": [
                "Check that all dependencies are installed",
                "Verify model files are accessible", 
                "Check available memory and storage",
                "Review server logs for detailed errors",
                f"GPTQ model loading may take up to {GPTQ_FIRST_LOAD_TIMEOUT//60} minutes",
                "Try refreshing in a few minutes",
                f"React SPA: {'Available' if is_react_built() else 'Needs build'}"
            ],
            "fallback_mode": True
        }
    
    return fallback_app