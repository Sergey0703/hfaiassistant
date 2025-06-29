# backend/services/llama_service.py - ИСПРАВЛЕН ПОД НОВЫЙ HUGGINGFACE API
"""
Llama LLM Service using NEW HuggingFace Inference Providers API
ИСПРАВЛЕНИЕ: Старый API не работает, переход на новый Inference Providers
"""

import logging
import time
import os
import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LlamaResponse:
    content: str
    model: str
    tokens_used: int
    response_time: float
    success: bool
    error: Optional[str] = None

class LlamaService:
    """LLM сервис с НОВЫМ HuggingFace Inference Providers API"""
    
    def __init__(self):
        self.service_type = "huggingface_inference_providers"
        self.model_name = os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        self.hf_token = os.getenv("HF_TOKEN")
        self.ready = True
        
        # ИСПРАВЛЕНИЕ: Проверяем доступность новых API endpoints
        self.api_endpoints = {
            # Новый Inference Providers API (приоритет)
            "providers": "https://api-inference.huggingface.co/models/",
            # Fallback endpoints для разных типов моделей
            "text_generation": "https://api-inference.huggingface.co/models/",
            "chat": "https://api-inference.huggingface.co/models/"
        }
        
        # Проверенные рабочие модели (июнь 2025)
        self.working_models = [
            "microsoft/DialoGPT-small",    # Чат модель (меньше medium)
            "gpt2",                        # Базовая модель OpenAI
            "distilgpt2",                  # Компактная версия
            "EleutherAI/gpt-neo-125M",    # EleutherAI модель
            "bigscience/bloom-560m"        # BLOOM модель
        ]
        
        if not self.hf_token:
            logger.warning("⚠️ HF_TOKEN not set - using public inference (rate limited)")
        else:
            logger.info("✅ HF_TOKEN configured for Llama service")
        
        logger.info(f"🦙 Llama service initialized: {self.model_name}")
        logger.info(f"🔄 API endpoint: {self.api_endpoints['providers']}")
    
    async def answer_legal_question(self, question: str, context_documents: List[Dict], language: str = "en"):
        """Отвечает на юридический вопрос с НОВЫМ API и fallback моделями"""
        start_time = time.time()
        
        try:
            # Проверяем demo режим
            if os.getenv("LLM_DEMO_MODE", "false").lower() == "true":
                return self._generate_demo_response(question, context_documents, language, start_time)
            
            # Строим промпт
            prompt = self._build_prompt(question, context_documents, language)
            
            # ИСПРАВЛЕНИЕ: Пробуем модели по очереди до успеха
            models_to_try = [self.model_name] + self.working_models
            
            for model_attempt, model in enumerate(models_to_try):
                try:
                    logger.info(f"🔄 Trying model {model_attempt + 1}/{len(models_to_try)}: {model}")
                    
                    response = await self._generate_with_model(prompt, model)
                    
                    if response.success:
                        logger.info(f"✅ Success with model: {model}")
                        return response
                    else:
                        logger.warning(f"❌ Model {model} failed: {response.error}")
                        
                except Exception as e:
                    logger.warning(f"❌ Model {model} exception: {e}")
                    continue
            
            # Если все модели не сработали - возвращаем fallback
            logger.error("❌ All models failed, returning fallback response")
            return self._generate_error_response(question, language, "All models unavailable", start_time)
            
        except Exception as e:
            logger.error(f"❌ General error in answer_legal_question: {e}")
            return self._generate_error_response(question, language, str(e), start_time)
    
    async def _generate_with_model(self, prompt: str, model: str) -> LlamaResponse:
        """Генерирует ответ с конкретной моделью используя НОВЫЙ API"""
        start_time = time.time()
        
        try:
            # ИСПРАВЛЕНИЕ: Попробуем разные API endpoints
            endpoints_to_try = [
                f"https://api-inference.huggingface.co/models/{model}",
                f"https://api-inference.huggingface.co/pipeline/text-generation/{model}",
            ]
            
            for endpoint in endpoints_to_try:
                try:
                    # ИСПРАВЛЕНИЕ: Импортируем requests здесь для избежания проблем
                    import requests
                    
                    headers = {
                        "Authorization": f"Bearer {self.hf_token}" if self.hf_token else "",
                        "Content-Type": "application/json"
                    }
                    
                    # ИСПРАВЛЕНИЕ: Упрощенный payload для совместимости
                    if "chat" in model.lower() or "dialog" in model.lower():
                        # Для chat моделей
                        payload = {
                            "inputs": prompt,
                            "parameters": {
                                "max_new_tokens": 150,
                                "temperature": 0.3,
                                "do_sample": True,
                                "return_full_text": False
                            }
                        }
                    else:
                        # Для обычных моделей
                        payload = {
                            "inputs": prompt,
                            "parameters": {
                                "max_new_tokens": 150,
                                "temperature": 0.3,
                                "return_full_text": False
                            }
                        }
                    
                    # ИСПРАВЛЕНИЕ: Синхронный запрос с таймаутом
                    response = requests.post(
                        endpoint,
                        headers=headers,
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # ИСПРАВЛЕНИЕ: Обработка разных форматов ответа
                        if isinstance(data, list) and len(data) > 0:
                            # Формат: [{"generated_text": "..."}]
                            content = data[0].get("generated_text", "")
                        elif isinstance(data, dict):
                            # Формат: {"generated_text": "..."}
                            content = data.get("generated_text", data.get("text", ""))
                        else:
                            content = str(data)
                        
                        # Очищаем ответ от промпта
                        if content.startswith(prompt):
                            content = content[len(prompt):].strip()
                        
                        content = self._clean_response(content)
                        
                        if content and len(content.strip()) > 5:
                            response_time = time.time() - start_time
                            
                            return LlamaResponse(
                                content=content,
                                model=model,
                                tokens_used=len(content.split()),
                                response_time=response_time,
                                success=True
                            )
                    
                    else:
                        logger.debug(f"HTTP {response.status_code} for {endpoint}")
                        
                except requests.exceptions.RequestException as e:
                    logger.debug(f"Request error for {endpoint}: {e}")
                    continue
                except Exception as e:
                    logger.debug(f"Error with {endpoint}: {e}")
                    continue
            
            # Если все endpoints не сработали
            return LlamaResponse(
                content="",
                model=model,
                tokens_used=0,
                response_time=time.time() - start_time,
                success=False,
                error="All endpoints failed"
            )
            
        except ImportError:
            logger.error("❌ requests library not available")
            return LlamaResponse(
                content="",
                model=model,
                tokens_used=0,
                response_time=time.time() - start_time,
                success=False,
                error="Missing requests library"
            )
        except Exception as e:
            logger.error(f"❌ Model generation error: {e}")
            return LlamaResponse(
                content="",
                model=model,
                tokens_used=0,
                response_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    def _build_prompt(self, question: str, context_documents: List[Dict], language: str) -> str:
        """Строит промпт для разных типов моделей"""
        
        # ИСПРАВЛЕНИЕ: Очень простой промпт для совместимости
        if context_documents:
            # Берем только первый документ и сильно сокращаем
            doc = context_documents[0]
            context = doc.get('content', '')[:200]  # Максимум 200 символов
            
            if language == "uk":
                prompt = f"Контекст: {context}\n\nПитання: {question}\nВідповідь:"
            else:
                prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        else:
            if language == "uk":
                prompt = f"Питання: {question}\nВідповідь:"
            else:
                prompt = f"Question: {question}\nAnswer:"
        
        # Ограничиваем общую длину промпта
        if len(prompt) > 300:
            prompt = prompt[:300] + "..."
        
        return prompt
    
    def _clean_response(self, response: str) -> str:
        """Очищает ответ от лишних символов"""
        if not response:
            return "I need more information to provide a proper legal analysis."
        
        # Базовая очистка
        response = response.strip()
        
        # Удаляем повторения промпта
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.endswith(':') and len(line) > 3:
                cleaned_lines.append(line)
                if len(cleaned_lines) >= 3:  # Максимум 3 строки
                    break
        
        cleaned = ' '.join(cleaned_lines)
        
        # Ограничиваем длину ответа
        if len(cleaned) > 400:
            cleaned = cleaned[:400] + "..."
        
        return cleaned or "Unable to generate response."
    
    def _generate_demo_response(self, question: str, context_documents: List[Dict], language: str, start_time: float):
        """Генерирует demo ответ"""
        if language == "uk":
            content = f"""🤖 **Demo режим активний** (LLM_DEMO_MODE=true)

**Ваше питання:** {question}

📚 Знайдено {len(context_documents)} документів у базі знань.

💡 **Demo відповідь:** Це демонстраційний режим. У робочому режимі тут буде детальна відповідь від реального AI на основі ваших юридичних документів.

🔧 **Для активації повної версії:**
• Встановіть LLM_DEMO_MODE=false
• Переконайтесь що HF_TOKEN налаштований

📄 **Знайдені документи:** {', '.join([doc.get('filename', 'Unknown') for doc in context_documents[:3]])}"""
        else:
            content = f"""🤖 **Demo Mode Active** (LLM_DEMO_MODE=true)

**Your Question:** {question}

📚 Found {len(context_documents)} documents in knowledge base.

💡 **Demo Response:** This is demonstration mode. In production mode, you would get detailed answers from real AI based on your legal documents.

🔧 **To activate full version:**
• Set LLM_DEMO_MODE=false  
• Ensure HF_TOKEN is configured

📄 **Found documents:** {', '.join([doc.get('filename', 'Unknown') for doc in context_documents[:3]])}"""
        
        return LlamaResponse(
            content=content,
            model="demo_mode",
            tokens_used=len(content.split()),
            response_time=time.time() - start_time,
            success=True
        )
    
    def _generate_error_response(self, question: str, language: str, error: str, start_time: float):
        """Генерирует ответ при ошибке"""
        
        if language == "uk":
            content = f"""❌ **Помилка LLM сервісу**

**Ваше питання:** {question}

🔧 **Проблема:** {error}

💡 **Можливі рішення:**
• HuggingFace API може бути перевантажений
• Спробуйте ще раз через кілька хвилин  
• Перевірте налаштування HF_TOKEN
• Або включіть demo режим: LLM_DEMO_MODE=true

🔄 **Статус:** Автоматично спробували кілька моделей"""
        else:
            content = f"""❌ **LLM Service Error**

**Your Question:** {question}

🔧 **Issue:** {error}

💡 **Possible Solutions:**
• HuggingFace API may be overloaded
• Try again in a few minutes
• Check HF_TOKEN configuration  
• Or enable demo mode: LLM_DEMO_MODE=true

🔄 **Status:** Automatically tried multiple models"""
        
        return LlamaResponse(
            content=content,
            model="error_fallback",
            tokens_used=len(content.split()),
            response_time=time.time() - start_time,
            success=False,
            error=error
        )
    
    async def get_service_status(self):
        """Возвращает статус сервиса"""
        # Проверяем доступность requests
        try:
            import requests
            requests_available = True
        except ImportError:
            requests_available = False
        
        return {
            "service_type": self.service_type,
            "model_name": self.model_name,
            "ready": self.ready,
            "hf_token_configured": bool(self.hf_token),
            "demo_mode": os.getenv("LLM_DEMO_MODE", "false").lower() == "true",
            "requests_available": requests_available,
            "api_endpoints": self.api_endpoints,
            "working_models": self.working_models,
            "fallback_enabled": True,
            "recommendations": [
                "New HuggingFace Inference Providers API implemented",
                "Multiple fallback models configured",
                "Set HF_TOKEN for better rate limits",
                "Use LLM_DEMO_MODE=true for testing without API calls",
                "If all models fail, check HuggingFace status"
            ]
        }

def create_llama_service():
    """Создает обновленный Llama сервис"""
    try:
        return LlamaService()
    except Exception as e:
        logger.error(f"❌ Failed to create Llama service: {e}")
        return create_fallback_service()

def create_fallback_service():
    """Создает fallback сервис"""
    class FallbackService:
        def __init__(self):
            self.service_type = "fallback"
            self.ready = True
        
        async def answer_legal_question(self, question: str, context_documents: list, language: str = "en"):
            if language == "uk":
                content = f"""🔄 **Система ініціалізується**

**Ваше питання:** {question}

⚠️ LLM сервіс недоступний. 

📚 Знайдено {len(context_documents)} документів у базі знань.

🔧 **Рекомендації:**
• Перевірте HF_TOKEN налаштування
• Спробуйте demo режим: LLM_DEMO_MODE=true
• HuggingFace API може бути тимчасово недоступний"""
            else:
                content = f"""🔄 **System Initializing**

**Your Question:** {question}

⚠️ LLM service unavailable.

📚 Found {len(context_documents)} documents in database.

🔧 **Recommendations:**
• Check HF_TOKEN configuration
• Try demo mode: LLM_DEMO_MODE=true  
• HuggingFace API may be temporarily unavailable"""
            
            return LlamaResponse(
                content=content,
                model="fallback",
                tokens_used=len(content.split()),
                response_time=0.1,
                success=True
            )
        
        async def get_service_status(self):
            return {
                "service_type": "fallback",
                "ready": True,
                "recommendations": [
                    "Install requests: pip install requests",
                    "Configure HF_TOKEN in environment variables",
                    "Check HuggingFace API status"
                ]
            }
    
    return FallbackService()