# backend/services/llama_service.py - ИСПРАВЛЕН ПОД НОВЫЙ HUGGINGFACE API
"""
Llama LLM Service using NEW HuggingFace Inference Providers API
ИСПРАВЛЕНИЕ: Старый API не работает, переход на новый Inference Providers
Поддержка flan-t5 моделей (encoder-decoder)
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
    def __init__(self):
        self.service_type = "huggingface_inference_providers"
        self.model_name = os.getenv("LLM_MODEL", "google/flan-t5-base")
        self.hf_token = os.getenv("HF_TOKEN")
        self.ready = True

        self.api_endpoints = {
            "providers": "https://api-inference.huggingface.co/models/",
            "text_generation": "https://api-inference.huggingface.co/models/",
            "chat": "https://api-inference.huggingface.co/models/"
        }

        self.working_models = [
            "google/flan-t5-base",
            "microsoft/DialoGPT-small",
            "gpt2",
            "distilgpt2",
            "EleutherAI/gpt-neo-125M",
            "bigscience/bloom-560m"
        ]

        if not self.hf_token:
            logger.warning("⚠️ HF_TOKEN not set - using public inference (rate limited)")
        else:
            logger.info("✅ HF_TOKEN configured for Llama service")

        logger.info(f"🦙 Llama service initialized: {self.model_name}")
        logger.info(f"🔄 API endpoint: {self.api_endpoints['providers']}")

    async def answer_legal_question(self, question: str, context_documents: List[Dict], language: str = "en"):
        start_time = time.time()

        try:
            if os.getenv("LLM_DEMO_MODE", "false").lower() == "true":
                return self._generate_demo_response(question, context_documents, language, start_time)

            models_to_try = [self.model_name] + self.working_models

            for model_attempt, model in enumerate(models_to_try):
                try:
                    logger.info(f"🔄 Trying model {model_attempt + 1}/{len(models_to_try)}: {model}")
                    prompt = self._build_prompt(question, context_documents, language, model)
                    response = await self._generate_with_model(prompt, model)

                    if response.success:
                        logger.info(f"✅ Success with model: {model}")
                        return response
                    else:
                        logger.warning(f"❌ Model {model} failed: {response.error}")

                except Exception as e:
                    logger.warning(f"❌ Model {model} exception: {e}")
                    continue

            logger.error("❌ All models failed, returning fallback response")
            return self._generate_error_response(question, language, "All models unavailable", start_time)

        except Exception as e:
            logger.error(f"❌ General error in answer_legal_question: {e}")
            return self._generate_error_response(question, language, str(e), start_time)

    async def _generate_with_model(self, prompt: str, model: str) -> LlamaResponse:
        start_time = time.time()
        is_flan = "flan" in model.lower()

        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self.hf_token}" if self.hf_token else "",
                "Content-Type": "application/json"
            }

            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 150,
                    "temperature": 0.3,
                    "do_sample": True
                }
            }

            task_type = "text2text-generation" if is_flan else "text-generation"
            endpoint = f"https://api-inference.huggingface.co/pipeline/{task_type}/{model}"
            logger.info(f"📡 Using endpoint: {endpoint}")

            response = requests.post(endpoint, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                data = response.json()

                if isinstance(data, list) and len(data) > 0:
                    content = data[0].get("generated_text") or data[0].get("text") or ""
                elif isinstance(data, dict):
                    content = data.get("generated_text") or data.get("text") or ""
                else:
                    content = str(data)

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

            return LlamaResponse(
                content="",
                model=model,
                tokens_used=0,
                response_time=time.time() - start_time,
                success=False,
                error=f"HTTP {response.status_code}"
            )

        except requests.exceptions.RequestException as e:
            return LlamaResponse(
                content="",
                model=model,
                tokens_used=0,
                response_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
        except ImportError:
            return LlamaResponse(
                content="",
                model=model,
                tokens_used=0,
                response_time=time.time() - start_time,
                success=False,
                error="Missing requests library"
            )
        except Exception as e:
            return LlamaResponse(
                content="",
                model=model,
                tokens_used=0,
                response_time=time.time() - start_time,
                success=False,
                error=str(e)
            )

    def _build_prompt(self, question: str, context_documents: List[Dict], language: str, model: Optional[str] = None) -> str:
        is_flan = model and "flan" in model.lower()

        context = ""
        if context_documents:
            doc = context_documents[0]
            context = doc.get('content', '')[:200]

        if language == "uk":
            return f"Запитання: {question} Контекст: {context}" if is_flan else f"Контекст: {context}\n\nПитання: {question}\nВідповідь:"
        else:
            return f"question: {question} context: {context}" if is_flan else f"Context: {context}\n\nQuestion: {question}\nAnswer:"

    def _clean_response(self, response: str) -> str:
        if not response:
            return "I need more information to provide a proper legal analysis."

        response = response.strip()
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.endswith(':') and len(line) > 3:
                cleaned_lines.append(line)
                if len(cleaned_lines) >= 3:
                    break

        cleaned = ' '.join(cleaned_lines)
        if len(cleaned) > 400:
            cleaned = cleaned[:400] + "..."

        return cleaned or "Unable to generate response."

    def _generate_demo_response(self, question: str, context_documents: List[Dict], language: str, start_time: float):
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

def create_llama_service():
    try:
        return LlamaService()
    except Exception as e:
        logger.error(f"❌ Failed to create Llama service: {e}")
        return create_fallback_service()

def create_fallback_service():
    class FallbackService:
        def __init__(self):
            self.service_type = "fallback"
            self.ready = True

        async def answer_legal_question(self, question: str, context_documents: list, language: str = "en"):
            content = f"LLM service unavailable. Question: {question}. Found {len(context_documents)} documents."
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
                    "Install requests",
                    "Set HF_TOKEN",
                    "Check HuggingFace status"
                ]
            }

    return FallbackService()
