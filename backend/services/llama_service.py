# backend/services/llama_service.py - ПРАВИЛЬНЫЙ LLM СЕРВИС
"""
Llama LLM Service using HuggingFace Inference API
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
    """LLM сервис с HuggingFace Inference API"""
    
    def __init__(self):
        self.service_type = "huggingface_inference"
        self.model_name = "meta-llama/Llama-3.1-8B-Instruct"
        self.hf_token = os.getenv("HF_TOKEN")
        self.ready = True
        
        # Проверяем наличие токена
        if not self.hf_token:
            logger.warning("⚠️ HF_TOKEN not set - using public inference (rate limited)")
        else:
            logger.info("✅ HF_TOKEN configured for Llama service")
        
        logger.info(f"🦙 Llama service initialized: {self.model_name}")
    
    async def answer_legal_question(self, question: str, context_documents: List[Dict], language: str = "en"):
        """Отвечает на юридический вопрос"""
        start_time = time.time()
        
        try:
            # Проверяем demo режим
            if os.getenv("LLM_DEMO_MODE", "false").lower() == "true":
                return self._generate_demo_response(question, context_documents, language, start_time)
            
            # Строим промпт
            prompt = self._build_prompt(question, context_documents, language)
            
            # Генерируем ответ через HuggingFace API
            try:
                from huggingface_hub import InferenceClient
                
                client = InferenceClient(
                    model=self.model_name,
                    token=self.hf_token
                )
                
                # Генерируем ответ
                response = client.text_generation(
                    prompt,
                    max_new_tokens=300,
                    temperature=0.3,
                    do_sample=True,
                    return_full_text=False
                )
                
                # Очищаем ответ
                if isinstance(response, str):
                    content = response.strip()
                else:
                    content = str(response).strip()
                
                response_time = time.time() - start_time
                
                return LlamaResponse(
                    content=content,
                    model=self.model_name,
                    tokens_used=len(content.split()),
                    response_time=response_time,
                    success=True
                )
                
            except ImportError:
                logger.error("❌ huggingface_hub not installed")
                return self._generate_error_response(question, language, "Missing huggingface_hub", start_time)
            
            except Exception as e:
                logger.error(f"❌ HuggingFace API error: {e}")
                return self._generate_error_response(question, language, str(e), start_time)
        
        except Exception as e:
            logger.error(f"❌ General error in answer_legal_question: {e}")
            return self._generate_error_response(question, language, str(e), start_time)
    
    def _build_prompt(self, question: str, context_documents: List[Dict], language: str) -> str:
        """Строит промпт для Llama"""
        
        if language == "uk":
            system_prompt = "Ти юридичний консультант. Відповідай коротко і по суті на основі документів."
            context_intro = "Документи:"
            answer_intro = f"Питання: {question}\nВідповідь:"
        else:
            system_prompt = "You are a legal consultant. Answer concisely based on the provided documents."
            context_intro = "Documents:"
            answer_intro = f"Question: {question}\nAnswer:"
        
        # Формируем контекст
        context_parts = []
        for i, doc in enumerate(context_documents[:2]):  # Ограничиваем 2 документами
            filename = doc.get('filename', f'Doc {i+1}')
            content = doc.get('content', '')[:500]  # Ограничиваем длину
            context_parts.append(f"{filename}: {content}")
        
        context = f"\n{context_intro}\n" + "\n".join(context_parts) if context_parts else ""
        
        # Собираем промпт
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>\n"
        
        if context:
            prompt += f"<|start_header_id|>user<|end_header_id|>\n{context}\n\n{answer_intro}<|eot_id|>\n"
        else:
            prompt += f"<|start_header_id|>user<|end_header_id|>\n{answer_intro}<|eot_id|>\n"
        
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
        
        return prompt
    
    def _generate_demo_response(self, question: str, context_documents: List[Dict], language: str, start_time: float):
        """Генерирует demo ответ"""
        if language == "uk":
            content = f"""🤖 **Demo режим активний**

**Ваше питання:** {question}

📚 Знайдено {len(context_documents)} документів у базі знань.

💡 **Demo відповідь:** Це демонстраційний режим Legal Assistant. У робочому режимі тут буде детальна відповідь від Llama-3.1-8B-Instruct на основі ваших юридичних документів.

🔧 **Для активації повної версії:**
1. Отримайте токен на https://huggingface.co/settings/tokens
2. Встановіть змінну HF_TOKEN
3. Встановіть LLM_DEMO_MODE=false"""
        else:
            content = f"""🤖 **Demo Mode Active**

**Your Question:** {question}

📚 Found {len(context_documents)} documents in knowledge base.

💡 **Demo Response:** This is Legal Assistant demo mode. In production mode, you would get detailed answers from Llama-3.1-8B-Instruct based on your legal documents.

🔧 **To activate full version:**
1. Get token at https://huggingface.co/settings/tokens
2. Set HF_TOKEN environment variable
3. Set LLM_DEMO_MODE=false"""
        
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

🔧 **Помилка:** {error}

💡 **Рекомендації:**
• Перевірте налаштування HF_TOKEN
• Спробуйте demo режим: LLM_DEMO_MODE=true
• Перевірте підключення до інтернету"""
        else:
            content = f"""❌ **LLM Service Error**

**Your Question:** {question}

🔧 **Error:** {error}

💡 **Recommendations:**
• Check HF_TOKEN configuration
• Try demo mode: LLM_DEMO_MODE=true
• Check internet connection"""
        
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
        return {
            "service_type": self.service_type,
            "model_name": self.model_name,
            "ready": self.ready,
            "hf_token_configured": bool(self.hf_token),
            "demo_mode": os.getenv("LLM_DEMO_MODE", "false").lower() == "true",
            "recommendations": [
                "Set HF_TOKEN for better rate limits",
                "Use LLM_DEMO_MODE=true for testing",
                "Check https://huggingface.co/settings/tokens for token"
            ]
        }

def create_llama_service():
    """Создает Llama сервис"""
    try:
        return LlamaService()
    except Exception as e:
        logger.error(f"❌ Failed to create Llama service: {e}")
        # Возвращаем fallback
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

⚠️ LLM сервіс недоступний. Встановіть:
1. HF_TOKEN - токен від HuggingFace
2. LLM_DEMO_MODE=true для демо режиму

📚 Знайдено {len(context_documents)} документів у базі."""
            else:
                content = f"""🔄 **System Initializing**

**Your Question:** {question}

⚠️ LLM service unavailable. Please set:
1. HF_TOKEN - HuggingFace token
2. LLM_DEMO_MODE=true for demo mode

📚 Found {len(context_documents)} documents in database."""
            
            return LlamaResponse(
                content=content,
                model="fallback",
                tokens_used=len(content.split()),
                response_time=0.1,
                success=True
            )
        
        async def get_service_status(self):
            return {"service_type": "fallback", "ready": True}
    
    return FallbackService()