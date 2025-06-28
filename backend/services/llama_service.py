# backend/services/llama_service.py - ПОЛНЫЙ ИСПРАВЛЕННЫЙ LLM СЕРВИС
"""
Llama LLM Service using HuggingFace Inference API с retry логикой
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
    """LLM сервис с HuggingFace Inference API и retry логикой"""
    
    def __init__(self):
        self.service_type = "huggingface_inference"
        self.model_name = os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        self.hf_token = os.getenv("HF_TOKEN")
        self.ready = True
        
        # Проверяем наличие токена
        if not self.hf_token:
            logger.warning("⚠️ HF_TOKEN not set - using public inference (rate limited)")
        else:
            logger.info("✅ HF_TOKEN configured for Llama service")
        
        logger.info(f"🦙 Llama service initialized: {self.model_name}")
    
    async def answer_legal_question(self, question: str, context_documents: List[Dict], language: str = "en"):
        """Отвечает на юридический вопрос с retry логикой"""
        start_time = time.time()
        
        try:
            # Проверяем demo режим
            if os.getenv("LLM_DEMO_MODE", "false").lower() == "true":
                return self._generate_demo_response(question, context_documents, language, start_time)
            
            # Строим промпт
            prompt = self._build_prompt(question, context_documents, language)
            
            # Генерируем ответ через HuggingFace API с retry
            try:
                from huggingface_hub import InferenceClient
                
                client = InferenceClient(
                    model=self.model_name,
                    token=self.hf_token
                )
                
                # ИСПРАВЛЕНО: Генерируем ответ с retry логикой
                max_retries = 2
                response = None
                
                for attempt in range(max_retries):
                    try:
                        logger.info(f"🔄 Attempt {attempt + 1}/{max_retries} for LLM generation...")
                        
                        response = client.text_generation(
                            prompt,
                            max_new_tokens=200,  # Уменьшено для скорости
                            temperature=0.3,
                            do_sample=True,
                            return_full_text=False,
                            timeout=60  # 60 секунд таймаут
                        )
                        
                        logger.info(f"✅ LLM generation successful on attempt {attempt + 1}")
                        break  # Успешно - выходим из цикла
                        
                    except Exception as retry_error:
                        if attempt < max_retries - 1:
                            logger.warning(f"❌ Attempt {attempt + 1} failed: {retry_error}, retrying in 5s...")
                            await asyncio.sleep(5)  # Пауза перед повтором
                            continue
                        else:
                            logger.error(f"❌ All {max_retries} attempts failed: {retry_error}")
                            raise retry_error  # Последняя попытка - поднимаем ошибку
                
                # Очищаем ответ
                if isinstance(response, str):
                    content = response.strip()
                else:
                    content = str(response).strip()
                
                # Базовая очистка ответа
                content = self._clean_response(content)
                
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
        
        # Формируем контекст (ограничиваем для скорости)
        context_parts = []
        for i, doc in enumerate(context_documents[:2]):  # Максимум 2 документа
            filename = doc.get('filename', f'Doc {i+1}')
            content = doc.get('content', '')[:400]  # Ограничиваем до 400 символов
            context_parts.append(f"{filename}: {content}")
        
        context = f"\n{context_intro}\n" + "\n".join(context_parts) if context_parts else ""
        
        # Собираем промпт в формате Llama-3.1
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>\n"
        
        if context:
            prompt += f"<|start_header_id|>user<|end_header_id|>\n{context}\n\n{answer_intro}<|eot_id|>\n"
        else:
            prompt += f"<|start_header_id|>user<|end_header_id|>\n{answer_intro}<|eot_id|>\n"
        
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
        
        # Ограничиваем общую длину промпта
        max_prompt_length = 1200
        if len(prompt) > max_prompt_length:
            prompt = prompt[:max_prompt_length] + "..."
        
        return prompt
    
    def _clean_response(self, response: str) -> str:
        """Очищает ответ от лишних символов"""
        if not response:
            return "I need more information to provide a proper legal analysis."
        
        # Удаляем системные токены если попали в ответ
        response = response.replace("<|eot_id|>", "")
        response = response.replace("<|end_header_id|>", "")
        response = response.replace("<|start_header_id|>", "")
        
        # Базовая очистка
        response = response.strip()
        
        # Ограничиваем длину ответа
        if len(response) > 800:
            response = response[:800] + "..."
        
        return response
    
    def _generate_demo_response(self, question: str, context_documents: List[Dict], language: str, start_time: float):
        """Генерирует demo ответ"""
        if language == "uk":
            content = f"""🤖 **Demo режим активний**

**Ваше питання:** {question}

📚 Знайдено {len(context_documents)} документів у базі знань.

💡 **Demo відповідь:** Це демонстраційний режим Legal Assistant. У робочому режимі тут буде детальна відповідь від Llama-3.1-8B-Instruct на основі ваших юридичних документів.

🔧 **Для активації повної версії:**
1. Отримайте токен на https://huggingface.co/settings/tokens
2. Встановіть змінну HF_TOKEN у настройках Space
3. Встановіть LLM_DEMO_MODE=false

📄 **Знайдені документи:** {', '.join([doc.get('filename', 'Unknown') for doc in context_documents[:3]])}"""
        else:
            content = f"""🤖 **Demo Mode Active**

**Your Question:** {question}

📚 Found {len(context_documents)} documents in knowledge base.

💡 **Demo Response:** This is Legal Assistant demo mode. In production mode, you would get detailed answers from Llama-3.1-8B-Instruct based on your legal documents.

🔧 **To activate full version:**
1. Get token at https://huggingface.co/settings/tokens
2. Set HF_TOKEN variable in Space settings
3. Set LLM_DEMO_MODE=false

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
        # Определяем тип ошибки для лучших рекомендаций
        if "504" in error or "timeout" in error.lower():
            error_type = "timeout"
        elif "503" in error or "overloaded" in error.lower():
            error_type = "overloaded"
        elif "401" in error or "token" in error.lower():
            error_type = "auth"
        else:
            error_type = "general"
        
        if language == "uk":
            if error_type == "timeout":
                content = f"""⏰ **Таймаут HuggingFace API**

**Ваше питання:** {question}

🔧 **Проблема:** Сервер HuggingFace перевантажений (504 Gateway Timeout)

💡 **Рекомендації:**
• Спробуйте ще раз через 1-2 хвилини
• Увімкніть demo режим: LLM_DEMO_MODE=true
• HuggingFace API може бути перевантажений у години пік

🔄 **Статус:** Система автоматично повторила запит 2 рази"""
            elif error_type == "auth":
                content = f"""🔑 **Помилка автентифікації**

**Ваше питання:** {question}

🔧 **Проблема:** Перевірте HF_TOKEN

💡 **Рекомендації:**
• Отримайте новий токен: https://huggingface.co/settings/tokens
• Встановіть HF_TOKEN у налаштуваннях Space
• Або увімкніть demo режим: LLM_DEMO_MODE=true"""
            else:
                content = f"""❌ **Помилка LLM сервісу**

**Ваше питання:** {question}

🔧 **Помилка:** {error}

💡 **Рекомендації:**
• Спробуйте demo режим: LLM_DEMO_MODE=true
• Перевірте налаштування HF_TOKEN
• Спробуйте ще раз через кілька хвилин"""
        else:
            if error_type == "timeout":
                content = f"""⏰ **HuggingFace API Timeout**

**Your Question:** {question}

🔧 **Issue:** HuggingFace server overloaded (504 Gateway Timeout)

💡 **Recommendations:**
• Try again in 1-2 minutes
• Enable demo mode: LLM_DEMO_MODE=true
• HuggingFace API may be overloaded during peak hours

🔄 **Status:** System automatically retried request 2 times"""
            elif error_type == "auth":
                content = f"""🔑 **Authentication Error**

**Your Question:** {question}

🔧 **Issue:** Check HF_TOKEN configuration

💡 **Recommendations:**
• Get new token: https://huggingface.co/settings/tokens
• Set HF_TOKEN in Space settings
• Or enable demo mode: LLM_DEMO_MODE=true"""
            else:
                content = f"""❌ **LLM Service Error**

**Your Question:** {question}

🔧 **Error:** {error}

💡 **Recommendations:**
• Try demo mode: LLM_DEMO_MODE=true
• Check HF_TOKEN configuration
• Try again in a few minutes"""
        
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
        # Проверяем доступность HuggingFace
        hf_available = False
        try:
            from huggingface_hub import InferenceClient
            hf_available = True
        except ImportError:
            pass
        
        return {
            "service_type": self.service_type,
            "model_name": self.model_name,
            "ready": self.ready,
            "hf_token_configured": bool(self.hf_token),
            "demo_mode": os.getenv("LLM_DEMO_MODE", "false").lower() == "true",
            "huggingface_hub_available": hf_available,
            "retry_enabled": True,
            "max_retries": 2,
            "timeout": 60,
            "recommendations": [
                "Set HF_TOKEN for better rate limits and priority access",
                "Use LLM_DEMO_MODE=true for testing without API calls",
                "Check https://status.huggingface.co/ for service status",
                "Visit https://huggingface.co/settings/tokens for token management"
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

⚠️ LLM сервіс недоступний. Рекомендації:
1. Встановіть HF_TOKEN - токен від HuggingFace
2. Або увімкніть LLM_DEMO_MODE=true для демо режиму

📚 Знайдено {len(context_documents)} документів у базі знань.

🔧 **Налаштування:**
• HF_TOKEN: отримайте на https://huggingface.co/settings/tokens
• LLM_DEMO_MODE: встановіть у налаштуваннях HuggingFace Space"""
            else:
                content = f"""🔄 **System Initializing**

**Your Question:** {question}

⚠️ LLM service unavailable. Recommendations:
1. Set HF_TOKEN - HuggingFace token
2. Or enable LLM_DEMO_MODE=true for demo mode

📚 Found {len(context_documents)} documents in database.

🔧 **Configuration:**
• HF_TOKEN: get at https://huggingface.co/settings/tokens
• LLM_DEMO_MODE: set in HuggingFace Space settings"""
            
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
                    "Install huggingface_hub: pip install huggingface_hub",
                    "Configure HF_TOKEN in environment variables",
                    "Enable LLM_DEMO_MODE for testing"
                ]
            }
    
    return FallbackService()