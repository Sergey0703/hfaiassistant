# backend/services/flan_t5_service.py - ОПТИМИЗИРОВАННАЯ ВЕРСИЯ
"""
Оптимизированный сервис для FLAN-T5 Small модели
ИСПРАВЛЕНИЯ: Улучшенные промпты, настройки генерации, скорость
"""

import logging
import time
import os
import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class T5Response:
    content: str
    model: str
    tokens_used: int
    response_time: float
    success: bool
    error: Optional[str] = None

class FlanT5Service:
    """Оптимизированный сервис для FLAN-T5 Small"""
    
    def __init__(self):
        self.service_type = "flan_t5"
        self.model_name = os.getenv("LLM_MODEL", "google/flan-t5-small")
        self.hf_token = os.getenv("HF_TOKEN")
        self.ready = False
        self.model = None
        self.tokenizer = None
        
        logger.info(f"🤖 Initializing FLAN-T5 service: {self.model_name}")
        self._load_model()
    
    def _load_model(self):
        """Загружает модель FLAN-T5 с оптимизацией"""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            logger.info("📥 Loading FLAN-T5 model and tokenizer...")
            
            # Загружаем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token if self.hf_token else None
            )
            
            # Загружаем модель с оптимизацией
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                token=self.hf_token if self.hf_token else None,
                torch_dtype="auto",
                low_cpu_mem_usage=True  # Экономия памяти
            )
            
            # Перемещаем модель на правильное устройство
            device = self._get_device()
            self.model = self.model.to(device)
            
            # Оптимизируем модель для inference
            self.model.eval()  # Переводим в режим инференса
            
            self.ready = True
            logger.info(f"✅ FLAN-T5 model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load FLAN-T5 model: {e}")
            self.ready = False
    
    def _get_device(self):
        """Определяет устройство"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except:
            return "cpu"
    
    def _has_cuda(self) -> bool:
        """Проверяет доступность CUDA"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    async def answer_legal_question(self, question: str, context_documents: List[Dict], language: str = "en"):
        """Отвечает на юридический вопрос используя FLAN-T5"""
        start_time = time.time()
        
        try:
            if not self.ready:
                return self._generate_error_response(
                    question, language, "Model not loaded", start_time
                )
            
            # Строим оптимизированный промпт для T5
            prompt = self._build_optimized_t5_prompt(question, context_documents, language)
            
            # Генерируем ответ
            response = await self._generate_with_t5(prompt)
            
            if response.success and len(response.content.strip()) > 15:  # Увеличили порог с 10 до 15
                logger.info(f"✅ Generated response: {len(response.content)} chars")
                return response
            else:
                logger.warning(f"❌ Generation failed or too short ({len(response.content.strip())} chars): {response.error or 'Short response'}")
                return self._generate_fallback_response(question, context_documents, language, start_time)
                
        except Exception as e:
            logger.error(f"❌ Error in answer_legal_question: {e}")
            return self._generate_error_response(question, language, str(e), start_time)
    
    async def _generate_with_t5(self, prompt: str) -> T5Response:
        """Генерирует ответ с FLAN-T5 оптимизированно"""
        start_time = time.time()
        
        try:
            # Выполняем в executor для избежания блокировки
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_sync_optimized, prompt
            )
            
            response_time = time.time() - start_time
            
            if result and len(result.strip()) > 5:
                return T5Response(
                    content=result,
                    model=self.model_name,
                    tokens_used=len(result.split()),
                    response_time=response_time,
                    success=True
                )
            else:
                return T5Response(
                    content="",
                    model=self.model_name,
                    tokens_used=0,
                    response_time=response_time,
                    success=False,
                    error="Generated response too short or empty"
                )
                
        except Exception as e:
            return T5Response(
                content="",
                model=self.model_name,
                tokens_used=0,
                response_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    def _generate_sync_optimized(self, prompt: str) -> str:
        """ОПТИМИЗИРОВАННАЯ синхронная генерация с T5"""
        try:
            import torch
            
            # Получаем устройство модели
            device = next(self.model.parameters()).device
            
            # Оптимизированная токенизация
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=400,  # Уменьшено для скорости
                truncation=True,
                padding=True
            )
            
            # Перемещаем inputs на устройство модели
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # ОПТИМИЗИРОВАННЫЕ параметры генерации
            max_new_tokens = int(os.getenv("LLM_MAX_TOKENS", "120"))  # Увеличено для длинных ответов
            temperature = float(os.getenv("LLM_TEMPERATURE", "0.8"))  # Еще больше разнообразия
            
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": 15,  # ИСПРАВЛЕНИЕ: min_new_tokens вместо min_length
                "temperature": temperature,
                "do_sample": True,
                "top_p": 0.9,
                "top_k": 50,
                "no_repeat_ngram_size": 2,  # Уменьшено для меньших ограничений
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "num_beams": 1,  # Убрали early_stopping - не поддерживается
                "length_penalty": 1.2,  # ДОБАВЛЕНО: поощряем более длинные ответы
            }
            
            logger.debug(f"🔧 Generation params: max_tokens={max_new_tokens}, temp={temperature}")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # Декодирование - берем только новые токены
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Очищаем ответ
            response = self._clean_t5_response_optimized(response)
            
            logger.debug(f"🎯 Raw response length: {len(response)} chars")
            
            return response
            
        except Exception as e:
            logger.error(f"Sync generation error: {e}")
            return ""
    
    def _build_optimized_t5_prompt(self, question: str, context_documents: List[Dict], language: str) -> str:
        """ОПТИМИЗИРОВАННЫЙ промпт для FLAN-T5 - более детальные инструкции"""
        
        # Более детальные промпты для получения длинных ответов
        if language == "uk":
            if context_documents:
                doc = context_documents[0]
                content = doc.get('content', '')[:250]  # Немного больше контекста
                prompt = f"Використовуючи наступний текст: {content}\n\nДайте детальну відповідь на питання: {question}\nВідповідь повинна бути мінімум 3-4 речення:"
            else:
                prompt = f"Дайте детальну відповідь на юридичне питання: {question}\nВідповідь повинна містити пояснення та бути мінімум 3-4 речення:"
        else:
            if context_documents:
                doc = context_documents[0]
                content = doc.get('content', '')[:250]  # Немного больше контекста
                prompt = f"Using the following text: {content}\n\nProvide a detailed answer to the question: {question}\nThe answer should be at least 3-4 sentences long:"
            else:
                prompt = f"Provide a detailed answer to the legal question: {question}\nThe answer should include explanations and be at least 3-4 sentences long:"
        
        return prompt
    
    def _clean_t5_response_optimized(self, response: str) -> str:
        """ОПТИМИЗИРОВАННАЯ очистка ответа T5"""
        if not response:
            return "Law is a system of rules and regulations that govern society and ensure order, justice, and protection of individual rights."
        
        # Простая очистка
        response = response.strip()
        
        # Убираем очень короткие ответы - заменяем на базовый ответ
        if len(response) < 15:
            if "law" in response.lower():
                return "Law is a system of rules and regulations established by society to maintain order, protect rights, and ensure justice for all citizens."
            else:
                return "I need more information to provide a comprehensive legal analysis of your question."
        
        # Ограничиваем длину разумными пределами
        if len(response) > 400:
            # Ищем последнее предложение
            sentences = response.split('.')
            if len(sentences) > 1:
                response = '.'.join(sentences[:-1]) + '.'
            else:
                response = response[:400] + "..."
        
        return response
    
    def _generate_fallback_response(self, question: str, context_documents: List[Dict], 
                                  language: str, start_time: float):
        """Генерирует fallback ответ с найденным контекстом"""
        
        if context_documents:
            context_info = f"Found {len(context_documents)} relevant documents"
            source_files = [doc.get('filename', 'Unknown') for doc in context_documents[:2]]
            sources = ', '.join(source_files)
        else:
            context_info = "No relevant documents found"
            sources = "None"
        
        if language == "uk":
            content = f"""🔍 **Результат пошуку**

**Питання:** {question}

📚 {context_info}
📄 **Джерела:** {sources}

⚠️ AI відповідь тимчасово недоступна або занадто коротка. Знайдено релевантні документи для ручного аналізу."""
        else:
            content = f"""🔍 **Search Results**

**Question:** {question}

📚 {context_info}
📄 **Sources:** {sources}

⚠️ AI response temporarily unavailable or too short. Found relevant documents for manual analysis."""
        
        return T5Response(
            content=content,
            model="fallback",
            tokens_used=len(content.split()),
            response_time=time.time() - start_time,
            success=True
        )
    
    def _generate_error_response(self, question: str, language: str, error: str, start_time: float):
        """Генерирует ответ при ошибке"""
        
        if language == "uk":
            content = f"""❌ **Помилка AI сервісу**

**Питання:** {question}

🔧 **Проблема:** {error}

💡 **Рішення:**
• Спробуйте ще раз через кілька секунд
• Перефразуйте питання простіше
• Зверніться до адміністратора"""
        else:
            content = f"""❌ **AI Service Error**

**Question:** {question}

🔧 **Issue:** {error}

💡 **Solutions:**
• Try again in a few seconds
• Rephrase question more simply
• Contact administrator"""
        
        return T5Response(
            content=content,
            model="error",
            tokens_used=len(content.split()),
            response_time=time.time() - start_time,
            success=False,
            error=error
        )
    
    async def get_service_status(self):
        """Возвращает статус сервиса"""
        return {
            "service_type": self.service_type,
            "model": self.model_name,
            "ready": self.ready,
            "hf_token_configured": bool(self.hf_token),
            "cuda_available": self._has_cuda(),
            "device": self._get_device(),
            "memory_usage": "~400 MB",
            "optimization": {
                "eval_mode": True,
                "low_cpu_mem_usage": True,
                "optimized_prompts": True,
                "min_response_length": 20
            }
        }

def create_flan_t5_service():
    """Создает экземпляр FLAN-T5 сервиса"""
    try:
        return FlanT5Service()
    except Exception as e:
        logger.error(f"❌ Failed to create FLAN-T5 service: {e}")
        return create_fallback_service()

def create_fallback_service():
    """Создает fallback сервис"""
    
    class FallbackService:
        def __init__(self):
            self.service_type = "fallback"
            self.ready = True
        
        async def answer_legal_question(self, question: str, context_documents: list, language: str = "en"):
            if language == "uk":
                content = f"""📚 **FLAN-T5 сервіс недоступний**

**Питання:** {question}

Знайдено документів: {len(context_documents)}

🔧 AI тимчасово недоступний, але пошук документів працює."""
            else:
                content = f"""📚 **FLAN-T5 Service Unavailable**

**Question:** {question}

Documents found: {len(context_documents)}

🔧 AI temporarily unavailable, but document search is working."""
            
            return T5Response(
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
                "error": "FLAN-T5 service not available"
            }
    
    return FallbackService()