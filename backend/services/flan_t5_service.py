# backend/services/flan_t5_service.py - ИСПРАВЛЕННЫЙ FLAN-T5 СЕРВИС
"""
Минимальный сервис для FLAN-T5 Small модели
Оптимизирован для < 1GB RAM использования
ИСПРАВЛЕНИЕ: Синтаксическая ошибка в import torch на строке 157
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
    """Упрощенный сервис для FLAN-T5 Small"""
    
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
        """Загружает модель FLAN-T5"""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            logger.info("📥 Loading FLAN-T5 model and tokenizer...")
            
            # Загружаем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token if self.hf_token else None
            )
            
            # Загружаем модель
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                token=self.hf_token if self.hf_token else None,
                torch_dtype="auto",
                device_map="auto" if self._has_cuda() else "cpu"
            )
            
            self.ready = True
            logger.info("✅ FLAN-T5 model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load FLAN-T5 model: {e}")
            self.ready = False
    
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
            
            # Строим промпт для T5
            prompt = self._build_t5_prompt(question, context_documents, language)
            
            # Генерируем ответ
            response = await self._generate_with_t5(prompt)
            
            if response.success:
                logger.info(f"✅ Generated response: {len(response.content)} chars")
                return response
            else:
                logger.warning(f"❌ Generation failed: {response.error}")
                return self._generate_fallback_response(question, context_documents, language, start_time)
                
        except Exception as e:
            logger.error(f"❌ Error in answer_legal_question: {e}")
            return self._generate_error_response(question, language, str(e), start_time)
    
    async def _generate_with_t5(self, prompt: str) -> T5Response:
        """Генерирует ответ с FLAN-T5"""
        start_time = time.time()
        
        try:
            # Выполняем в executor для избежания блокировки
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_sync, prompt
            )
            
            response_time = time.time() - start_time
            
            if result:
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
                    error="Empty generation result"
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
    
    def _generate_sync(self, prompt: str) -> str:
        """Синхронная генерация с T5"""
        try:
            # Токенизация
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Генерация
            max_new_tokens = int(os.getenv("LLM_MAX_TOKENS", "150"))
            
            # ИСПРАВЛЕНИЕ: Правильный импорт torch
            import torch
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Декодирование
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Очищаем ответ от промпта
            response = self._clean_t5_response(response, prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Sync generation error: {e}")
            return ""
    
    def _build_t5_prompt(self, question: str, context_documents: List[Dict], language: str) -> str:
        """Строит промпт для FLAN-T5 в text2text формате"""
        
        # Извлекаем контекст
        context = ""
        if context_documents:
            # Берем только первый документ для экономии памяти
            doc = context_documents[0]
            content = doc.get('content', '')
            # Обрезаем контекст для T5 Small
            max_context = int(os.getenv("CONTEXT_TRUNCATE_LENGTH", "300"))
            context = content[:max_context] + "..." if len(content) > max_context else content
        
        # Формируем промпт в зависимости от языка
        if language == "uk":
            if context:
                prompt = f"Контекст: {context}\n\nПитання: {question}\n\nДайте юридичну відповідь на основі контексту:"
            else:
                prompt = f"Питання: {question}\n\nДайте коротку юридичну відповідь:"
        else:
            if context:
                prompt = f"Context: {context}\n\nQuestion: {question}\n\nProvide a legal answer based on the context:"
            else:
                prompt = f"Question: {question}\n\nProvide a brief legal answer:"
        
        return prompt
    
    def _clean_t5_response(self, response: str, prompt: str) -> str:
        """Очищает ответ T5 от артефактов"""
        if not response:
            return "I need more information to provide a proper legal analysis."
        
        # Убираем промпт из ответа если он там есть
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        # Убираем повторяющиеся фразы
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 3:
                cleaned_lines.append(line)
                if len(cleaned_lines) >= 3:  # Ограничиваем длину ответа
                    break
        
        cleaned = ' '.join(cleaned_lines)
        
        # Ограничиваем длину
        max_length = 400
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length] + "..."
        
        return cleaned or "Unable to generate a proper response."
    
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

⚠️ AI відповідь тимчасово недоступна, але знайдено релевантні документи для ручного аналізу."""
        else:
            content = f"""🔍 **Search Results**

**Question:** {question}

📚 {context_info}
📄 **Sources:** {sources}

⚠️ AI response temporarily unavailable, but found relevant documents for manual analysis."""
        
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
            "memory_usage": "~400 MB"
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
            content = f"FLAN-T5 service unavailable. Question: {question}. Found {len(context_documents)} documents."
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