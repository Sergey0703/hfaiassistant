# backend/services/huggingface_llm_service.py - ВОССТАНОВЛЕННАЯ ВЕРСИЯ
"""
LLM Service using HuggingFace Transformers with GPTQ quantization
Ваши оригинальные GPTQ модели
"""

import logging
import time
import os
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_used: int
    response_time: float
    success: bool
    error: Optional[str] = None

class HuggingFaceLLMService:
    """LLM Service using HuggingFace Transformers with GPTQ - ВАШИ МОДЕЛИ"""
    
    def __init__(self, model_name: str = "TheBloke/Llama-2-7B-Chat-GPTQ"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.service_type = "huggingface_gptq"
        self.model_loaded = False
        self._initialize_model()
    
    def _initialize_model(self):
        """Инициализирует ВАШИ GPTQ модели"""
        try:
            logger.info(f"🤖 Loading YOUR GPTQ model: {self.model_name}")
            
            # Импортируем библиотеки
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # ВАШИ оригинальные GPTQ модели (в порядке предпочтения)
            your_gptq_models = [
                "TheBloke/Llama-2-7B-Chat-GPTQ",  # Ваша основная модель
                "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ",  # Ваша альтернатива
                "TheBloke/Llama-2-13B-Chat-GPTQ",  # Если нужна более мощная
            ]
            
            for model_name in your_gptq_models:
                try:
                    logger.info(f"🔄 Trying to load YOUR model: {model_name}")
                    
                    # Загружаем токенайзер
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        use_fast=True
                    )
                    
                    # Настройки для ВАШИХ GPTQ моделей
                    model_kwargs = {
                        "torch_dtype": torch.float16,
                        "device_map": "auto",
                        "trust_remote_code": True,
                        "low_cpu_mem_usage": True,
                    }
                    
                    # Пытаемся загрузить с GPTQ квантизацией
                    try:
                        logger.info(f"🔄 Loading {model_name} with GPTQ quantization...")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            **model_kwargs,
                            quantization_config={"load_in_4bit": True}
                        )
                        logger.info(f"✅ Loaded with GPTQ quantization: {model_name}")
                    except Exception as gptq_error:
                        logger.warning(f"⚠️ GPTQ quantization failed for {model_name}: {gptq_error}")
                        logger.info(f"🔄 Trying {model_name} without quantization...")
                        # Fallback без квантизации
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            **model_kwargs
                        )
                        logger.info(f"✅ Loaded without quantization: {model_name}")
                    
                    # Настройка токенайзера
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    self.model_name = model_name
                    self.model_loaded = True
                    logger.info(f"🎉 YOUR model ready: {model_name}")
                    return
                    
                except Exception as e:
                    logger.warning(f"❌ Failed to load YOUR model {model_name}: {e}")
                    continue
            
            # Если ваши модели не загрузились
            raise Exception("Failed to load any of YOUR GPTQ models")
            
        except ImportError as e:
            logger.error(f"❌ Missing dependencies for YOUR models: {e}")
            logger.error("Install: pip install transformers torch auto-gptq accelerate")
            raise
        except Exception as e:
            logger.error(f"❌ YOUR model initialization failed: {e}")
            raise
    
    async def answer_legal_question(self, question: str, context_documents: List[Dict], language: str = "en"):
        """Отвечает на юридический вопрос используя ВАШИ модели"""
        start_time = time.time()
        
        try:
            if not self.model_loaded:
                return LLMResponse(
                    content="Your GPTQ model not loaded. Please check logs for errors.",
                    model=self.model_name,
                    tokens_used=0,
                    response_time=time.time() - start_time,
                    success=False,
                    error="Your model not initialized"
                )
            
            # Формируем промпт для ВАШИХ моделей
            prompt = self._build_legal_prompt(question, context_documents, language)
            
            # Генерируем ответ с ВАШИМИ моделями
            response_text = await self._generate_response(prompt)
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response_text,
                model=self.model_name,
                tokens_used=len(response_text.split()),
                response_time=response_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"❌ Error generating response with YOUR model: {e}")
            return LLMResponse(
                content="I apologize, but I'm experiencing technical difficulties with your GPTQ model. Please try again later.",
                model=self.model_name,
                tokens_used=0,
                response_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    def _build_legal_prompt(self, question: str, context_documents: List[Dict], language: str) -> str:
        """Строит промпт для юридического вопроса - оптимизировано для ВАШИХ моделей"""
        
        if language == "uk":
            system_prompt = """Ви - досвідчений юридичний консультант з expertise в українському та ірландському праві. 
Відповідайте на питання чітко та професійно, базуючись на наданих документах.

Правила:
- Використовуйте тільки інформацію з наданих документів
- Якщо інформації недостатньо, скажіть про це чесно
- Давайте практичні поради з посиланнями на конкретні закони
- Відповідайте українською мовою"""
            
            context_intro = "📚 Юридичні документи для аналізу:"
            answer_intro = f"❓ Питання: {question}\n\n📋 Професійна юридична відповідь:"
        else:
            system_prompt = """You are an experienced legal consultant with expertise in Irish and Ukrainian law. 
Answer questions clearly and professionally based on the provided legal documents.

Rules:
- Use only information from the provided documents
- If information is insufficient, say so honestly
- Provide practical advice with specific legal references
- Be concise but thorough"""
            
            context_intro = "📚 Legal documents for analysis:"
            answer_intro = f"❓ Question: {question}\n\n📋 Professional legal response:"
        
        # Добавляем контекст из документов
        context_parts = []
        for i, doc in enumerate(context_documents[:3]):  # Ограничиваем до 3 документов
            filename = doc.get('filename', f'Document {i+1}')
            content = doc.get('content', '')[:800]  # Увеличиваем лимит для ваших моделей
            context_parts.append(f"📄 {filename}:\n{content}")
        
        context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."
        
        # Собираем финальный промпт
        prompt = f"{system_prompt}\n\n{context_intro}\n{context}\n\n{answer_intro}"
        
        # Ограничиваем общую длину промпта для ваших моделей
        max_length = 3000  # Увеличено для GPTQ моделей
        if len(prompt) > max_length:
            prompt = prompt[:max_length] + "..."
        
        return prompt
    
    async def _generate_response(self, prompt: str, max_new_tokens: int = 800) -> str:
        """Генерирует ответ используя ВАШИ GPTQ модели"""
        try:
            import torch
            
            # Токенизируем промпт
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2000  # Увеличено для ваших моделей
            )
            
            # Генерируем ответ с ВАШИМИ моделями
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    temperature=0.3,  # Консервативная температура для юридических вопросов
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # Декодируем ответ
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Извлекаем только новую часть (ответ)
            response = full_response[len(prompt):].strip()
            
            # Очищаем ответ
            response = self._clean_response(response)
            
            return response if response else "I need more specific information to provide a proper legal analysis."
            
        except Exception as e:
            logger.error(f"❌ Error in response generation with YOUR model: {e}")
            return f"Technical error occurred while generating response with your GPTQ model. Please try with a simpler question."
    
    def _clean_response(self, response: str) -> str:
        """Очищает сгенерированный ответ от ВАШИХ моделей"""
        # Убираем повторения и лишние символы
        lines = response.split('\n')
        cleaned_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if line and line not in seen_lines and len(line) > 10:
                cleaned_lines.append(line)
                seen_lines.add(line)
        
        cleaned = '\n'.join(cleaned_lines)
        
        # Ограничиваем длину для ваших моделей
        if len(cleaned) > 2000:  # Увеличено для GPTQ
            cleaned = cleaned[:2000] + "..."
        
        return cleaned
    
    async def get_service_status(self):
        """Возвращает статус ВАШИХ GPTQ моделей"""
        return {
            "service_type": "your_gptq_models",
            "model_loaded": self.model_loaded,
            "model_name": self.model_name,
            "gptq_quantization": True,
            "your_models": [
                "TheBloke/Llama-2-7B-Chat-GPTQ",
                "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
            ],
            "supported_languages": ["en", "uk"],
            "memory_usage": "~4-6GB with GPTQ quantization",
            "capabilities": [
                "High-quality legal analysis",
                "Multi-language support",
                "GPTQ quantization efficiency",
                "Professional legal responses"
            ],
            "recommendations": [
                "Using YOUR chosen GPTQ models",
                "Optimized for legal consultations",
                "Memory-efficient quantization",
                "Professional response quality"
            ]
        }

def create_llm_service(model_name: str = "TheBloke/Llama-2-7B-Chat-GPTQ"):
    """Создает LLM сервис с ВАШИМИ GPTQ моделями"""
    try:
        return HuggingFaceLLMService(model_name)
    except Exception as e:
        logger.error(f"Failed to create YOUR GPTQ LLM service: {e}")
        # Возвращаем fallback только в крайнем случае
        from app.dependencies import FallbackLLMService
        return FallbackLLMService()