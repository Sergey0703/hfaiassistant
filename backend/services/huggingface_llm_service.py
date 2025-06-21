# backend/services/huggingface_llm_service.py
"""
LLM Service using HuggingFace Transformers with GPTQ quantization
Заменяет Ollama для работы на HuggingFace Spaces
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
    """LLM Service using HuggingFace Transformers with GPTQ"""
    
    def __init__(self, model_name: str = "TheBloke/Llama-2-7B-Chat-GPTQ"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.service_type = "huggingface_gptq"
        self.model_loaded = False
        self._initialize_model()
    
    def _initialize_model(self):
        """Инициализирует GPTQ модель"""
        try:
            # Пытаемся загрузить модель
            logger.info(f"🤖 Loading GPTQ model: {self.model_name}")
            
            # Импортируем библиотеки
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # Определяем доступные GPTQ модели (в порядке предпочтения)
            gptq_models = [
                "TheBloke/Llama-2-7B-Chat-GPTQ",  # Стабильная модель
                "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ",  # Быстрая альтернатива
                "microsoft/DialoGPT-medium",  # Fallback модель
            ]
            
            for model_name in gptq_models:
                try:
                    logger.info(f"🔄 Trying to load: {model_name}")
                    
                    # Загружаем токенайзер
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        use_fast=True
                    )
                    
                    # Настройки для GPTQ
                    model_kwargs = {
                        "torch_dtype": torch.float16,
                        "device_map": "auto",
                        "trust_remote_code": True,
                        "low_cpu_mem_usage": True,
                    }
                    
                    # Пытаемся загрузить с GPTQ
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            **model_kwargs,
                            quantization_config={"load_in_4bit": True}
                        )
                        logger.info(f"✅ Loaded with 4-bit quantization: {model_name}")
                    except:
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
                    logger.info(f"🎉 Model ready: {model_name}")
                    return
                    
                except Exception as e:
                    logger.warning(f"❌ Failed to load {model_name}: {e}")
                    continue
            
            # Если ничего не загрузилось
            raise Exception("Failed to load any GPTQ model")
            
        except ImportError as e:
            logger.error(f"❌ Missing dependencies: {e}")
            logger.error("Install: pip install transformers torch auto-gptq")
            raise
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            raise
    
    async def answer_legal_question(self, question: str, context_documents: List[Dict], language: str = "en"):
        """Отвечает на юридический вопрос"""
        start_time = time.time()
        
        try:
            if not self.model_loaded:
                return LLMResponse(
                    content="Model not loaded. Please check logs for errors.",
                    model=self.model_name,
                    tokens_used=0,
                    response_time=time.time() - start_time,
                    success=False,
                    error="Model not initialized"
                )
            
            # Формируем промпт
            prompt = self._build_legal_prompt(question, context_documents, language)
            
            # Генерируем ответ
            response_text = await self._generate_response(prompt)
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response_text,
                model=self.model_name,
                tokens_used=len(response_text.split()),  # Приблизительно
                response_time=response_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return LLMResponse(
                content="I apologize, but I'm experiencing technical difficulties. Please try again later.",
                model=self.model_name,
                tokens_used=0,
                response_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    def _build_legal_prompt(self, question: str, context_documents: List[Dict], language: str) -> str:
        """Строит промпт для юридического вопроса"""
        
        if language == "uk":
            system_prompt = """Ви - досвідчений юридичний консультант. Відповідайте на питання чітко та професійно, базуючись на наданих документах.

Правила:
- Використовуйте тільки інформацію з наданих документів
- Якщо інформації недостатньо, скажіть про це чесно
- Давайте практичні поради
- Відповідайте українською мовою"""
            
            context_intro = "📚 Документи для аналізу:"
            answer_intro = f"❓ Питання: {question}\n\n📋 Відповідь:"
        else:
            system_prompt = """You are an experienced legal consultant. Answer questions clearly and professionally based on the provided documents.

Rules:
- Use only information from the provided documents
- If information is insufficient, say so honestly
- Provide practical advice
- Be concise but thorough"""
            
            context_intro = "📚 Documents for analysis:"
            answer_intro = f"❓ Question: {question}\n\n📋 Answer:"
        
        # Добавляем контекст из документов
        context_parts = []
        for i, doc in enumerate(context_documents[:3]):  # Ограничиваем до 3 документов
            filename = doc.get('filename', f'Document {i+1}')
            content = doc.get('content', '')[:600]  # Ограничиваем длину
            context_parts.append(f"📄 {filename}:\n{content}")
        
        context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."
        
        # Собираем финальный промпт
        prompt = f"{system_prompt}\n\n{context_intro}\n{context}\n\n{answer_intro}"
        
        # Ограничиваем общую длину промпта
        max_length = 2000
        if len(prompt) > max_length:
            prompt = prompt[:max_length] + "..."
        
        return prompt
    
    async def _generate_response(self, prompt: str, max_new_tokens: int = 500) -> str:
        """Генерирует ответ используя модель"""
        try:
            import torch
            
            # Токенизируем промпт
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1500
            )
            
            # Генерируем ответ
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    temperature=0.7,
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
            
            return response if response else "I need more specific information to provide a proper answer."
            
        except Exception as e:
            logger.error(f"❌ Error in response generation: {e}")
            return f"Technical error occurred while generating response. Please try with a simpler question."
    
    def _clean_response(self, response: str) -> str:
        """Очищает сгенерированный ответ"""
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
        
        # Ограничиваем длину
        if len(cleaned) > 1500:
            cleaned = cleaned[:1500] + "..."
        
        return cleaned
    
    async def get_service_status(self):
        """Возвращает статус сервиса"""
        return {
            "service_type": "huggingface_gptq",
            "model_loaded": self.model_loaded,
            "model_name": self.model_name,
            "ollama_available": False,
            "huggingface_available": True,
            "models_available": [self.model_name] if self.model_loaded else [],
            "default_model": self.model_name,
            "base_url": "local_transformers",
            "supported_languages": ["en", "uk"],
            "memory_usage": "~4-6GB with GPTQ quantization",
            "recommendations": [
                "Using HuggingFace Transformers with GPTQ quantization",
                "Model runs locally without external dependencies",
                "Optimized for HuggingFace Spaces environment"
            ]
        }

def create_llm_service(model_name: str = "TheBloke/Llama-2-7B-Chat-GPTQ"):
    """Создает LLM сервис для HuggingFace Spaces"""
    try:
        return HuggingFaceLLMService(model_name)
    except Exception as e:
        logger.error(f"Failed to create HuggingFace LLM service: {e}")
        # Возвращаем fallback сервис
        from app.dependencies import FallbackLLMService
        return FallbackLLMService()