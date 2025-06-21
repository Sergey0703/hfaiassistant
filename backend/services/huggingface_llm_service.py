# backend/services/huggingface_llm_service.py - ИСПРАВЛЕННАЯ ВЕРСИЯ ДЛЯ HF SPACES
"""
LLM Service using HuggingFace Transformers with GPTQ quantization
ИСПРАВЛЕНИЯ: Оптимизация для HF Spaces, улучшенная обработка ошибок, memory management
"""

import logging
import time
import os
import gc
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
    """Оптимизированный LLM Service для HF Spaces с GPTQ поддержкой"""
    
    def __init__(self, model_name: str = "TheBloke/Llama-2-7B-Chat-GPTQ"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.service_type = "huggingface_gptq_optimized"
        self.model_loaded = False
        self.loading_started = False
        self.loading_error = None
        self.hf_spaces = os.getenv("SPACE_ID") is not None
        
        # HF Spaces оптимизации
        self.max_memory_gb = 14 if self.hf_spaces else 32  # HF Spaces лимит ~16GB
        self.loading_timeout = 300 if self.hf_spaces else 180  # 5 минут для HF Spaces
        
        logger.info(f"🤖 Initializing GPTQ LLM service for: {model_name}")
        logger.info(f"🌍 Platform: {'HuggingFace Spaces' if self.hf_spaces else 'Local'}")
        
        # Автоматически начинаем загрузку
        self._start_model_loading()
    
    def _start_model_loading(self):
        """Безопасно начинает загрузку модели"""
        if self.loading_started:
            return
        
        self.loading_started = True
        
        try:
            logger.info(f"🔄 Starting GPTQ model loading: {self.model_name}")
            self._load_model_with_optimizations()
        except Exception as e:
            self.loading_error = str(e)
            logger.error(f"❌ Model loading initiation failed: {e}")
    
    def _load_model_with_optimizations(self):
        """Загружает модель с оптимизациями для HF Spaces"""
        start_time = time.time()
        
        try:
            # Проверяем доступность библиотек
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            logger.info("📚 Dependencies available, starting model load...")
            
            # HF Spaces оптимизации
            if self.hf_spaces:
                # Освобождаем память перед загрузкой
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                logger.info("🧹 Memory cleaned for HF Spaces")
            
            # Список моделей для попытки (ваша приоритетная первая)
            model_candidates = [
                self.model_name,  # Ваша основная модель
                "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ",  # Более легкая альтернатива
                "TheBloke/Llama-2-7B-Chat-GPTQ"  # Fallback
            ]
            
            for attempt, candidate_model in enumerate(model_candidates):
                try:
                    logger.info(f"🔄 Attempt {attempt + 1}: Loading {candidate_model}")
                    
                    # Загружаем токенайзер
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        candidate_model,
                        trust_remote_code=True,
                        use_fast=True,
                        cache_dir="./.cache" if self.hf_spaces else None
                    )
                    
                    # Настройки для GPTQ с HF Spaces оптимизациями
                    # ИСПРАВЛЕННЫЕ настройки для GPTQ с агрессивной оптимизацией памяти
                    model_kwargs = {
                    "torch_dtype": torch.float16,  # Изменено с float32
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                    "cache_dir": "./.cache" if self.hf_spaces else None,
                    "device_map": "auto",
                    "max_memory": {"cpu": "12GB", 0: "3GB"},  # Жесткие лимиты
                    "offload_folder": "./offload",  # CPU offloading
                    "use_cache": False  # Отключаем кэш
                    }
                    else:
                        model_kwargs["device_map"] = "auto"
                    
                    # Пытаемся загрузить с GPTQ
                    try:
                        logger.info(f"🔄 Loading {candidate_model} with GPTQ quantization...")
                        
                        # Проверяем наличие auto-gptq
                        try:
                            import auto_gptq
                            logger.info("✅ auto-gptq available")
                        except ImportError:
                            logger.warning("⚠️ auto-gptq not available, trying without")
                        
                        self.model = AutoModelForCausalLM.from_pretrained(
                            candidate_model,
                            **model_kwargs
                        )
                        
                        logger.info(f"✅ Successfully loaded with GPTQ: {candidate_model}")
                        
                    except Exception as gptq_error:
                        logger.warning(f"⚠️ GPTQ loading failed: {gptq_error}")
                        logger.info(f"🔄 Trying {candidate_model} without GPTQ optimizations...")
                        
                        # Fallback без специальных GPTQ настроек
                        simplified_kwargs = {
                            "torch_dtype": torch.float16,
                            "device_map": "cpu" if self.hf_spaces and not torch.cuda.is_available() else "auto",
                            "low_cpu_mem_usage": True
                        }
                        
                        self.model = AutoModelForCausalLM.from_pretrained(
                            candidate_model,
                            **simplified_kwargs
                        )
                        
                        logger.info(f"✅ Loaded without GPTQ optimizations: {candidate_model}")
                    
                    # Настройка токенайзера
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    # Обновляем model_name на успешно загруженную модель
                    self.model_name = candidate_model
                    self.model_loaded = True
                    
                    loading_time = time.time() - start_time
                    logger.info(f"🎉 GPTQ model ready: {candidate_model} (loaded in {loading_time:.1f}s)")
                    
                    # Очищаем память после загрузки
                    if self.hf_spaces:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    return  # Успешно загрузили, выходим
                    
                except Exception as model_error:
                    logger.warning(f"❌ Failed to load {candidate_model}: {model_error}")
                    
                    # Очищаем память перед следующей попыткой
                    if hasattr(self, 'model') and self.model is not None:
                        del self.model
                        self.model = None
                    if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                        del self.tokenizer  
                        self.tokenizer = None
                    
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    if attempt == len(model_candidates) - 1:
                        # Последняя попытка
                        raise model_error
                    else:
                        logger.info(f"🔄 Trying next model candidate...")
                        time.sleep(2)  # Небольшая пауза между попытками
            
        except ImportError as e:
            error_msg = f"Missing dependencies: {e}"
            self.loading_error = error_msg
            logger.error(f"❌ {error_msg}")
            logger.error("Install: pip install transformers torch auto-gptq accelerate")
            raise
            
        except Exception as e:
            self.loading_error = str(e)
            logger.error(f"❌ Model loading failed completely: {e}")
            raise
    
    async def answer_legal_question(self, question: str, context_documents: List[Dict], language: str = "en"):
        """Отвечает на юридический вопрос с оптимизацией для HF Spaces"""
        start_time = time.time()
        
        try:
            # Проверяем статус модели
            if not self.model_loaded:
                if self.loading_error:
                    return LLMResponse(
                        content=self._generate_loading_error_response(question, language),
                        model=self.model_name,
                        tokens_used=0,
                        response_time=time.time() - start_time,
                        success=False,
                        error=f"Model loading failed: {self.loading_error}"
                    )
                else:
                    return LLMResponse(
                        content=self._generate_loading_response(question, language),
                        model=self.model_name,
                        tokens_used=0,
                        response_time=time.time() - start_time,
                        success=True,
                        error="Model still loading"
                    )
            
            # Генерируем промпт с оптимизацией для HF Spaces
            prompt = self._build_optimized_prompt(question, context_documents, language)
            
            # Генерируем ответ
            response_text = await self._generate_response_optimized(prompt)
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response_text,
                model=self.model_name,
                tokens_used=len(response_text.split()),
                response_time=response_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return LLMResponse(
                content=self._generate_error_response(question, language, str(e)),
                model=self.model_name,
                tokens_used=0,
                response_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    def _build_optimized_prompt(self, question: str, context_documents: List[Dict], language: str) -> str:
        """Строит оптимизированный промпт для HF Spaces (короче и эффективнее)"""
        
        # Сокращенные системные промпты для экономии токенов
        if language == "uk":
            system_prompt = "Ви - юридичний консультант. Відповідайте коротко та по суті на основі наданих документів."
            context_intro = "Документи:"
            answer_intro = f"Питання: {question}\nВідповідь:"
        else:
            system_prompt = "You are a legal consultant. Answer concisely based on provided documents."
            context_intro = "Documents:"
            answer_intro = f"Question: {question}\nAnswer:"
        
        # Ограничиваем контекст для HF Spaces
        max_docs = 2 if self.hf_spaces else 3
        max_doc_length = 800 if self.hf_spaces else 1200
        
        context_parts = []
        for i, doc in enumerate(context_documents[:max_docs]):
            filename = doc.get('filename', f'Doc {i+1}')
            content = doc.get('content', '')[:max_doc_length]
            context_parts.append(f"{filename}: {content}")
        
        context = f"\n{context_intro}\n" + "\n".join(context_parts) if context_parts else ""
        
        # Собираем компактный промпт
        prompt = f"{system_prompt}{context}\n\n{answer_intro}"
        
        # Жесткое ограничение длины для HF Spaces
        max_prompt_length = 2000 if self.hf_spaces else 3000
        if len(prompt) > max_prompt_length:
            prompt = prompt[:max_prompt_length] + "..."
        
        return prompt
    
    async def _generate_response_optimized(self, prompt: str) -> str:
        """Генерирует ответ с оптимизацией для HF Spaces"""
        try:
            import torch
            
            # Сокращенные настройки для HF Spaces
            max_new_tokens = 20 if self.hf_spaces else 800
            max_input_length = 1500 if self.hf_spaces else 2000
            
            # Токенизируем с ограничениями
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_input_length
            )
            
            # Оптимизированная генерация для HF Spaces
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "num_return_sequences": 1,
                "temperature": 0.1,  # Более консервативная для юридических вопросов
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "no_repeat_ngram_size": 2,  # Избегаем повторений
                "use_cache": False
            }
            
            # HF Spaces specific optimizations
            if self.hf_spaces:
                generation_kwargs.update({
                    "use_cache": True,
                    "attention_mask": torch.ones_like(inputs)
                })
            
            # Генерируем с контролем памяти
            with torch.no_grad():
                if self.hf_spaces:
                    # Очищаем кэш перед генерацией
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                outputs = self.model.generate(inputs, **generation_kwargs)
            
            # Декодируем ответ
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Извлекаем только новую часть
            response = full_response[len(prompt):].strip()
            
            # Очищаем ответ
            response = self._clean_response_optimized(response)
            
            # Очищаем память после генерации
            if self.hf_spaces:
                del outputs, inputs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return response if response else "I need more specific information to provide a proper legal analysis."
            
        except torch.cuda.OutOfMemoryError:
            logger.error("❌ CUDA out of memory during generation")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return "Memory limit reached. Please try with a shorter question."
            
        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            return f"Technical error during response generation. Please try again."
    
    def _clean_response_optimized(self, response: str) -> str:
        """Быстрая очистка ответа для HF Spaces"""
        if not response:
            return response
        
        # Удаляем повторения и обрезаем
        lines = response.split('\n')
        cleaned_lines = []
        seen = set()
        
        for line in lines[:10]:  # Ограничиваем количество строк
            line = line.strip()
            if line and len(line) > 10 and line not in seen:
                cleaned_lines.append(line)
                seen.add(line)
        
        cleaned = '\n'.join(cleaned_lines)
        
        # Ограничиваем длину для HF Spaces
        max_length = 1500 if self.hf_spaces else 2000
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length] + "..."
        
        return cleaned
    
    def _generate_loading_response(self, question: str, language: str) -> str:
        """Ответ пока модель загружается"""
        if language == "uk":
            return f"""🤖 **GPTQ модель завантажується...**

**Ваше питання:** {question}

⏳ Модель `{self.model_name}` ініціалізується. Це зазвичай займає 1-2 хвилини для першого запуску.

🔄 **Статус:** Завантаження GPTQ квантизованої моделі...
🎯 **Якість:** Висока якість юридичних консультацій після завершення
🌍 **Мови:** Англійська та українська
⚡ **Оптимізація:** GPTQ 4-bit квантизація для ефективності

💡 **Порада:** Спробуйте ще раз через хвилину для отримання повної AI відповіді."""
        else:
            return f"""🤖 **GPTQ Model Loading...**

**Your Question:** {question}

⏳ Model `{self.model_name}` is initializing. This typically takes 1-2 minutes for first startup.

🔄 **Status:** Loading GPTQ quantized model...
🎯 **Quality:** High-quality legal consultations when complete
🌍 **Languages:** English and Ukrainian  
⚡ **Optimization:** GPTQ 4-bit quantization for efficiency

💡 **Tip:** Try again in a minute for full AI response."""
    
    def _generate_loading_error_response(self, question: str, language: str) -> str:
        """Ответ при ошибке загрузки модели"""
        if language == "uk":
            return f"""❌ **Помилка завантаження GPTQ моделі**

**Ваше питання:** {question}

🚫 **Проблема:** Модель `{self.model_name}` не змогла завантажитися.
📋 **Помилка:** {self.loading_error}

🔧 **Можливі причини:**
• Недостатньо пам'яті для GPTQ моделі
• Відсутні залежності (auto-gptq, transformers)
• Тимчасові проблеми з HuggingFace Hub

💡 **Рекомендації:**
• Спробуйте ще раз через кілька хвилин
• Модель може завантажуватися у фоновому режимі
• Перевірте доступність HuggingFace сервісів"""
        else:
            return f"""❌ **GPTQ Model Loading Error**

**Your Question:** {question}

🚫 **Issue:** Model `{self.model_name}` failed to load.
📋 **Error:** {self.loading_error}

🔧 **Possible Causes:**
• Insufficient memory for GPTQ model
• Missing dependencies (auto-gptq, transformers)
• Temporary HuggingFace Hub issues

💡 **Recommendations:**
• Try again in a few minutes
• Model may be loading in background
• Check HuggingFace services availability"""
    
    def _generate_error_response(self, question: str, language: str, error: str) -> str:
        """Ответ при ошибке генерации"""
        if language == "uk":
            return f"""⚠️ **Тимчасова технічна проблема**

Виникла помилка при обробці запиту: "{question}"

🔧 Деталі: {error}

Спробуйте переформулювати питання або повторіть спробу."""
        else:
            return f"""⚠️ **Temporary Technical Issue**

Error processing query: "{question}"

🔧 Details: {error}

Please try rephrasing your question or try again."""
    
    async def get_service_status(self):
        """Детальний статус сервиса для диагностики"""
        status = {
            "service_type": "huggingface_gptq_optimized",
            "model_name": self.model_name,
            "model_loaded": self.model_loaded,
            "loading_started": self.loading_started,
            "loading_error": self.loading_error,
            "platform": "HuggingFace Spaces" if self.hf_spaces else "Local",
            "supported_languages": ["en", "uk"],
            "optimizations": {
                "gptq_quantization": True,
                "memory_optimized": self.hf_spaces,
                "hf_spaces_mode": self.hf_spaces,
                "max_memory_gb": self.max_memory_gb,
                "loading_timeout": self.loading_timeout
            }
        }
        
        # Добавляем диагностическую информацию
        try:
            import torch
            status["torch_available"] = True
            status["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                status["cuda_device_count"] = torch.cuda.device_count()
        except ImportError:
            status["torch_available"] = False
        
        try:
            import transformers
            status["transformers_version"] = transformers.__version__
        except ImportError:
            status["transformers_available"] = False
        
        try:
            import auto_gptq
            status["auto_gptq_available"] = True
        except ImportError:
            status["auto_gptq_available"] = False
        
        # Рекомендации на основе статуса
        recommendations = []
        if not self.model_loaded:
            if self.loading_error:
                recommendations.extend([
                    "Check available memory and dependencies",
                    "Model loading failed - see error details",
                    "Try restarting the application"
                ])
            else:
                recommendations.extend([
                    "Model is loading in background",
                    "First load may take 1-2 minutes",
                    "Check /startup-progress for loading status"
                ])
        else:
            recommendations.extend([
                "Model ready for high-quality responses",
                "Supports Ukrainian legal consultations",
                "Optimized for HuggingFace Spaces"
            ])
        
        status["recommendations"] = recommendations
        
        return status

def create_llm_service(model_name: str = "TheBloke/Llama-2-7B-Chat-GPTQ"):
    """Создает оптимизированный LLM сервис"""
    try:
        return HuggingFaceLLMService(model_name)
    except Exception as e:
        logger.error(f"Failed to create HuggingFace LLM service: {e}")
        # Возвращаем улучшенный fallback
        from app.dependencies import ImprovedFallbackLLMService
        return ImprovedFallbackLLMService()