# backend/services/huggingface_llm_service.py - ИСПРАВЛЕННАЯ ВЕРСИЯ С ОПТИМИЗАЦИЕЙ ПАМЯТИ
"""
LLM Service using HuggingFace Transformers with GPTQ quantization
ИСПРАВЛЕНИЯ: Агрессивная оптимизация памяти для Llama-2-7B-Chat-GPTQ на HF Spaces
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
    """Оптимизированный LLM Service для HF Spaces с агрессивной оптимизацией памяти"""
    
    def __init__(self, model_name: str = "TheBloke/Llama-2-7B-Chat-GPTQ"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.service_type = "huggingface_gptq_memory_optimized"
        self.model_loaded = False
        self.loading_started = False
        self.loading_error = None
        self.hf_spaces = os.getenv("SPACE_ID") is not None
        
        # АГРЕССИВНАЯ оптимизация памяти для HF Spaces
        self.max_memory_gb = 12 if self.hf_spaces else 32  # Снижено с 14GB
        self.loading_timeout = 300 if self.hf_spaces else 180  # 5 минут для HF Spaces
        
        logger.info(f"🤖 Initializing MEMORY-OPTIMIZED GPTQ LLM service for: {model_name}")
        logger.info(f"🌍 Platform: {'HuggingFace Spaces' if self.hf_spaces else 'Local'}")
        logger.info(f"💾 Memory limit: {self.max_memory_gb}GB")
        
        # Автоматически начинаем загрузку
        self._start_model_loading()
    
    def _start_model_loading(self):
        """Безопасно начинает загрузку модели"""
        if self.loading_started:
            return
        
        self.loading_started = True
        
        try:
            logger.info(f"🔄 Starting MEMORY-OPTIMIZED GPTQ model loading: {self.model_name}")
            self._load_model_with_memory_optimization()
        except Exception as e:
            self.loading_error = str(e)
            logger.error(f"❌ Model loading initiation failed: {e}")
    
    def _load_model_with_memory_optimization(self):
        """Загружает модель с АГРЕССИВНОЙ оптимизацией памяти для HF Spaces"""
        start_time = time.time()
        
        try:
            # Проверяем доступность библиотек
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            logger.info("📚 Dependencies available, starting MEMORY-OPTIMIZED model load...")
            
            # АГРЕССИВНАЯ очистка памяти перед загрузкой
            if self.hf_spaces:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
                logger.info("🧹 Aggressive memory cleanup for HF Spaces")
            
            # Список моделей для попытки (ваша приоритетная первая)
            model_candidates = [
                self.model_name,  # Ваша основная модель
                "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ",  # Более легкая альтернатива
                "TheBloke/Llama-2-7B-Chat-GPTQ"  # Fallback
            ]
            
            for attempt, candidate_model in enumerate(model_candidates):
                try:
                    logger.info(f"🔄 Attempt {attempt + 1}: Loading {candidate_model} with MEMORY OPTIMIZATION")
                    
                    # Загружаем токенайзер с минимальными настройками
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        candidate_model,
                        trust_remote_code=True,
                        use_fast=True,
                        cache_dir="./.cache" if self.hf_spaces else None,
                        legacy=False  # Отключаем legacy mode
                    )
                    
                    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Агрессивные настройки памяти
                    model_kwargs = {
                        "torch_dtype": torch.float16,  # ИСПРАВЛЕНО: float16 вместо float32
                        "trust_remote_code": True,
                        "low_cpu_mem_usage": True,
                        "cache_dir": "./.cache" if self.hf_spaces else None,
                        "device_map": "auto",
                        "max_memory": {"cpu": "10GB", 0: "3GB"} if torch.cuda.is_available() else {"cpu": "12GB"},  # Жесткие лимиты
                        "offload_folder": "./offload",  # CPU offloading
                        "offload_state_dict": True,  # Offload state dict
                    }
                    
                    logger.info(f"🔧 Using aggressive memory settings: max_memory={model_kwargs['max_memory']}")
                    
                    # Пытаемся загрузить с GPTQ
                    try:
                        logger.info(f"🔄 Loading {candidate_model} with GPTQ + MEMORY OPTIMIZATION...")
                        
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
                        
                        logger.info(f"✅ Successfully loaded with GPTQ + MEMORY OPTIMIZATION: {candidate_model}")
                        
                    except Exception as gptq_error:
                        logger.warning(f"⚠️ GPTQ loading failed: {gptq_error}")
                        logger.info(f"🔄 Trying {candidate_model} without GPTQ optimizations...")
                        
                        # Fallback без специальных GPTQ настроек но с memory optimization
                        simplified_kwargs = {
                            "torch_dtype": torch.float16,
                            "device_map": "cpu" if self.hf_spaces and not torch.cuda.is_available() else "auto",
                            "low_cpu_mem_usage": True,
                            "max_memory": {"cpu": "10GB", 0: "3GB"} if torch.cuda.is_available() else {"cpu": "12GB"},
                            "offload_folder": "./offload"
                        }
                        
                        self.model = AutoModelForCausalLM.from_pretrained(
                            candidate_model,
                            **simplified_kwargs
                        )
                        
                        logger.info(f"✅ Loaded without GPTQ but with MEMORY OPTIMIZATION: {candidate_model}")
                    
                    # Настройка токенайзера
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    # Обновляем model_name на успешно загруженную модель
                    self.model_name = candidate_model
                    self.model_loaded = True
                    
                    loading_time = time.time() - start_time
                    logger.info(f"🎉 MEMORY-OPTIMIZED GPTQ model ready: {candidate_model} (loaded in {loading_time:.1f}s)")
                    
                    # АГРЕССИВНАЯ очистка памяти после загрузки
                    if self.hf_spaces:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        logger.info("🧹 Post-loading memory cleanup completed")
                    
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
        """Отвечает на юридический вопрос с MEMORY-OPTIMIZED генерацией"""
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
            
            # Генерируем ответ с MEMORY OPTIMIZATION
            response_text = await self._generate_response_memory_optimized(prompt)
            
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
        max_doc_length = 600 if self.hf_spaces else 1000  # Еще более строгое ограничение
        
        context_parts = []
        for i, doc in enumerate(context_documents[:max_docs]):
            filename = doc.get('filename', f'Doc {i+1}')
            content = doc.get('content', '')[:max_doc_length]
            context_parts.append(f"{filename}: {content}")
        
        context = f"\n{context_intro}\n" + "\n".join(context_parts) if context_parts else ""
        
        # Собираем компактный промпт
        prompt = f"{system_prompt}{context}\n\n{answer_intro}"
        
        # Жесткое ограничение длины для HF Spaces
        max_prompt_length = 1500 if self.hf_spaces else 2500  # Еще более строгое ограничение
        if len(prompt) > max_prompt_length:
            prompt = prompt[:max_prompt_length] + "..."
        
        return prompt
    
    async def _generate_response_memory_optimized(self, prompt: str) -> str:
        """Генерирует ответ с АГРЕССИВНОЙ оптимизацией памяти для HF Spaces"""
        try:
            import torch
            
            # АГРЕССИВНО сокращенные настройки для HF Spaces
            max_new_tokens = 100 if self.hf_spaces else 200  # Увеличено с 20
            max_input_length = 1200 if self.hf_spaces else 1800  # Строгое ограничение
            
            # Токенизируем с ограничениями
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_input_length
            )
            
            # MEMORY-OPTIMIZED генерация для HF Spaces
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "num_return_sequences": 1,
                "temperature": 0.1,  # Очень консервативная для стабильности
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "no_repeat_ngram_size": 2,
                "use_cache": False,  # КРИТИЧНО: отключаем кэш для экономии памяти
                "return_dict_in_generate": False,  # Экономим память
            }
            
            logger.info(f"🔧 Memory-optimized generation: max_tokens={max_new_tokens}, use_cache=False")
            
            # Генерируем с AGGRESSIVE контролем памяти
            with torch.no_grad():
                if self.hf_spaces:
                    # Очищаем кэш перед генерацией
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()
                
                outputs = self.model.generate(inputs, **generation_kwargs)
            
            # Декодируем ответ
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Извлекаем только новую часть
            response = full_response[len(prompt):].strip()
            
            # Очищаем ответ
            response = self._clean_response_optimized(response)
            
            # АГРЕССИВНАЯ очистка памяти после генерации
            if self.hf_spaces:
                del outputs, inputs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
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
        
        for line in lines[:8]:  # Ограничиваем количество строк
            line = line.strip()
            if line and len(line) > 10 and line not in seen:
                cleaned_lines.append(line)
                seen.add(line)
        
        cleaned = '\n'.join(cleaned_lines)
        
        # Ограничиваем длину для HF Spaces
        max_length = 1200 if self.hf_spaces else 1800
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length] + "..."
        
        return cleaned
    
    def _generate_loading_response(self, question: str, language: str) -> str:
        """Ответ пока модель загружается"""
        if language == "uk":
            return f"""🤖 **MEMORY-OPTIMIZED GPTQ модель завантажується...**

**Ваше питання:** {question}

⏳ Модель `{self.model_name}` ініціалізується з оптимізацією пам'яті. Це зазвичай займає 1-2 хвилини для першого запуску.

🔄 **Статус:** Завантаження GPTQ квантизованої моделі з агрессивною оптимізацією пам'яті...
🎯 **Якість:** Висока якість юридичних консультацій після завершення
🌍 **Мови:** Англійська та українська
⚡ **Оптимізація:** GPTQ 4-bit квантизація + агрессивне управління пам'яттю

💡 **Порада:** Спробуйте ще раз через хвилину для отримання повної AI відповіді."""
        else:
            return f"""🤖 **MEMORY-OPTIMIZED GPTQ Model Loading...**

**Your Question:** {question}

⏳ Model `{self.model_name}` is initializing with memory optimization. This typically takes 1-2 minutes for first startup.

🔄 **Status:** Loading GPTQ quantized model with aggressive memory optimization...
🎯 **Quality:** High-quality legal consultations when complete
🌍 **Languages:** English and Ukrainian  
⚡ **Optimization:** GPTQ 4-bit quantization + aggressive memory management

💡 **Tip:** Try again in a minute for full AI response."""
    
    def _generate_loading_error_response(self, question: str, language: str) -> str:
        """Ответ при ошибке загрузки модели"""
        if language == "uk":
            return f"""❌ **Помилка завантаження MEMORY-OPTIMIZED GPTQ моделі**

**Ваше питання:** {question}

🚫 **Проблема:** Модель `{self.model_name}` не змогла завантажитися навіть з оптимізацією пам'яті.
📋 **Помилка:** {self.loading_error}

🔧 **Можливі причини:**
• Недостатньо пам'яті навіть з агрессивною оптимізацією
• Відсутні залежності (auto-gptq, transformers)
• Тимчасові проблеми з HuggingFace Hub

💡 **Рекомендації:**
• Спробуйте ще раз через кілька хвилин
• Модель може завантажуватися у фоновому режимі
• Перевірте доступність HuggingFace сервісів"""
        else:
            return f"""❌ **MEMORY-OPTIMIZED GPTQ Model Loading Error**

**Your Question:** {question}

🚫 **Issue:** Model `{self.model_name}` failed to load even with memory optimization.
📋 **Error:** {self.loading_error}

🔧 **Possible Causes:**
• Insufficient memory even with aggressive optimization
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
            "service_type": "huggingface_gptq_memory_optimized",
            "model_name": self.model_name,
            "model_loaded": self.model_loaded,
            "loading_started": self.loading_started,
            "loading_error": self.loading_error,
            "platform": "HuggingFace Spaces" if self.hf_spaces else "Local",
            "supported_languages": ["en", "uk"],
            "optimizations": {
                "gptq_quantization": True,
                "memory_optimized": True,
                "aggressive_memory_management": True,
                "hf_spaces_mode": self.hf_spaces,
                "max_memory_gb": self.max_memory_gb,
                "loading_timeout": self.loading_timeout,
                "use_cache_disabled": True,
                "cpu_offloading": True
            }
        }
        
        # Добавляем диагностическую информацию
        try:
            import torch
            status["torch_available"] = True
            status["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                status["cuda_device_count"] = torch.cuda.device_count()
                status["cuda_memory_allocated"] = torch.cuda.memory_allocated()
                status["cuda_memory_cached"] = torch.cuda.memory_reserved()
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
                    "Memory optimization attempted but failed",
                    "Check available memory and dependencies",
                    "Consider using a smaller model",
                    "Try restarting with fewer background processes"
                ])
            else:
                recommendations.extend([
                    "Model loading with memory optimization in progress",
                    "First load may take 1-2 minutes",
                    "Aggressive memory management active",
                    "Check /startup-progress for loading status"
                ])
        else:
            recommendations.extend([
                "Model ready with memory optimization",
                "GPTQ quantization active",
                "Aggressive memory management enabled",
                "Optimized for HuggingFace Spaces constraints"
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
        from app.dependencies import HFSpacesImprovedLLMFallback
        return HFSpacesImprovedLLMFallback()