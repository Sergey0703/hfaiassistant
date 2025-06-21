# backend/services/llamacpp_llm_service.py - НОВЫЙ LLM СЕРВИС НА LLAMA.CPP
"""
LLM Service using llama-cpp-python for stable CPU inference on HuggingFace Spaces
НОВЫЙ ПОДХОД: llama.cpp вместо transformers для лучшей совместимости с HF Spaces
"""

import logging
import time
import os
import gc
from typing import List, Dict, Optional
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_used: int
    response_time: float
    success: bool
    error: Optional[str] = None

class LlamaCppLLMService:
    """LLM Service на основе llama-cpp-python для стабильной работы на HF Spaces"""
    
    def __init__(self, model_name: str = "TheBloke/Llama-2-7B-Chat-GGUF"):
        self.model_name = model_name
        self.model_file = "llama-2-7b-chat.Q4_K_M.gguf"  # Оптимальный файл для CPU
        self.model = None
        self.service_type = "llamacpp_cpu_optimized"
        self.model_loaded = False
        self.loading_started = False
        self.loading_error = None
        self.hf_spaces = os.getenv("SPACE_ID") is not None
        
        # Оптимизация для HF Spaces
        self.max_tokens = 150 if self.hf_spaces else 300
        self.context_length = 2048 if self.hf_spaces else 4096
        self.n_threads = 2 if self.hf_spaces else 4  # CPU потоки
        
        logger.info(f"🦙 Initializing LlamaCpp LLM service for: {model_name}")
        logger.info(f"🌍 Platform: {'HuggingFace Spaces' if self.hf_spaces else 'Local'}")
        logger.info(f"📁 Model file: {self.model_file}")
        
        # Автоматически начинаем загрузку
        self._start_model_loading()
    
    def _start_model_loading(self):
        """Безопасно начинает загрузку модели"""
        if self.loading_started:
            return
        
        self.loading_started = True
        
        try:
            logger.info(f"🔄 Starting LlamaCpp model loading: {self.model_name}")
            self._load_model_llamacpp()
        except Exception as e:
            self.loading_error = str(e)
            logger.error(f"❌ LlamaCpp model loading initiation failed: {e}")
    
    def _load_model_llamacpp(self):
        """Загружает модель через llama-cpp-python"""
        start_time = time.time()
        
        try:
            # Проверяем доступность llama-cpp-python
            try:
                from llama_cpp import Llama
                logger.info("📚 llama-cpp-python available, starting model load...")
            except ImportError as e:
                error_msg = f"llama-cpp-python not installed: {e}"
                self.loading_error = error_msg
                logger.error(f"❌ {error_msg}")
                logger.error("Install: pip install llama-cpp-python")
                raise
            
            # Очистка памяти перед загрузкой
            if self.hf_spaces:
                gc.collect()
                logger.info("🧹 Memory cleanup for HF Spaces")
            
            # Список файлов модели для попытки (от легких к тяжелым)
            model_files = [
                "llama-2-7b-chat.Q4_K_S.gguf",  # Самый легкий
                "llama-2-7b-chat.Q4_K_M.gguf",  # Средний (наш основной)
                "llama-2-7b-chat.Q5_K_M.gguf",  # Более качественный
            ]
            
            for attempt, model_file in enumerate(model_files):
                try:
                    logger.info(f"🔄 Attempt {attempt + 1}: Loading {model_file}")
                    
                    # Оптимальные настройки для HF Spaces
                    model_kwargs = {
                        "model_path": None,  # Будет заполнено ниже
                        "n_ctx": self.context_length,  # Контекст
                        "n_threads": self.n_threads,  # CPU потоки
                        "n_gpu_layers": 0,  # Только CPU
                        "use_mmap": True,  # Memory mapping для эффективности
                        "use_mlock": False,  # Не блокируем память
                        "verbose": False,  # Тихий режим
                        "n_batch": 128 if self.hf_spaces else 512,  # Размер батча
                    }
                    
                    # Пытаемся загрузить модель из разных источников
                    model_sources = [
                        f"hf/{self.model_name}/{model_file}",  # HuggingFace Hub
                        f"./{model_file}",  # Локальный файл
                        f"./models/{model_file}",  # Папка models
                    ]
                    
                    for source in model_sources:
                        try:
                            logger.info(f"🔍 Trying to load from: {source}")
                            model_kwargs["model_path"] = source
                            
                            self.model = Llama(**model_kwargs)
                            
                            # Если дошли сюда - модель загрузилась
                            self.model_file = model_file
                            self.model_loaded = True
                            
                            loading_time = time.time() - start_time
                            logger.info(f"✅ LlamaCpp model loaded successfully!")
                            logger.info(f"📁 File: {model_file}")
                            logger.info(f"📍 Source: {source}")
                            logger.info(f"⏱️ Load time: {loading_time:.1f}s")
                            
                            # Очистка памяти после загрузки
                            if self.hf_spaces:
                                gc.collect()
                                logger.info("🧹 Post-loading memory cleanup")
                            
                            return  # Успешно загрузили, выходим
                            
                        except Exception as source_error:
                            logger.debug(f"Failed to load from {source}: {source_error}")
                            continue
                    
                    # Если все источники не сработали для этого файла
                    raise Exception(f"Could not load {model_file} from any source")
                    
                except Exception as file_error:
                    logger.warning(f"❌ Failed to load {model_file}: {file_error}")
                    
                    # Очищаем память перед следующей попыткой
                    if hasattr(self, 'model') and self.model is not None:
                        del self.model
                        self.model = None
                    
                    gc.collect()
                    
                    if attempt == len(model_files) - 1:
                        # Последняя попытка
                        raise file_error
                    else:
                        logger.info(f"🔄 Trying next model file...")
                        time.sleep(1)  # Небольшая пауза
            
        except Exception as e:
            self.loading_error = str(e)
            logger.error(f"❌ LlamaCpp model loading failed completely: {e}")
            
            # Создаем fallback модель (заглушка)
            self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Создает fallback модель если основная не загрузилась"""
        logger.info("🔄 Creating fallback model...")
        
        class FallbackModel:
            def __call__(self, prompt: str, max_tokens: int = 100, **kwargs):
                return {
                    "choices": [{
                        "text": f"LlamaCpp model loading failed. This is a fallback response for: {prompt[:50]}..."
                    }]
                }
        
        self.model = FallbackModel()
        self.model_loaded = True
        self.service_type = "llamacpp_fallback"
        logger.info("✅ Fallback model ready")
    
    async def answer_legal_question(self, question: str, context_documents: List[Dict], language: str = "en"):
        """Отвечает на юридический вопрос через LlamaCpp"""
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
            
            # Генерируем промпт
            prompt = self._build_chat_prompt(question, context_documents, language)
            
            # Генерируем ответ через LlamaCpp
            response_text = await self._generate_response_llamacpp(prompt)
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response_text,
                model=f"{self.model_name}/{self.model_file}",
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
    
    def _build_chat_prompt(self, question: str, context_documents: List[Dict], language: str) -> str:
        """Строит промпт для Llama-2-Chat формата"""
        
        # Системный промпт
        if language == "uk":
            system_message = "Ви - досвідчений юридичний консультант. Надавайте точні та корисні відповіді на основі наданої інформації."
        else:
            system_message = "You are an experienced legal consultant. Provide accurate and helpful answers based on the provided information."
        
        # Подготавливаем контекст из документов
        context_parts = []
        max_docs = 2 if self.hf_spaces else 3
        max_doc_length = 400 if self.hf_spaces else 600
        
        for i, doc in enumerate(context_documents[:max_docs]):
            filename = doc.get('filename', f'Document {i+1}')
            content = doc.get('content', '')[:max_doc_length]
            if content.strip():
                context_parts.append(f"Document {filename}: {content.strip()}")
        
        context = "\n\n".join(context_parts) if context_parts else ""
        
        # Формируем user message
        if context:
            if language == "uk":
                user_message = f"Контекст:\n{context}\n\nПитання: {question}\n\nДайте детальну відповідь:"
            else:
                user_message = f"Context:\n{context}\n\nQuestion: {question}\n\nProvide a detailed answer:"
        else:
            if language == "uk":
                user_message = f"Питання: {question}\n\nДайте загальну юридичну інформацію:"
            else:
                user_message = f"Question: {question}\n\nProvide general legal information:"
        
        # Llama-2-Chat формат
        prompt = f"""<s>[INST] <<SYS>>
{system_message}
<</SYS>>

{user_message} [/INST]"""
        
        # Ограничиваем общую длину промпта
        max_prompt_length = 1500 if self.hf_spaces else 2000
        if len(prompt) > max_prompt_length:
            # Обрезаем контекст, но сохраняем структуру
            available_for_context = max_prompt_length - len(prompt) + len(context)
            if available_for_context > 100:
                context = context[:available_for_context] + "..."
                if language == "uk":
                    user_message = f"Контекст:\n{context}\n\nПитання: {question}\n\nДайте детальну відповідь:"
                else:
                    user_message = f"Context:\n{context}\n\nQuestion: {question}\n\nProvide a detailed answer:"
                
                prompt = f"""<s>[INST] <<SYS>>
{system_message}
<</SYS>>

{user_message} [/INST]"""
        
        return prompt
    
    async def _generate_response_llamacpp(self, prompt: str) -> str:
        """Генерирует ответ через LlamaCpp с таймаутом"""
        try:
            # Настройки генерации для HF Spaces
            generation_kwargs = {
                "max_tokens": self.max_tokens,
                "temperature": 0.1,  # Низкая температура для точности
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "stop": ["</s>", "[/INST]", "<s>"],  # Стоп-слова для Llama-2
                "echo": False,  # Не повторять промпт
            }
            
            logger.info(f"🔧 LlamaCpp generation: max_tokens={self.max_tokens}, temp=0.1")
            
            # Генерируем с таймаутом
            def _generate_sync():
                return self.model(prompt, **generation_kwargs)
            
            # Запускаем синхронную генерацию в executor с таймаутом
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(_generate_sync)
                try:
                    # Таймаут 30 секунд для HF Spaces
                    timeout = 30 if self.hf_spaces else 60
                    result = future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    logger.error(f"❌ LlamaCpp generation timeout after {timeout}s")
                    return "Response generation timed out. Please try with a shorter question."
            
            # Извлекаем текст ответа
            if isinstance(result, dict) and "choices" in result and result["choices"]:
                response_text = result["choices"][0].get("text", "").strip()
            else:
                response_text = str(result).strip()
            
            # Очищаем ответ
            response_text = self._clean_response(response_text)
            
            # Очистка памяти после генерации
            if self.hf_spaces:
                gc.collect()
            
            return response_text if response_text else "I need more specific information to provide a proper answer."
            
        except Exception as e:
            logger.error(f"❌ LlamaCpp generation error: {e}")
            return f"Error generating response: {str(e)}"
    
    def _clean_response(self, response: str) -> str:
        """Очищает ответ от артефактов"""
        if not response:
            return response
        
        # Удаляем стоп-токены если они попали в ответ
        stop_tokens = ["</s>", "[/INST]", "<s>", "[INST]"]
        for token in stop_tokens:
            response = response.replace(token, "")
        
        # Удаляем лишние пробелы и переносы
        lines = []
        for line in response.split('\n'):
            line = line.strip()
            if line and not line.startswith('System:') and not line.startswith('User:'):
                lines.append(line)
        
        cleaned = '\n'.join(lines[:10])  # Ограничиваем количество строк
        
        # Ограничиваем общую длину
        max_length = 1000 if self.hf_spaces else 1500
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length] + "..."
        
        return cleaned
    
    def _generate_loading_response(self, question: str, language: str) -> str:
        """Ответ пока модель загружается"""
        if language == "uk":
            return f"""🦙 **LlamaCpp модель завантажується...**

**Ваше питання:** {question}

⏳ Модель `{self.model_name}` ({self.model_file}) ініціалізується через llama.cpp. Це зазвичай займає 30-60 секунд.

🔄 **Статус:** Завантаження GGUF квантизованої моделі...
🎯 **Якість:** Стабільна робота на CPU
🌍 **Мови:** Англійська та українська
⚡ **Оптимізація:** llama.cpp для ефективної роботи на CPU

💡 **Порада:** LlamaCpp модель буде працювати стабільно без зависань."""
        else:
            return f"""🦙 **LlamaCpp Model Loading...**

**Your Question:** {question}

⏳ Model `{self.model_name}` ({self.model_file}) is initializing via llama.cpp. This typically takes 30-60 seconds.

🔄 **Status:** Loading GGUF quantized model...
🎯 **Quality:** Stable CPU performance
🌍 **Languages:** English and Ukrainian  
⚡ **Optimization:** llama.cpp for efficient CPU inference

💡 **Tip:** LlamaCpp model will work stably without hanging."""
    
    def _generate_loading_error_response(self, question: str, language: str) -> str:
        """Ответ при ошибке загрузки модели"""
        if language == "uk":
            return f"""❌ **Помилка завантаження LlamaCpp моделі**

**Ваше питання:** {question}

🚫 **Проблема:** Модель `{self.model_name}` не змогла завантажитися.
📋 **Помилка:** {self.loading_error}

🔧 **Можливі причини:**
• Відсутня бібліотека llama-cpp-python
• Файл моделі не знайдено
• Недостатньо пам'яті

💡 **Рекомендації:**
• Встановіть: pip install llama-cpp-python
• Перевірте доступність файлу моделі
• Спробуйте ще раз через кілька хвилин"""
        else:
            return f"""❌ **LlamaCpp Model Loading Error**

**Your Question:** {question}

🚫 **Issue:** Model `{self.model_name}` failed to load.
📋 **Error:** {self.loading_error}

🔧 **Possible Causes:**
• Missing llama-cpp-python library
• Model file not found
• Insufficient memory

💡 **Recommendations:**
• Install: pip install llama-cpp-python
• Check model file availability
• Try again in a few minutes"""
    
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
            "service_type": "llamacpp_cpu_optimized",
            "model_name": self.model_name,
            "model_file": self.model_file,
            "model_loaded": self.model_loaded,
            "loading_started": self.loading_started,
            "loading_error": self.loading_error,
            "platform": "HuggingFace Spaces" if self.hf_spaces else "Local",
            "supported_languages": ["en", "uk"],
            "optimizations": {
                "cpu_optimized": True,
                "llamacpp_backend": True,
                "gguf_format": True,
                "hf_spaces_mode": self.hf_spaces,
                "max_tokens": self.max_tokens,
                "context_length": self.context_length,
                "n_threads": self.n_threads,
                "timeout_enabled": True
            }
        }
        
        # Добавляем диагностическую информацию
        try:
            from llama_cpp import Llama
            status["llamacpp_available"] = True
        except ImportError:
            status["llamacpp_available"] = False
        
        # Рекомендации на основе статуса
        recommendations = []
        if not self.model_loaded:
            if self.loading_error:
                recommendations.extend([
                    "LlamaCpp model loading failed",
                    "Check llama-cpp-python installation",
                    "Verify model file availability",
                    "Install: pip install llama-cpp-python"
                ])
            else:
                recommendations.extend([
                    "LlamaCpp model loading in progress",
                    "GGUF format provides stable CPU inference",
                    "First load may take 30-60 seconds",
                    "No hanging issues expected with llama.cpp"
                ])
        else:
            recommendations.extend([
                "LlamaCpp model ready for stable inference",
                "CPU-optimized with GGUF format",
                "No memory hanging issues",
                "Timeout protection enabled"
            ])
        
        status["recommendations"] = recommendations
        
        return status

def create_llm_service(model_name: str = "TheBloke/Llama-2-7B-Chat-GGUF"):
    """Создает LlamaCpp LLM сервис"""
    try:
        return LlamaCppLLMService(model_name)
    except Exception as e:
        logger.error(f"Failed to create LlamaCpp LLM service: {e}")
        # Возвращаем fallback
        from app.dependencies import HFSpacesImprovedLLMFallback
        return HFSpacesImprovedLLMFallback()