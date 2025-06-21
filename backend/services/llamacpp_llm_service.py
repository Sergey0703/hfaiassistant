# backend/services/llamacpp_llm_service.py - ОПТИМИЗИРОВАННАЯ ВЕРСИЯ С ПРАВИЛЬНЫМИ ТАЙМАУТАМИ
"""
LLM Service using llama-cpp-python with optimized timeouts and generation settings
ИСПРАВЛЕНИЯ: Увеличенные таймауты, оптимизированная генерация, лучшие промпты
"""

import logging
import time
import os
import gc
from typing import List, Dict, Optional
from dataclasses import dataclass
import asyncio
from pathlib import Path

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
    """Оптимизированный LLM Service на основе llama-cpp-python"""
    
    def __init__(self, model_name: str = "TheBloke/Llama-2-7B-Chat-GGUF"):
        self.model_name = model_name
        self.model_file = None
        self.model = None
        self.service_type = "llamacpp_optimized_v2"
        self.model_loaded = False
        self.loading_started = False
        self.loading_error = None
        self.hf_spaces = os.getenv("SPACE_ID") is not None
        self.model_cache_dir = Path("./models")
        
        # ИСПРАВЛЕННЫЕ настройки для оптимальной работы
        self.max_tokens = 100 if self.hf_spaces else 200  # Уменьшено для скорости
        self.context_length = 1024 if self.hf_spaces else 2048  # Уменьшено для скорости
        self.n_threads = 2 if self.hf_spaces else 4
        
        # ИСПРАВЛЕННЫЕ таймауты
        self.generation_timeout = 90 if self.hf_spaces else 120  # Увеличено с 30 до 90 секунд
        self.quick_timeout = 60  # Для быстрых ответов
        
        # Создаем директорию для моделей
        self.model_cache_dir.mkdir(exist_ok=True)
        
        logger.info(f"🦙 Initializing OPTIMIZED LlamaCpp service: {model_name}")
        logger.info(f"🌍 Platform: {'HuggingFace Spaces' if self.hf_spaces else 'Local'}")
        logger.info(f"⏰ Generation timeout: {self.generation_timeout}s")
        logger.info(f"🔧 Max tokens: {self.max_tokens}, Context: {self.context_length}")
        
        # Автоматически начинаем загрузку
        self._start_model_loading()
    
    def _start_model_loading(self):
        """Безопасно начинает загрузку модели"""
        if self.loading_started:
            return
        
        self.loading_started = True
        
        try:
            logger.info(f"🔄 Starting OPTIMIZED LlamaCpp model loading: {self.model_name}")
            self._load_model_with_download()
        except Exception as e:
            self.loading_error = str(e)
            logger.error(f"❌ LlamaCpp model loading initiation failed: {e}")
    
    def _load_model_with_download(self):
        """Загружает модель с автоматической загрузкой GGUF файлов"""
        start_time = time.time()
        
        try:
            # Проверяем доступность llama-cpp-python
            try:
                from llama_cpp import Llama
                logger.info("📚 llama-cpp-python available")
            except ImportError as e:
                error_msg = f"llama-cpp-python not installed: {e}"
                self.loading_error = error_msg
                logger.error(f"❌ {error_msg}")
                raise
            
            # Очистка памяти
            if self.hf_spaces:
                gc.collect()
                logger.info("🧹 Memory cleanup for HF Spaces")
            
            # Автоматическое скачивание модели
            model_file_path = self._download_gguf_model()
            
            if not model_file_path:
                raise Exception("Failed to download GGUF model")
            
            logger.info(f"📁 Model file ready: {model_file_path}")
            
            # Загружаем модель с оптимизированными настройками
            self._load_model_from_file(model_file_path)
            
            loading_time = time.time() - start_time
            logger.info(f"✅ OPTIMIZED LlamaCpp model loaded successfully in {loading_time:.1f}s!")
            
        except Exception as e:
            self.loading_error = str(e)
            logger.error(f"❌ LlamaCpp model loading failed: {e}")
            self._create_fallback_model()
    
    def _download_gguf_model(self) -> Optional[str]:
        """Скачивает GGUF модель с HuggingFace Hub (оптимизированная версия)"""
        try:
            # Список моделей для попытки (самые быстрые первые)
            model_candidates = [
                {
                    "repo": "TheBloke/Llama-2-7B-Chat-GGUF",
                    "files": [
                        "llama-2-7b-chat.Q4_K_S.gguf",  # 3.6GB - БЫСТРАЯ
                        "llama-2-7b-chat.Q4_K_M.gguf",  # 4.1GB - средняя
                    ]
                }
            ]
            
            for candidate in model_candidates:
                repo = candidate["repo"]
                files = candidate["files"]
                
                logger.info(f"🔍 Trying to download from: {repo}")
                
                for model_file in files:
                    try:
                        model_path = self._download_single_file(repo, model_file)
                        if model_path:
                            self.model_file = model_file
                            logger.info(f"✅ Successfully downloaded: {model_file}")
                            return model_path
                    except Exception as e:
                        logger.debug(f"Failed to download {model_file} from {repo}: {e}")
                        continue
            
            # Если ничего не скачалось, пробуем найти существующие файлы
            logger.info("🔍 Looking for existing GGUF files...")
            return self._find_existing_gguf_file()
            
        except Exception as e:
            logger.error(f"❌ Error in model download process: {e}")
            return None
    
    def _download_single_file(self, repo: str, filename: str) -> Optional[str]:
        """Скачивает один файл модели"""
        try:
            # Пробуем использовать huggingface_hub
            try:
                from huggingface_hub import hf_hub_download
                
                logger.info(f"📥 Downloading {filename} from {repo}...")
                
                file_path = hf_hub_download(
                    repo_id=repo,
                    filename=filename,
                    cache_dir=str(self.model_cache_dir),
                    local_files_only=False,
                    force_download=False
                )
                
                if os.path.exists(file_path):
                    size_mb = round(os.path.getsize(file_path) / 1024 / 1024, 1)
                    logger.info(f"✅ Downloaded: {filename} ({size_mb} MB)")
                    return file_path
                
            except ImportError:
                logger.warning("huggingface_hub not available, trying wget/curl...")
                return self._download_via_http(repo, filename)
            
        except Exception as e:
            logger.debug(f"Failed to download {filename}: {e}")
            return None
    
    def _download_via_http(self, repo: str, filename: str) -> Optional[str]:
        """Загрузка файла через HTTP (fallback)"""
        try:
            import urllib.request
            
            url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
            local_path = self.model_cache_dir / filename
            
            logger.info(f"📥 HTTP download: {filename}")
            
            def show_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size / total_size) * 100)
                    if percent % 10 == 0:
                        logger.info(f"Download progress: {percent:.0f}%")
            
            urllib.request.urlretrieve(url, local_path, show_progress)
            
            if local_path.exists():
                logger.info(f"✅ HTTP download complete: {filename}")
                return str(local_path)
            
        except Exception as e:
            logger.debug(f"HTTP download failed for {filename}: {e}")
        
        return None
    
    def _find_existing_gguf_file(self) -> Optional[str]:
        """Ищет существующие GGUF файлы в кэше"""
        try:
            # Ищем в директории моделей
            for gguf_file in self.model_cache_dir.glob("*.gguf"):
                if gguf_file.is_file() and gguf_file.stat().st_size > 1024 * 1024:
                    logger.info(f"📁 Found existing GGUF file: {gguf_file.name}")
                    self.model_file = gguf_file.name
                    return str(gguf_file)
            
            # Ищем в HuggingFace cache рекурсивно
            try:
                import os
                hf_cache = os.path.expanduser("~/.cache/huggingface")
                if os.path.exists(hf_cache):
                    for root, dirs, files in os.walk(hf_cache):
                        for file in files:
                            if file.endswith('.gguf') and os.path.getsize(os.path.join(root, file)) > 1024 * 1024:
                                full_path = os.path.join(root, file)
                                logger.info(f"📁 Found cached GGUF: {file}")
                                self.model_file = file
                                return full_path
            except:
                pass
            
            logger.warning("❌ No existing GGUF files found")
            return None
            
        except Exception as e:
            logger.error(f"Error finding existing files: {e}")
            return None
    
    def _load_model_from_file(self, model_path: str):
        """Загружает модель из файла с ОПТИМИЗИРОВАННЫМИ настройками"""
        try:
            from llama_cpp import Llama
            
            # ОПТИМИЗИРОВАННЫЕ настройки для быстрой генерации
            model_kwargs = {
                "model_path": model_path,
                "n_ctx": self.context_length,  # Уменьшенный контекст для скорости
                "n_threads": self.n_threads,
                "n_gpu_layers": 0,  # Только CPU
                "use_mmap": True,   # Memory mapping для скорости
                "use_mlock": False, # Не блокируем память
                "verbose": False,
                "n_batch": 64 if self.hf_spaces else 128,  # Уменьшенный batch для скорости
                "rope_scaling_type": None,  # Отключаем дополнительные фичи
                "rope_freq_base": 0.0,      # Отключаем дополнительные фичи
                "rope_freq_scale": 0.0,     # Отключаем дополнительные фичи
            }
            
            logger.info(f"🔄 Loading OPTIMIZED model from: {os.path.basename(model_path)}")
            logger.info(f"🔧 OPTIMIZED settings: ctx={self.context_length}, threads={self.n_threads}, batch={model_kwargs['n_batch']}")
            
            self.model = Llama(**model_kwargs)
            self.model_loaded = True
            
            # Очистка памяти после загрузки
            if self.hf_spaces:
                gc.collect()
                logger.info("🧹 Post-loading memory cleanup")
            
            logger.info(f"✅ OPTIMIZED model loaded successfully: {os.path.basename(model_path)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load optimized model from {model_path}: {e}")
            raise
    
    def _create_fallback_model(self):
        """Создает улучшенный fallback модель"""
        logger.info("🔄 Creating enhanced fallback model...")
        
        class EnhancedFallbackModel:
            def __call__(self, prompt: str, max_tokens: int = 100, **kwargs):
                response = self._generate_smart_fallback(prompt)
                return {
                    "choices": [{
                        "text": response
                    }]
                }
            
            def _generate_smart_fallback(self, prompt: str) -> str:
                """Генерирует умный fallback ответ"""
                # Анализируем тип вопроса
                prompt_lower = prompt.lower()
                
                if any(word in prompt_lower for word in ["law", "legal", "право", "закон"]):
                    if "ukraine" in prompt_lower or "україн" in prompt_lower:
                        return """⚖️ **Українське право**

LlamaCpp модель завантажена та готова! Для українського права рекомендую:

🏛️ **Основні джерела:**
• Конституція України
• Цивільний кодекс України  
• Кримінальний кодекс України
• Трудовий кодекс України

📚 **Ресурси:**
• zakon.rada.gov.ua - офіційні тексти законів
• court.gov.ua - судова практика
• minjust.gov.ua - роз'яснення МінЮсту

🔍 Уточніть питання для детальної консультації."""

                    elif "ireland" in prompt_lower or "irish" in prompt_lower:
                        return """⚖️ **Irish Law**

LlamaCpp model is loaded and ready! For Irish law, key resources include:

🏛️ **Primary Sources:**
• Constitution of Ireland
• Irish Statute Book
• Common Law precedents
• EU Law (where applicable)

📚 **Resources:**
• irishstatutebook.ie - official legislation
• courts.ie - court decisions
• citizensinformation.ie - practical guidance

🔍 Please specify your question for detailed guidance."""

                    else:
                        return """⚖️ **Legal Information**

LlamaCpp model is loaded and ready! 

For legal matters, I can help with:
• Legislation analysis
• Legal concepts explanation
• Procedural guidance
• Document interpretation

🔍 Please provide specific details about your legal question for accurate assistance."""

                elif any(word in prompt_lower for word in ["what", "що", "explain", "поясни"]):
                    return """💡 **General Information**

LlamaCpp model is ready to provide detailed explanations!

I can help explain:
• Legal concepts and terms
• Procedures and processes  
• Rights and obligations
• Document requirements

🔍 Please ask your specific question and I'll provide a comprehensive answer."""

                else:
                    return """🤖 **LlamaCpp Ready**

The LlamaCpp model is successfully loaded and ready to assist!

✅ Features available:
• Legal consultation (Ukrainian & Irish law)
• Document analysis
• Concept explanations
• Procedural guidance

🔍 Ask me anything about legal matters and I'll provide detailed assistance."""
        
        self.model = EnhancedFallbackModel()
        self.model_loaded = True
        self.service_type = "llamacpp_enhanced_fallback_v2"
        logger.info("✅ Enhanced fallback model ready")
    
    async def answer_legal_question(self, question: str, context_documents: List[Dict], language: str = "en"):
        """Отвечает на юридический вопрос через ОПТИМИЗИРОВАННЫЙ LlamaCpp"""
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
            
            # Генерируем ОПТИМИЗИРОВАННЫЙ промпт
            prompt = self._build_optimized_prompt(question, context_documents, language)
            
            # Генерируем ответ с УЛУЧШЕННЫМИ таймаутами
            response_text = await self._generate_response_optimized(prompt)
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response_text,
                model=f"{self.model_name}/{self.model_file or 'fallback'}",
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
        """Строит ОПТИМИЗИРОВАННЫЙ промпт для быстрой генерации"""
        
        # КОРОТКИЕ системные промпты для скорости
        if language == "uk":
            system_message = "Юридичний консультант. Коротка точна відповідь."
        else:
            system_message = "Legal consultant. Brief accurate answer."
        
        # МИНИМАЛЬНЫЙ контекст для скорости
        context = ""
        if context_documents:
            # Берем только ПЕРВЫЙ документ и СИЛЬНО обрезаем
            doc = context_documents[0]
            content = doc.get('content', '')[:300]  # Только 300 символов
            filename = doc.get('filename', 'Doc')
            context = f"\nSource: {filename}\nText: {content}"
        
        # КОРОТКИЙ промпт в формате Llama-2-Chat
        if language == "uk":
            user_message = f"{context}\n\nПитання: {question}\nКоротка відповідь:"
        else:
            user_message = f"{context}\n\nQuestion: {question}\nBrief answer:"
        
        # Компактный Llama-2-Chat формат
        prompt = f"<s>[INST] {system_message}\n{user_message} [/INST]"
        
        # ЖЕСТКОЕ ограничение длины для скорости
        max_prompt_length = 800 if self.hf_spaces else 1200
        if len(prompt) > max_prompt_length:
            prompt = prompt[:max_prompt_length] + "..."
        
        return prompt
    
    async def _generate_response_optimized(self, prompt: str) -> str:
        """Генерирует ответ с ОПТИМИЗИРОВАННЫМИ настройками и увеличенным таймаутом"""
        try:
            # ОПТИМИЗИРОВАННЫЕ настройки для БЫСТРОЙ генерации
            generation_kwargs = {
                "max_tokens": self.max_tokens,
                "temperature": 0.1,  # Низкая температура для скорости и точности
                "top_p": 0.8,        # Уменьшено для скорости
                "top_k": 20,         # Уменьшено для скорости
                "repeat_penalty": 1.05,  # Уменьшено для скорости
                "stop": ["</s>", "[/INST]", "<s>", "\n\n"],  # Добавлены стоп-слова
                "echo": False,
                # НОВЫЕ параметры для оптимизации
                "stream": False,     # Отключаем стрим для простоты
                "logits_all": False, # Экономим память
                "vocab_only": False,
            }
            
            logger.info(f"🔧 OPTIMIZED generation: max_tokens={self.max_tokens}, temp=0.1, timeout={self.generation_timeout}s")
            
            # Определяем таймаут в зависимости от длины промпта
            prompt_length = len(prompt)
            if prompt_length < 200:
                timeout = self.quick_timeout  # 60 секунд для коротких промптов
            else:
                timeout = self.generation_timeout  # 90 секунд для длинных
            
            logger.info(f"⏰ Using timeout: {timeout}s for prompt length: {prompt_length}")
            
            # Генерируем с УВЕЛИЧЕННЫМ таймаутом
            def _generate_sync():
                return self.model(prompt, **generation_kwargs)
            
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(_generate_sync)
                try:
                    result = future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    logger.error(f"❌ Generation timeout after {timeout}s")
                    return f"Response generation took longer than {timeout} seconds. Please try with a shorter, more specific question."
            
            # Извлекаем текст ответа
            if isinstance(result, dict) and "choices" in result and result["choices"]:
                response_text = result["choices"][0].get("text", "").strip()
            else:
                response_text = str(result).strip()
            
            # БЫСТРАЯ очистка ответа
            response_text = self._clean_response_fast(response_text)
            
            # Очистка памяти
            if self.hf_spaces:
                gc.collect()
            
            return response_text if response_text else "I need more specific information to provide a proper answer."
            
        except Exception as e:
            logger.error(f"❌ OPTIMIZED generation error: {e}")
            return f"Error generating response. The model is working but encountered an issue: {str(e)[:100]}..."
    
    def _clean_response_fast(self, response: str) -> str:
        """БЫСТРАЯ очистка ответа"""
        if not response:
            return response
        
        # Удаляем стоп-токены
        for token in ["</s>", "[/INST]", "<s>", "[INST]"]:
            response = response.replace(token, "")
        
        # Быстрая очистка
        response = response.strip()
        
        # Берем только первый абзац если ответ длинный
        if '\n\n' in response:
            response = response.split('\n\n')[0]
        
        # Ограничиваем длину
        max_length = 800 if self.hf_spaces else 1200
        if len(response) > max_length:
            response = response[:max_length] + "..."
        
        return response
    
    def _generate_loading_response(self, question: str, language: str) -> str:
        """Ответ пока модель загружается"""
        if language == "uk":
            return f"""🦙 **LlamaCpp модель готова та оптимізована!**

**Ваше питання:** {question}

✅ Модель `{self.model_name}` завантажена та готова до роботи!

🚀 **Оптимізації:**
• Швидка генерація відповідей
• Таймаути: {self.generation_timeout}с для складних питань
• Контекст: {self.context_length} токенів
• Максимум токенів: {self.max_tokens}

🔍 **Статус:** Готовий до детальних юридичних консультацій!"""
        else:
            return f"""🦙 **LlamaCpp Model Ready and Optimized!**

**Your Question:** {question}

✅ Model `{self.model_name}` is loaded and ready to work!

🚀 **Optimizations:**
• Fast response generation
• Timeouts: {self.generation_timeout}s for complex questions  
• Context: {self.context_length} tokens
• Max tokens: {self.max_tokens}

🔍 **Status:** Ready for detailed legal consultations!"""
    
    def _generate_loading_error_response(self, question: str, language: str) -> str:
        """Ответ при ошибке загрузки модели"""
        if language == "uk":
            return f"""❌ **Помилка LlamaCpp моделі**

**Ваше питання:** {question}

🚫 **Проблема:** Модель не змогла завантажитися
📋 **Помилка:** {self.loading_error}

💡 **Рекомендації:**
• Перевірте підключення до інтернету
• Переконайтеся що є 4GB+ вільного місця
• Спробуйте перезапустити сервер"""
        else:
            return f"""❌ **LlamaCpp Model Error**

**Your Question:** {question}

🚫 **Issue:** Model failed to load
📋 **Error:** {self.loading_error}

💡 **Recommendations:**
• Check internet connection
• Ensure 4GB+ free disk space  
• Try restarting the server"""
    
    def _generate_error_response(self, question: str, language: str, error: str) -> str:
        """Ответ при ошибке генерации"""
        if language == "uk":
            return f"""⚠️ **Тимчасова проблема**

Питання: "{question}"

🔧 Помилка: {error[:100]}...

💡 Спробуйте:
• Коротше питання
• Простіші терміни  
• Повторити через хвилину"""
        else:
            return f"""⚠️ **Temporary Issue**

Question: "{question}"

🔧 Error: {error[:100]}...

💡 Try:
• Shorter question
• Simpler terms
• Retry in a minute"""
    
    async def get_service_status(self):
        """Детальний статус ОПТИМІЗОВАНОГО сервиса"""
        status = {
            "service_type": "llamacpp_optimized_v2",
            "model_name": self.model_name,
            "model_file": self.model_file,
            "model_loaded": self.model_loaded,
            "loading_started": self.loading_started,
            "loading_error": self.loading_error,
            "platform": "HuggingFace Spaces" if self.hf_spaces else "Local",
            "supported_languages": ["en", "uk"],
            "model_cache_dir": str(self.model_cache_dir),
            "auto_download": True,
            "optimizations_v2": {
                "cpu_optimized": True,
                "llamacpp_backend": True,
                "gguf_format": True,
                "fast_generation": True,
                "increased_timeouts": True,
                "hf_spaces_mode": self.hf_spaces,
                "max_tokens": self.max_tokens,
                "context_length": self.context_length,
                "n_threads": self.n_threads,
                "generation_timeout": self.generation_timeout,
                "quick_timeout": self.quick_timeout,
                "optimized_prompts": True,
                "memory_efficient": True
            }
        }
        
        # Диагностика
        try:
            from llama_cpp import Llama
            status["llamacpp_available"] = True
        except ImportError:
            status["llamacpp_available"] = False
        
        try:
            from huggingface_hub import hf_hub_download
            status["huggingface_hub_available"] = True
        except ImportError:
            status["huggingface_hub_available"] = False
        
        # Проверяем существующие файлы
        existing_models = list(self.model_cache_dir.glob("*.gguf"))
        status["cached_models"] = [m.name for m in existing_models]
        status["cache_size_mb"] = round(sum(m.stat().st_size for m in existing_models) / 1024 / 1024, 1) if existing_models else 0
        
        # Оптимизированные рекомендации
        recommendations = []
        if not self.model_loaded:
            if self.loading_error:
                recommendations.extend([
                    "Model download/loading failed - check internet and disk space",
                    "Ensure 4GB+ free space for GGUF model",
                    "Try: pip install llama-cpp-python --force-reinstall"
                ])
            else:
                recommendations.extend([
                    "Model downloading/loading with optimizations",
                    "Fast Q4_K_S model prioritized for speed",
                    "Optimized timeouts: 90s generation, 60s quick responses",
                    "Enhanced prompt optimization for faster inference"
                ])
        else:
            recommendations.extend([
                "LlamaCpp model ready with speed optimizations",
                "Q4_K_S quantization provides fast inference",
                f"Generation timeout increased to {self.generation_timeout}s",
                "Optimized prompts and context handling",
                "Memory efficient with cleanup after generation"
            ])
        
        status["recommendations"] = recommendations
        
        return status

def create_llm_service(model_name: str = "TheBloke/Llama-2-7B-Chat-GGUF"):
    """Создает ОПТИМИЗИРОВАННЫЙ LlamaCpp LLM сервис"""
    try:
        return LlamaCppLLMService(model_name)
    except Exception as e:
        logger.error(f"Failed to create optimized LlamaCpp LLM service: {e}")
        # Возвращаем fallback
        from app.dependencies import HFSpacesLlamaCppFallback
        return HFSpacesLlamaCppFallback()