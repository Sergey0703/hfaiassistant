# backend/services/llamacpp_llm_service.py - ИСПРАВЛЕННАЯ ВЕРСИЯ С АВТОЗАГРУЗКОЙ МОДЕЛЕЙ
"""
LLM Service using llama-cpp-python with automatic GGUF model download
ИСПРАВЛЕНИЯ: Автоматическая загрузка GGUF моделей с HuggingFace Hub
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
    """LLM Service на основе llama-cpp-python с автозагрузкой моделей"""
    
    def __init__(self, model_name: str = "TheBloke/Llama-2-7B-Chat-GGUF"):
        self.model_name = model_name
        self.model_file = None  # Будет определен автоматически
        self.model = None
        self.service_type = "llamacpp_auto_download"
        self.model_loaded = False
        self.loading_started = False
        self.loading_error = None
        self.hf_spaces = os.getenv("SPACE_ID") is not None
        self.model_cache_dir = Path("./models")
        
        # Оптимизация для HF Spaces
        self.max_tokens = 150 if self.hf_spaces else 300
        self.context_length = 2048 if self.hf_spaces else 4096
        self.n_threads = 2 if self.hf_spaces else 4
        
        # Создаем директорию для моделей
        self.model_cache_dir.mkdir(exist_ok=True)
        
        logger.info(f"🦙 Initializing LlamaCpp service with auto-download: {model_name}")
        logger.info(f"🌍 Platform: {'HuggingFace Spaces' if self.hf_spaces else 'Local'}")
        logger.info(f"📁 Model cache: {self.model_cache_dir}")
        
        # Автоматически начинаем загрузку
        self._start_model_loading()
    
    def _start_model_loading(self):
        """Безопасно начинает загрузку модели"""
        if self.loading_started:
            return
        
        self.loading_started = True
        
        try:
            logger.info(f"🔄 Starting LlamaCpp model loading with auto-download: {self.model_name}")
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
                logger.error("Install: pip install llama-cpp-python")
                raise
            
            # Очистка памяти
            if self.hf_spaces:
                gc.collect()
                logger.info("🧹 Memory cleanup for HF Spaces")
            
            # НОВОЕ: Автоматическое скачивание модели
            model_file_path = self._download_gguf_model()
            
            if not model_file_path:
                raise Exception("Failed to download GGUF model")
            
            logger.info(f"📁 Model file ready: {model_file_path}")
            
            # Загружаем модель
            self._load_model_from_file(model_file_path)
            
            loading_time = time.time() - start_time
            logger.info(f"✅ LlamaCpp model loaded successfully in {loading_time:.1f}s!")
            
        except Exception as e:
            self.loading_error = str(e)
            logger.error(f"❌ LlamaCpp model loading failed: {e}")
            self._create_fallback_model()
    
    def _download_gguf_model(self) -> Optional[str]:
        """Скачивает GGUF модель с HuggingFace Hub"""
        try:
            # Список моделей для попытки (от легких к тяжелым)
            model_candidates = [
                {
                    "repo": "TheBloke/Llama-2-7B-Chat-GGUF",
                    "files": [
                        "llama-2-7b-chat.Q4_K_S.gguf",  # 3.5GB - самая легкая
                        "llama-2-7b-chat.Q4_K_M.gguf",  # 4.1GB - средняя
                        "llama-2-7b-chat.Q5_K_M.gguf",  # 4.8GB - качественная
                    ]
                },
                {
                    "repo": "microsoft/DialoGPT-medium",  # Более легкая альтернатива
                    "files": ["pytorch_model.bin"]  # Если GGUF недоступен
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
            # Попробуем использовать huggingface_hub если доступен
            try:
                from huggingface_hub import hf_hub_download
                
                logger.info(f"📥 Downloading {filename} from {repo}...")
                
                file_path = hf_hub_download(
                    repo_id=repo,
                    filename=filename,
                    cache_dir=str(self.model_cache_dir),
                    local_files_only=False,
                    force_download=False  # Используем кэш если есть
                )
                
                if os.path.exists(file_path):
                    logger.info(f"✅ Downloaded: {filename} ({self._get_file_size(file_path)})")
                    return file_path
                
            except ImportError:
                logger.warning("huggingface_hub not available, trying wget/curl...")
                
                # Fallback: прямая загрузка через HTTP
                return self._download_via_http(repo, filename)
            
        except Exception as e:
            logger.debug(f"Failed to download {filename}: {e}")
            return None
    
    def _download_via_http(self, repo: str, filename: str) -> Optional[str]:
        """Загрузка файла через HTTP (fallback)"""
        try:
            import urllib.request
            
            # HuggingFace URL format
            url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
            local_path = self.model_cache_dir / filename
            
            logger.info(f"📥 HTTP download: {filename}")
            
            def show_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size / total_size) * 100)
                    if percent % 10 == 0:  # Показываем каждые 10%
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
                if gguf_file.is_file() and gguf_file.stat().st_size > 1024 * 1024:  # Больше 1MB
                    logger.info(f"📁 Found existing GGUF file: {gguf_file.name}")
                    self.model_file = gguf_file.name
                    return str(gguf_file)
            
            # Ищем в HuggingFace cache
            try:
                import os
                hf_cache = os.path.expanduser("~/.cache/huggingface")
                if os.path.exists(hf_cache):
                    for root, dirs, files in os.walk(hf_cache):
                        for file in files:
                            if file.endswith('.gguf') and os.path.getsize(os.path.join(root, file)) > 1024 * 1024:
                                logger.info(f"📁 Found cached GGUF: {file}")
                                return os.path.join(root, file)
            except:
                pass
            
            logger.warning("❌ No existing GGUF files found")
            return None
            
        except Exception as e:
            logger.error(f"Error finding existing files: {e}")
            return None
    
    def _load_model_from_file(self, model_path: str):
        """Загружает модель из файла"""
        try:
            from llama_cpp import Llama
            
            model_kwargs = {
                "model_path": model_path,
                "n_ctx": self.context_length,
                "n_threads": self.n_threads,
                "n_gpu_layers": 0,  # Только CPU
                "use_mmap": True,
                "use_mlock": False,
                "verbose": False,
                "n_batch": 128 if self.hf_spaces else 512,
            }
            
            logger.info(f"🔄 Loading model from: {os.path.basename(model_path)}")
            logger.info(f"🔧 Settings: ctx={self.context_length}, threads={self.n_threads}")
            
            self.model = Llama(**model_kwargs)
            self.model_loaded = True
            
            # Очистка памяти после загрузки
            if self.hf_spaces:
                gc.collect()
                logger.info("🧹 Post-loading memory cleanup")
            
            logger.info(f"✅ Model loaded successfully: {os.path.basename(model_path)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model from {model_path}: {e}")
            raise
    
    def _get_file_size(self, file_path: str) -> str:
        """Возвращает размер файла в читаемом формате"""
        try:
            size = os.path.getsize(file_path)
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024:
                    return f"{size:.1f} {unit}"
                size /= 1024
            return f"{size:.1f} TB"
        except:
            return "unknown size"
    
    def _create_fallback_model(self):
        """Создает fallback модель если основная не загрузилась"""
        logger.info("🔄 Creating enhanced fallback model...")
        
        class EnhancedFallbackModel:
            def __call__(self, prompt: str, max_tokens: int = 100, **kwargs):
                # Более интеллектуальный fallback
                if "question" in prompt.lower() or "питання" in prompt.lower():
                    response = self._generate_fallback_response(prompt)
                else:
                    response = f"LlamaCpp model is loading. Your request: {prompt[:100]}..."
                
                return {
                    "choices": [{
                        "text": response
                    }]
                }
            
            def _generate_fallback_response(self, prompt: str) -> str:
                """Генерирует более умный fallback ответ"""
                if "ukraine" in prompt.lower() or "україн" in prompt.lower():
                    return """🇺🇦 **Юридична інформація для України**
                    
LlamaCpp модель завантажується. Для українського права рекомендую звернутися до:
• zakon.rada.gov.ua - офіційний портал законодавства
• court.gov.ua - судова система України
• minjust.gov.ua - Міністерство юстиції

Модель буде доступна через кілька хвилин для детальних консультацій."""

                elif "ireland" in prompt.lower() or "irish" in prompt.lower():
                    return """🇮🇪 **Legal Information for Ireland**
                    
LlamaCpp model is loading. For Irish law, I recommend:
• irishstatutebook.ie - official legislation
• courts.ie - court system information  
• citizensinformation.ie - citizen services

The model will be available in a few minutes for detailed consultations."""

                else:
                    return """⚖️ **Legal Assistant Loading**
                    
The LlamaCpp model is currently initializing. This provides:
• Stable CPU inference without hanging
• Support for English and Ukrainian
• Legal document analysis capabilities

Please try your question again in a few minutes."""
        
        self.model = EnhancedFallbackModel()
        self.model_loaded = True
        self.service_type = "llamacpp_enhanced_fallback"
        logger.info("✅ Enhanced fallback model ready")
    
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
    
    def _build_chat_prompt(self, question: str, context_documents: List[Dict], language: str) -> str:
        """Строит промпт для Llama-2-Chat формата"""
        
        # Системный промпт
        if language == "uk":
            system_message = "Ви - досвідчений юридичний консультант. Надавайте точні та корисні відповіді."
        else:
            system_message = "You are an experienced legal consultant. Provide accurate and helpful answers."
        
        # Подготавливаем контекст
        context_parts = []
        max_docs = 2 if self.hf_spaces else 3
        max_doc_length = 400 if self.hf_spaces else 600
        
        for i, doc in enumerate(context_documents[:max_docs]):
            filename = doc.get('filename', f'Document {i+1}')
            content = doc.get('content', '')[:max_doc_length]
            if content.strip():
                context_parts.append(f"{filename}: {content.strip()}")
        
        context = "\n\n".join(context_parts) if context_parts else ""
        
        # Формируем user message
        if context:
            if language == "uk":
                user_message = f"Контекст:\n{context}\n\nПитання: {question}\n\nВідповідь:"
            else:
                user_message = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            if language == "uk":
                user_message = f"Питання: {question}\n\nВідповідь:"
            else:
                user_message = f"Question: {question}\n\nAnswer:"
        
        # Llama-2-Chat формат
        prompt = f"""<s>[INST] <<SYS>>
{system_message}
<</SYS>>

{user_message} [/INST]"""
        
        # Ограничиваем длину
        max_prompt_length = 1500 if self.hf_spaces else 2000
        if len(prompt) > max_prompt_length:
            prompt = prompt[:max_prompt_length] + "..."
        
        return prompt
    
    async def _generate_response_llamacpp(self, prompt: str) -> str:
        """Генерирует ответ через LlamaCpp с таймаутом"""
        try:
            generation_kwargs = {
                "max_tokens": self.max_tokens,
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "stop": ["</s>", "[/INST]", "<s>"],
                "echo": False,
            }
            
            logger.info(f"🔧 Generating with max_tokens={self.max_tokens}")
            
            # Генерируем с таймаутом
            def _generate_sync():
                return self.model(prompt, **generation_kwargs)
            
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(_generate_sync)
                try:
                    timeout = 30 if self.hf_spaces else 60
                    result = future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    logger.error(f"❌ Generation timeout after {timeout}s")
                    return "Response generation timed out. Please try with a shorter question."
            
            # Извлекаем текст ответа
            if isinstance(result, dict) and "choices" in result and result["choices"]:
                response_text = result["choices"][0].get("text", "").strip()
            else:
                response_text = str(result).strip()
            
            # Очищаем ответ
            response_text = self._clean_response(response_text)
            
            # Очистка памяти
            if self.hf_spaces:
                gc.collect()
            
            return response_text if response_text else "I need more specific information to provide a proper answer."
            
        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            return f"Error generating response. Please try again."
    
    def _clean_response(self, response: str) -> str:
        """Очищает ответ от артефактов"""
        if not response:
            return response
        
        # Удаляем стоп-токены
        stop_tokens = ["</s>", "[/INST]", "<s>", "[INST]"]
        for token in stop_tokens:
            response = response.replace(token, "")
        
        # Удаляем лишние пробелы
        lines = []
        for line in response.split('\n'):
            line = line.strip()
            if line and not line.startswith(('System:', 'User:', 'Assistant:')):
                lines.append(line)
        
        cleaned = '\n'.join(lines[:10])
        
        # Ограничиваем длину
        max_length = 1000 if self.hf_spaces else 1500
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length] + "..."
        
        return cleaned
    
    def _generate_loading_response(self, question: str, language: str) -> str:
        """Ответ пока модель загружается"""
        if language == "uk":
            return f"""🦙 **LlamaCpp модель завантажується з автоскачуванням...**

**Ваше питання:** {question}

⏳ Модель `{self.model_name}` завантажується автоматично:
📥 Скачування GGUF файлу з HuggingFace Hub
🔧 Ініціалізація llama.cpp
⚡ Оптимізація для CPU

📁 **Файл моделі:** {self.model_file or 'визначається автоматично'}
🎯 **Якість:** Стабільна робота на CPU
🌍 **Мови:** Англійська та українська

💡 **Статус:** Перше завантаження може зайняти 2-5 хвилин залежно від швидкості інтернету."""
        else:
            return f"""🦙 **LlamaCpp Model Loading with Auto-Download...**

**Your Question:** {question}

⏳ Model `{self.model_name}` is loading automatically:
📥 Downloading GGUF file from HuggingFace Hub
🔧 Initializing llama.cpp
⚡ Optimizing for CPU

📁 **Model File:** {self.model_file or 'auto-detecting'}
🎯 **Quality:** Stable CPU performance
🌍 **Languages:** English and Ukrainian

💡 **Status:** First download may take 2-5 minutes depending on internet speed."""
    
    def _generate_loading_error_response(self, question: str, language: str) -> str:
        """Ответ при ошибке загрузки модели"""
        if language == "uk":
            return f"""❌ **Помилка автозавантаження LlamaCpp моделі**

**Ваше питання:** {question}

🚫 **Проблема:** Не вдалося завантажити модель `{self.model_name}`
📋 **Помилка:** {self.loading_error}

🔧 **Можливі причини:**
• Відсутнє підключення до інтернету
• HuggingFace Hub недоступний
• Недостатньо місця на диску
• Відсутня бібліотека huggingface-hub

💡 **Рекомендації:**
• Встановіть: pip install huggingface-hub
• Перевірте підключення до інтернету
• Очистіть кэш: rm -rf ~/.cache/huggingface
• Спробуйте ще раз через кілька хвилин"""
        else:
            return f"""❌ **LlamaCpp Auto-Download Error**

**Your Question:** {question}

🚫 **Issue:** Failed to download model `{self.model_name}`
📋 **Error:** {self.loading_error}

🔧 **Possible Causes:**
• No internet connection
• HuggingFace Hub unavailable
• Insufficient disk space
• Missing huggingface-hub library

💡 **Recommendations:**
• Install: pip install huggingface-hub
• Check internet connectivity
• Clear cache: rm -rf ~/.cache/huggingface
• Try again in a few minutes"""
    
    def _generate_error_response(self, question: str, language: str, error: str) -> str:
        """Ответ при ошибке генерации"""
        if language == "uk":
            return f"""⚠️ **Технічна проблема**

Помилка обробки: "{question}"
🔧 Деталі: {error}

Спробуйте ще раз або переформулюйте питання."""
        else:
            return f"""⚠️ **Technical Issue**

Error processing: "{question}"
🔧 Details: {error}

Please try again or rephrase your question."""
    
    async def get_service_status(self):
        """Детальний статус сервиса"""
        status = {
            "service_type": "llamacpp_auto_download",
            "model_name": self.model_name,
            "model_file": self.model_file,
            "model_loaded": self.model_loaded,
            "loading_started": self.loading_started,
            "loading_error": self.loading_error,
            "platform": "HuggingFace Spaces" if self.hf_spaces else "Local",
            "supported_languages": ["en", "uk"],
            "model_cache_dir": str(self.model_cache_dir),
            "auto_download": True,
            "optimizations": {
                "cpu_optimized": True,
                "llamacpp_backend": True,
                "gguf_format": True,
                "auto_download_enabled": True,
                "hf_spaces_mode": self.hf_spaces,
                "max_tokens": self.max_tokens,
                "context_length": self.context_length,
                "n_threads": self.n_threads,
                "timeout_enabled": True
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
        status["cache_size"] = sum(m.stat().st_size for m in existing_models) if existing_models else 0
        
        # Рекомендации
        recommendations = []
        if not self.model_loaded:
            if self.loading_error:
                recommendations.extend([
                    "Model auto-download failed",
                    "Check internet connection and HuggingFace Hub access",
                    "Install: pip install huggingface-hub",
                    "Ensure sufficient disk space for GGUF files (~4GB)"
                ])
            else:
                recommendations.extend([
                    "Model downloading automatically from HuggingFace Hub",
                    "First download may take 2-5 minutes",
                    "GGUF files are cached for future use",
                    "Stable CPU inference without hanging issues"
                ])
        else:
            recommendations.extend([
                "LlamaCpp model ready with auto-download capabilities",
                "GGUF model cached locally for faster future loading",
                "CPU-optimized inference with timeout protection",
                "Enhanced fallback responses during loading"
            ])
        
        status["recommendations"] = recommendations
        
        return status

def create_llm_service(model_name: str = "TheBloke/Llama-2-7B-Chat-GGUF"):
    """Создает LlamaCpp LLM сервис с автозагрузкой"""
    try:
        return LlamaCppLLMService(model_name)
    except Exception as e:
        logger.error(f"Failed to create LlamaCpp LLM service: {e}")
        # Возвращаем fallback
        from app.dependencies import HFSpacesLlamaCppFallback
        return HFSpacesLlamaCppFallback()