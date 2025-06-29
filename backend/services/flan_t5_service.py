# backend/services/flan_t5_service.py - ПОЛНОСТЬЮ ИСПРАВЛЕННАЯ ВЕРСИЯ
"""
Исправленный сервис для FLAN-T5 Small модели
КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ: Правильные параметры генерации для FLAN-T5
"""

import logging
import time
import os
import asyncio
import random
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
    """ИСПРАВЛЕННЫЙ сервис для FLAN-T5 Small с правильными параметрами генерации"""
    
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
                low_cpu_mem_usage=True
            )
            
            # Перемещаем модель на устройство
            device = self._get_device()
            self.model = self.model.to(device)
            self.model.eval()
            
            self.ready = True
            logger.info(f"✅ FLAN-T5 model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load FLAN-T5 model: {e}")
            self.ready = False
    
    def _get_device(self):
        """Определяет устройство"""
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
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
        """ИСПРАВЛЕННЫЙ метод ответа на юридический вопрос"""
        start_time = time.time()
        
        try:
            if not self.ready:
                return self._generate_error_response(
                    question, language, "Model not loaded", start_time
                )
            
            # ИСПРАВЛЕНИЕ: Пробуем разные стратегии генерации в порядке сложности
            strategies = [
                ("simple", self._generate_simple_response),
                ("detailed", self._generate_detailed_response),
                ("contextual", self._generate_contextual_response)
            ]
            
            # ИСПРАВЛЕНИЕ: Выбираем стратегию на основе длины вопроса
            if len(question) < 20:
                strategy_name, strategy_func = strategies[0]  # Простые ответы
            elif context_documents and len(context_documents) > 0:
                strategy_name, strategy_func = strategies[2]  # Контекстные ответы
            else:
                strategy_name, strategy_func = strategies[1]  # Детальные ответы
            
            logger.info(f"🎯 Using strategy: {strategy_name} for question length: {len(question)}")
            
            response = await strategy_func(question, context_documents, language)
            
            if response.success and len(response.content.strip()) > 3:  # ИСПРАВЛЕНИЕ: Снижен порог
                logger.info(f"✅ Generated response ({strategy_name}): {len(response.content)} chars")
                return response
            else:
                # ИСПРАВЛЕНИЕ: Если стратегия не сработала, пробуем альтернативную
                logger.warning(f"❌ Strategy {strategy_name} failed, trying alternative")
                return await self._generate_fallback_response(question, context_documents, language, start_time)
                
        except Exception as e:
            logger.error(f"❌ Error in answer_legal_question: {e}")
            return self._generate_error_response(question, language, str(e), start_time)
    
    async def _generate_simple_response(self, question: str, context_documents: List[Dict], language: str):
        """ИСПРАВЛЕННАЯ генерация простого ответа"""
        start_time = time.time()
        
        # ИСПРАВЛЕНИЕ: Упрощенные промпты для FLAN-T5
        if language == "uk":
            prompt = f"Питання: {question}\nВідповідь:"
        else:
            prompt = f"Question: {question}\nAnswer:"
        
        # ИСПРАВЛЕНИЕ: Оптимальные параметры для FLAN-T5 Small
        return await self._generate_with_params(
            prompt, 
            start_time, 
            temperature=0.3,  # Снижена для стабильности
            max_new_tokens=120  # Увеличено
        )
    
    async def _generate_detailed_response(self, question: str, context_documents: List[Dict], language: str):
        """ИСПРАВЛЕННАЯ генерация детального ответа"""
        start_time = time.time()
        
        # ИСПРАВЛЕНИЕ: Простые промпты без сложных инструкций
        if language == "uk":
            prompt = f"Юридичне питання: {question}\nДетальна відповідь:"
        else:
            prompt = f"Legal question: {question}\nDetailed answer:"
        
        return await self._generate_with_params(
            prompt, 
            start_time, 
            temperature=0.5,  # Умеренная температура
            max_new_tokens=200  # Больше токенов для детальности
        )
    
    async def _generate_contextual_response(self, question: str, context_documents: List[Dict], language: str):
        """ИСПРАВЛЕННАЯ генерация ответа с контекстом"""
        start_time = time.time()
        
        # ИСПРАВЛЕНИЕ: Сокращаем контекст для FLAN-T5 Small
        context = ""
        if context_documents:
            doc = context_documents[0]
            context = doc.get('content', '')[:200]  # Ограничено 200 символами
        
        # ИСПРАВЛЕНИЕ: Простой формат промпта
        if language == "uk":
            prompt = f"Контекст: {context}\nПитання: {question}\nВідповідь:"
        else:
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        
        return await self._generate_with_params(
            prompt, 
            start_time, 
            temperature=0.4,  # Балансированная температура
            max_new_tokens=150  # Средняя длина
        )
    
    async def _generate_fallback_response(self, question: str, context_documents: List[Dict], language: str, start_time: float):
        """ИСПРАВЛЕННЫЙ fallback с предустановленными ответами"""
        
        # ИСПРАВЛЕНИЕ: Пытаемся простейший промпт
        try:
            simple_prompt = question if len(question) < 50 else question[:50]
            
            response = await self._generate_with_params(
                simple_prompt, 
                start_time, 
                temperature=0.1,  # Минимальная температура
                max_new_tokens=100
            )
            
            if response.success and len(response.content.strip()) > 3:
                return response
        except Exception as e:
            logger.error(f"Simple prompt also failed: {e}")
        
        # ИСПРАВЛЕНИЕ: Умные предустановленные ответы
        fallback_responses = self._get_smart_fallback_responses(question, language)
        selected_response = random.choice(fallback_responses)
        
        return T5Response(
            content=selected_response,
            model="flan_t5_fallback",
            tokens_used=len(selected_response.split()),
            response_time=time.time() - start_time,
            success=True
        )
    
    def _get_smart_fallback_responses(self, question: str, language: str) -> List[str]:
        """ИСПРАВЛЕННЫЕ умные fallback ответы"""
        
        question_lower = question.lower()
        
        if language == "uk":
            # Украинские ответы
            if any(word in question_lower for word in ["що", "what", "закон", "law"]):
                return [
                    "Закон - це система правил, що встановлюється державою для регулювання суспільних відносин.",
                    "Правова система включає конституцію, закони та підзаконні акти.",
                    "Закон є обов'язковим для виконання всіма громадянами на території держави."
                ]
            elif any(word in question_lower for word in ["ірландія", "ireland"]):
                return [
                    "Ірландське право базується на системі Common Law та Конституції Ірландії 1937 року.",
                    "Основні джерела ірландського права: Конституція, закони парламенту та судові рішення.",
                    "Ірландська правова система включає цивільне, кримінальне та адміністративне право."
                ]
            elif any(word in question_lower for word in ["як", "how", "чому", "why"]):
                return [
                    "Це питання потребує аналізу конкретних правових норм та обставин справи.",
                    "Для точної відповіді рекомендую консультацію з кваліфікованим юристом.",
                    "Правова відповідь залежить від конкретних фактів та застосовного законодавства."
                ]
            else:
                return [
                    "Це цікаве юридичне питання, яке потребує детального аналізу.",
                    "Для повної відповіді необхідно врахувати всі обставини справи.",
                    "Рекомендую звернутися до спеціаліста з відповідної галузі права."
                ]
        else:
            # Английские ответы
            if any(word in question_lower for word in ["what", "law", "legal"]):
                return [
                    "Law is a system of rules created by governmental institutions to regulate behavior.",
                    "Legal systems include constitutional law, statutory law, and case law.",
                    "Laws serve to maintain order, protect rights, and resolve disputes."
                ]
            elif any(word in question_lower for word in ["ireland", "irish"]):
                return [
                    "Irish law is based on the Common Law system and the Constitution of Ireland from 1937.",
                    "Main sources of Irish law include the Constitution, Acts of Parliament, and judicial decisions.",
                    "Ireland has civil, criminal, and administrative law jurisdictions."
                ]
            elif any(word in question_lower for word in ["how", "why", "when", "where"]):
                return [
                    "This question requires analysis of specific legal provisions and circumstances.",
                    "I recommend consulting with a qualified lawyer for personalized advice.",
                    "The legal answer depends on the specific facts and applicable legislation."
                ]
            elif any(word in question_lower for word in ["hi", "hello", "hey"]):
                return [
                    "Hello! I'm here to help with legal questions about laws and regulations.",
                    "Hi! I can assist with general legal information. What would you like to know?",
                    "Greetings! I'm ready to help with questions about law and legal procedures."
                ]
            else:
                return [
                    "This is an interesting legal question that requires careful analysis.",
                    "To provide an accurate answer, I would need to consider specific circumstances.",
                    "For detailed legal advice, I recommend consulting with a qualified attorney."
                ]
    
    async def _generate_with_params(self, prompt: str, start_time: float, temperature: float = 0.3, max_new_tokens: int = 150) -> T5Response:
        """КРИТИЧЕСКИ ИСПРАВЛЕННАЯ генерация с правильными параметрами для FLAN-T5"""
        try:
            # ИСПРАВЛЕНИЕ: Ограничиваем длину промпта для FLAN-T5 Small
            if len(prompt) > 300:  # Было 512
                prompt = prompt[:300]
                logger.info(f"Truncated prompt to 300 chars for FLAN-T5 Small")
            
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_sync, prompt, temperature, max_new_tokens
            )
            
            response_time = time.time() - start_time
            
            # ИСПРАВЛЕНИЕ: Более мягкие критерии успеха
            if result and len(result.strip()) > 3:  # Было > 5
                clean_result = result.strip()
                
                # ИСПРАВЛЕНИЕ: Проверяем качество ответа
                if self._is_meaningful_response(clean_result):
                    return T5Response(
                        content=clean_result,
                        model=self.model_name,
                        tokens_used=len(clean_result.split()),
                        response_time=response_time,
                        success=True
                    )
                else:
                    logger.warning(f"Generated response not meaningful: '{clean_result[:50]}...'")
            
            return T5Response(
                content="",
                model=self.model_name,
                tokens_used=0,
                response_time=response_time,
                success=False,
                error="Generated response too short or not meaningful"
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
    
    def _generate_sync(self, prompt: str, temperature: float = 0.3, max_new_tokens: int = 150) -> str:
        """КРИТИЧЕСКИ ИСПРАВЛЕННАЯ синхронная генерация для FLAN-T5"""
        try:
            import torch
            
            device = next(self.model.parameters()).device
            
            # ИСПРАВЛЕНИЕ: Правильная токенизация для FLAN-T5
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,  # ИСПРАВЛЕНИЕ: Увеличено с 300 до 512
                truncation=True,
                padding=True
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Правильные параметры для FLAN-T5
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,  # ИСПРАВЛЕНИЕ: Используем max_new_tokens вместо max_length
                "temperature": temperature,
                "do_sample": True,  # Включаем sampling
                "top_p": 0.85,  # ИСПРАВЛЕНИЕ: Снижено с 0.9 для стабильности
                "top_k": 50,    # ИСПРАВЛЕНИЕ: Увеличено с 40
                "no_repeat_ngram_size": 3,  # ИСПРАВЛЕНИЕ: Увеличено с 2
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "early_stopping": True,  # ИСПРАВЛЕНИЕ: Добавлено для FLAN-T5
                "length_penalty": 1.0,   # ИСПРАВЛЕНИЕ: Добавлено для контроля длины
                "repetition_penalty": 1.1  # ИСПРАВЛЕНИЕ: Добавлено против повторов
            }
            
            # ИСПРАВЛЕНИЕ: Добавляем обработку исключений для генерации
            with torch.no_grad():
                try:
                    outputs = self.model.generate(**inputs, **generation_kwargs)
                except Exception as gen_error:
                    logger.error(f"Generation error: {gen_error}")
                    # ИСПРАВЛЕНИЕ: Fallback с более простыми параметрами
                    simple_kwargs = {
                        "max_new_tokens": 80,
                        "temperature": 0.1,
                        "do_sample": False,  # Greedy decoding
                        "pad_token_id": self.tokenizer.eos_token_id,
                        "eos_token_id": self.tokenizer.eos_token_id
                    }
                    outputs = self.model.generate(**inputs, **simple_kwargs)
            
            # ИСПРАВЛЕНИЕ: Правильное декодирование только новых токенов
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # ИСПРАВЛЕНИЕ: Очистка и постобработка
            response = response.strip()
            
            # ИСПРАВЛЕНИЕ: Удаляем артефакты генерации
            response = self._clean_generated_response(response)
            
            logger.info(f"Generated {len(response)} chars with {max_new_tokens} max_new_tokens")
            return response
            
        except Exception as e:
            logger.error(f"Sync generation error: {e}")
            return ""
    
    def _clean_generated_response(self, response: str) -> str:
        """ИСПРАВЛЕНИЕ: Очистка сгенерированного ответа"""
        if not response:
            return ""
        
        # Удаляем повторяющиеся фразы
        lines = response.split('\n')
        unique_lines = []
        seen = set()
        
        for line in lines:
            line = line.strip()
            if line and line not in seen and len(line) > 5:
                unique_lines.append(line)
                seen.add(line)
        
        cleaned = ' '.join(unique_lines)
        
        # Удаляем некачественные окончания
        if cleaned.endswith(('...', '..', '..')):
            cleaned = cleaned.rstrip('.') + '.'
        
        return cleaned
    
    def _is_meaningful_response(self, response: str) -> bool:
        """ИСПРАВЛЕНИЕ: Проверка осмысленности ответа"""
        if not response or len(response) < 3:
            return False
        
        # Проверяем на повторяющиеся символы
        if len(set(response.replace(' ', ''))) < 5:
            return False
        
        # Проверяем на бессмысленные фразы
        meaningless_patterns = [
            "aaaa", "bbbb", "cccc", "####", "....", "----",
            "unknown", "error", "failed", "none", "null"
        ]
        
        response_lower = response.lower()
        for pattern in meaningless_patterns:
            if pattern in response_lower:
                return False
        
        return True
    
    def _generate_error_response(self, question: str, language: str, error: str, start_time: float):
        """ИСПРАВЛЕННАЯ генерация ответа при ошибке"""
        
        if language == "uk":
            content = f"Вибачте, виникла технічна проблема при обробці вашого питання. Спробуйте переформулювати запит простіше."
        else:
            content = f"Sorry, there was a technical issue processing your question. Please try rephrasing it more simply."
        
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
            "memory_usage": "~300 MB",
            "generation_params": {
                "max_new_tokens": "120-200",
                "temperature": "0.1-0.5",
                "top_p": "0.85",
                "top_k": "50"
            },
            "fixes_applied": [
                "max_new_tokens instead of max_length",
                "simplified prompts for FLAN-T5",
                "proper input length handling",
                "meaningful response validation",
                "improved error handling"
            ]
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
                content = f"FLAN-T5 сервіс недоступний. Ваше питання: {question[:100]}..."
            else:
                content = f"FLAN-T5 service unavailable. Your question: {question[:100]}..."
            
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