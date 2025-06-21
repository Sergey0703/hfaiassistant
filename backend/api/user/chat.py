# backend/api/user/chat.py - ИСПРАВЛЕНИЯ ДЛЯ ТАЙМАУТОВ И ASYNC

"""
User Chat Endpoints - Пользовательские endpoints для чата с КРИТИЧЕСКИМИ ИСПРАВЛЕНИЯМИ
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import logging
import time
import asyncio

from models.requests import ChatMessage, ChatHistoryRequest
from models.responses import ChatResponse, ChatHistoryResponse, ChatHistoryItem
from app.dependencies import get_document_service, get_llm_service
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Временное хранение для истории чатов
chat_history: List[Dict[str, Any]] = []

# КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Глобальные таймауты
SEARCH_TIMEOUT = 10.0    # 10 секунд на поиск
LLM_TIMEOUT = 120.0      # 2 минуты на LLM
TOTAL_TIMEOUT = 180.0    # 3 минуты общий таймаут

@router.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(
    message: ChatMessage,
    document_service = Depends(get_document_service),
    llm_service = Depends(get_llm_service)
):
    """Основной endpoint для чата с юридическим ассистентом С ТАЙМАУТАМИ"""
    
    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Оборачиваем весь endpoint в таймаут
    try:
        return await asyncio.wait_for(
            _process_chat_message(message, document_service, llm_service),
            timeout=TOTAL_TIMEOUT
        )
    except asyncio.TimeoutError:
        logger.error(f"❌ Chat request timeout after {TOTAL_TIMEOUT}s")
        
        # Сохраняем в историю даже при таймауте
        timeout_entry = {
            "message": message.message,
            "response": "⏰ Request timeout - please try with a shorter question",
            "language": message.language,
            "sources": [],
            "timestamp": time.time(),
            "timeout": True,
            "search_stats": {"timeout": True},
            "ai_stats": {"timeout": True}
        }
        chat_history.append(timeout_entry)
        
        if language := message.language == "uk":
            timeout_response = "⏰ Запит перевищив час очікування. Спробуйте коротше питання."
        else:
            timeout_response = "⏰ Request timeout. Please try with a shorter question."
        
        return ChatResponse(
            response=timeout_response,
            sources=[]
        )
    except Exception as e:
        logger.error(f"❌ Critical chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat service error: {str(e)}")

async def _process_chat_message(message: ChatMessage, document_service, llm_service):
    """Обрабатывает сообщение чата с контролем времени"""
    
    search_results = []
    sources = []
    search_start_time = time.time()
    
    # ====================================
    # ЭТАП 1: ПОИСК РЕЛЕВАНТНЫХ ДОКУМЕНТОВ С ТАЙМАУТОМ
    # ====================================
    try:
        logger.info(f"🔍 Starting search for: '{message.message[:50]}...'")
        
        # ИСПРАВЛЕНИЕ: Поиск с таймаутом
        search_results = await asyncio.wait_for(
            document_service.search(
                query=message.message,
                limit=settings.MAX_CONTEXT_DOCUMENTS,
                min_relevance=0.3
            ),
            timeout=SEARCH_TIMEOUT
        )
        
        search_time = time.time() - search_start_time
        
        # ИСПРАВЛЕНИЕ: Правильная проверка результатов
        if search_results and isinstance(search_results, list) and len(search_results) > 0:
            sources = [result.get('filename', 'Unknown') for result in search_results]
            logger.info(f"✅ Search completed: {len(search_results)} results in {search_time:.2f}s")
        else:
            logger.info(f"ℹ️ No relevant documents found in {search_time:.2f}s")
            search_results = []
        
    except asyncio.TimeoutError:
        search_time = time.time() - search_start_time
        logger.error(f"❌ Search timeout after {SEARCH_TIMEOUT}s")
        search_results = []
        
    except Exception as e:
        search_time = time.time() - search_start_time
        logger.error(f"❌ Search error after {search_time:.2f}s: {e}")
        search_results = []
    
    # ====================================
    # ЭТАП 2: ГЕНЕРАЦИЯ AI ОТВЕТА С ТАЙМАУТОМ
    # ====================================
    ai_response = None
    response_text = ""
    llm_start_time = time.time()
    
    if search_results and len(search_results) > 0:
        try:
            logger.info("🤖 Generating AI response based on found documents...")
            
            # Подготавливаем контекст для LLM
            context_documents = []
            for result in search_results:
                # Ограничиваем длину каждого документа
                content = result.get('content', '')
                if len(content) > settings.CONTEXT_TRUNCATE_LENGTH:
                    content = content[:settings.CONTEXT_TRUNCATE_LENGTH] + "..."
                
                context_doc = {
                    "filename": result.get('filename', 'Unknown'),
                    "content": content,
                    "relevance_score": result.get('relevance_score', 0.0),
                    "metadata": result.get('metadata', {})
                }
                context_documents.append(context_doc)
            
            # ИСПРАВЛЕНИЕ: Генерируем ответ через LLM с таймаутом
            ai_response = await asyncio.wait_for(
                llm_service.answer_legal_question(
                    question=message.message,
                    context_documents=context_documents,
                    language=message.language
                ),
                timeout=LLM_TIMEOUT
            )
            
            llm_time = time.time() - llm_start_time
            
            if ai_response and ai_response.success and ai_response.content.strip():
                # Используем AI ответ
                response_text = ai_response.content.strip()
                
                logger.info(f"✅ AI response generated: {len(response_text)} chars, "
                          f"{ai_response.tokens_used} tokens, {llm_time:.2f}s")
            else:
                # AI не смог ответить, используем fallback
                logger.warning(f"AI response failed: {getattr(ai_response, 'error', 'Unknown error')}")
                response_text = _generate_fallback_response_with_context(
                    message.message, search_results, message.language, 
                    getattr(ai_response, 'error', 'AI response failed')
                )
                    
        except asyncio.TimeoutError:
            llm_time = time.time() - llm_start_time
            logger.error(f"❌ LLM timeout after {LLM_TIMEOUT}s")
            # Fallback если AI недоступен из-за таймаута
            response_text = _generate_timeout_fallback_response(
                message.message, search_results, message.language
            )
            
        except Exception as e:
            llm_time = time.time() - llm_start_time
            logger.error(f"❌ LLM generation error after {llm_time:.2f}s: {e}")
            # Fallback если AI полностью недоступен
            response_text = _generate_fallback_response_with_context(
                message.message, search_results, message.language, str(e)
            )
    else:
        # Нет релевантных документов - генерируем ответ об отсутствии информации
        try:
            response_text = await asyncio.wait_for(
                _generate_no_context_response(message.message, message.language, llm_service),
                timeout=LLM_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(f"❌ No-context LLM timeout after {LLM_TIMEOUT}s")
            response_text = _generate_no_documents_found_response(message.message, message.language)
        except Exception as e:
            logger.error(f"❌ No-context LLM error: {e}")
            response_text = _generate_no_documents_found_response(message.message, message.language)
    
    # ====================================
    # ЭТАП 3: СОХРАНЕНИЕ В ИСТОРИЮ
    # ====================================
    chat_entry = {
        "message": message.message,
        "response": response_text,
        "language": message.language,
        "sources": sources,
        "timestamp": time.time(),
        "search_stats": {
            "found_documents": len(search_results),
            "has_relevant_results": len(search_results) > 0,
            "search_query": message.message,
            "search_time": search_time if 'search_time' in locals() else 0.0
        },
        "ai_stats": {
            "ai_used": ai_response is not None and getattr(ai_response, 'success', False),
            "model": getattr(ai_response, 'model', 'fallback'),
            "tokens_used": getattr(ai_response, 'tokens_used', 0),
            "response_time": getattr(ai_response, 'response_time', 0.0),
            "error": getattr(ai_response, 'error', None) if ai_response and not getattr(ai_response, 'success', False) else None,
            "llm_time": llm_time if 'llm_time' in locals() else 0.0
        }
    }
    chat_history.append(chat_entry)
    
    # Ограничиваем историю последними 100 сообщениями
    if len(chat_history) > 100:
        chat_history.pop(0)
    
    # ====================================
    # ЭТАП 4: ЛОГИРОВАНИЕ РЕЗУЛЬТАТА
    # ====================================
    total_time = time.time() - search_start_time
    
    if ai_response and getattr(ai_response, 'success', False):
        logger.info(f"💬 Chat response completed with AI: query='{message.message[:30]}...', "
                   f"sources={len(sources)}, total_time={total_time:.2f}s")
    else:
        logger.info(f"💬 Chat response completed with fallback: query='{message.message[:30]}...', "
                   f"sources={len(sources)}, total_time={total_time:.2f}s")
    
    return ChatResponse(
        response=response_text,
        sources=sources if sources else None
    )

def _generate_timeout_fallback_response(query: str, search_results: List[Dict], language: str) -> str:
    """Генерирует ответ при таймауте AI"""
    
    if search_results:
        # Формируем контекст из найденных документов
        context_snippets = []
        for result in search_results[:2]:  # Только первые 2 результата
            content = result.get('content', '')
            snippet = content[:200] + "..." if len(content) > 200 else content
            filename = result.get('filename', 'Unknown')
            relevance = result.get('relevance_score', 0.0)
            
            context_snippets.append(f"📄 {filename} (релевантность: {relevance:.1%}): {snippet}")
        
        context = "\n\n".join(context_snippets)
        
        if language == "uk":
            response_text = f"""⏰ **AI асистент перевищив час очікування**

**Ваше питання:** {query}

📚 Знайдено {len(search_results)} релевантних документів у базі знань:

{context}

🔄 **Статус:** AI модель зайняла забагато часу для відповіді.

💡 **Рекомендації:**
• Спробуйте коротше та конкретніше питання
• Перефразуйте запит простішими словами
• Використовуйте ключові слова замість довгих речень

🔧 **Для адміністратора:** Перевірте навантаження на GPTQ модель"""
        else:
            response_text = f"""⏰ **AI Assistant Timeout**

**Your Question:** {query}

📚 Found {len(search_results)} relevant documents in knowledge base:

{context}

🔄 **Status:** AI model took too long to respond.

💡 **Recommendations:**
• Try a shorter, more specific question
• Rephrase using simpler language
• Use keywords instead of long sentences

🔧 **For administrator:** Check GPTQ model load"""
        
    else:
        # Нет документов и таймаут AI
        if language == "uk":
            response_text = f"""⏰ **Таймаут AI та відсутність документів**

**Ваше питання:** {query}

❌ Не знайдено релевантних документів у базі знань.
⏰ AI модель також перевищила час очікування.

💡 **Спробуйте:**
• Використати інші ключові слова
• Сформулювати питання простіше
• Додати релевантні документи через адмін панель"""
        else:
            response_text = f"""⏰ **AI Timeout and No Documents**

**Your Question:** {query}

❌ No relevant documents found in knowledge base.
⏰ AI model also exceeded timeout.

💡 **Try:**
• Using different keywords
• Simplifying your question
• Adding relevant documents via admin panel"""
    
    return response_text

def _generate_no_documents_found_response(query: str, language: str) -> str:
    """Быстрый ответ когда нет документов и AI недоступен"""
    
    if language == "uk":
        return f"""🔍 **Результати пошуку для:** "{query}"

❌ На жаль, не знайдено релевантних документів у базі знань.

🤖 AI-помічник тимчасово недоступний.

💡 **Рекомендації:**
• Спробуйте інші ключові слова
• Використовуйте більш загальні терміни
• Додайте документи через адмін панель
• Спробуйте пізніше, коли AI буде доступний"""
    else:
        return f"""🔍 **Search Results for:** "{query}"

❌ Unfortunately, no relevant documents found in knowledge base.

🤖 AI assistant temporarily unavailable.

💡 **Recommendations:**
• Try different keywords
• Use more general terms  
• Add documents via admin panel
• Try again later when AI is available"""

def _generate_fallback_response_with_context(query: str, search_results: List[Dict], 
                                           language: str, error: str = None) -> str:
    """Генерирует fallback ответ когда AI недоступен, но есть найденный контекст"""
    
    # Формируем контекст из найденных документов с информацией о типе совпадения
    context_snippets = []
    for result in search_results[:3]:  # Максимум 3 документа для быстрого ответа
        content = result.get('content', '')
        snippet = content[:300] + "..." if len(content) > 300 else content
        filename = result.get('filename', 'Unknown')
        
        # Показываем тип совпадения
        match_type = result.get('search_info', {}).get('match_type', 'unknown')
        relevance = result.get('relevance_score', 0.0)
        
        # Формируем описание совпадения
        if match_type == "exact":
            if language == "uk":
                match_description = "📍 Точне співпадіння"
            else:
                match_description = "📍 Exact match"
        elif match_type == "semantic":
            if language == "uk":
                match_description = "🔍 Семантичне співпадіння"
            else:
                match_description = "🔍 Semantic match"
        else:
            if language == "uk":
                match_description = f"📊 Релевантність: {relevance:.1%}"
            else:
                match_description = f"📊 Relevance: {relevance:.1%}"
        
        context_snippets.append(f"📄 {filename} ({match_description}): {snippet}")
    
    context = "\n\n".join(context_snippets)
    
    if language == "uk":
        response_text = f"""🔍 Результати пошуку для запитання: "{query}"

📚 Знайдено {len(search_results)} релевантних документів у базі знань:

{context}

⚠️ AI-помічник тимчасово недоступний"""
        
        if error:
            response_text += f" (причина: {error})"
        
        response_text += """.

💡 На основі знайдених документів ви можете:
• Переглянути повний текст документів через адмін панель
• Уточнити запит для кращих результатів  
• Спробувати пошук іншими ключовими словами

🔧 Для адміністратора: перевірте статус GPTQ модели"""
    else:
        response_text = f"""🔍 Search results for query: "{query}"

📚 Found {len(search_results)} relevant documents in the knowledge base:

{context}

⚠️ AI assistant temporarily unavailable"""
        
        if error:
            response_text += f" (reason: {error})"
            
        response_text += """.

💡 Based on the found documents, you can:
• Review the full document text through the admin panel
• Refine your query for better results
• Try searching with different keywords

🔧 For administrator: check GPTQ model status"""
    
    return response_text

async def _generate_no_context_response(query: str, language: str, llm_service) -> str:
    """Генерирует ответ когда не найдено релевантных документов"""
    
    # Пытаемся получить AI ответ даже без контекста (общие знания) с таймаутом
    try:
        ai_response = await asyncio.wait_for(
            llm_service.answer_legal_question(
                question=query,
                context_documents=[],
                language=language
            ),
            timeout=60.0  # 1 минута для общих знаний
        )
        
        if ai_response and ai_response.success and ai_response.content.strip():
            # AI смог ответить без контекста
            if language == "uk":
                return f"""🤖 Відповідь на основі загальних знань:

{ai_response.content}

⚠️ Зверніть увагу: ця відповідь базується на загальних знаннях AI, а не на документах у вашій базі знань. Для більш точної інформації рекомендуємо додати релевантні документи через адмін панель."""
            else:
                return f"""🤖 Response based on general knowledge:

{ai_response.content}

⚠️ Note: This response is based on the AI's general knowledge, not on documents in your knowledge base. For more accurate information, we recommend adding relevant documents through the admin panel."""
        
    except asyncio.TimeoutError:
        logger.debug(f"AI general knowledge response timeout for: {query}")
    except Exception as e:
        logger.debug(f"AI general knowledge response failed: {e}")
    
    # Fallback если AI недоступен совсем
    return _generate_no_documents_found_response(query, language)

# Остальные endpoints остаются без изменений...

@router.get("/chat/history", response_model=ChatHistoryResponse)
async def get_chat_history(request: ChatHistoryRequest = ChatHistoryRequest()):
    """Получить историю чата"""
    try:
        # Получаем последние N сообщений
        recent_history = chat_history[-request.limit:] if chat_history else []
        
        # Преобразуем в нужный формат
        formatted_history = []
        for entry in recent_history:
            formatted_entry = ChatHistoryItem(
                message=entry["message"],
                response=entry["response"],
                language=entry["language"],
                sources=entry.get("sources"),
                timestamp=entry.get("timestamp")
            )
            formatted_history.append(formatted_entry)
        
        return ChatHistoryResponse(
            history=formatted_history,
            total_messages=len(chat_history)
        )
        
    except Exception as e:
        logger.error(f"Chat history error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get chat history: {str(e)}")

@router.delete("/chat/history")
async def clear_chat_history():
    """Очистить историю чатов"""
    try:
        global chat_history
        old_count = len(chat_history)
        chat_history.clear()
        
        logger.info(f"Chat history cleared: {old_count} messages removed")
        
        return {
            "message": f"Cleared {old_count} chat messages",
            "remaining": len(chat_history)
        }
        
    except Exception as e:
        logger.error(f"Clear history error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear chat history: {str(e)}")

@router.get("/chat/stats")
async def get_chat_stats():
    """Получить статистику чатов с AI метриками"""
    try:
        # Анализируем языки
        languages = {}
        sources_used = 0
        successful_searches = 0
        total_search_results = 0
        ai_responses = 0
        total_tokens = 0
        total_ai_time = 0.0
        timeouts = 0
        
        for entry in chat_history:
            lang = entry.get("language", "unknown")
            languages[lang] = languages.get(lang, 0) + 1
            
            if entry.get("sources"):
                sources_used += 1
            
            if entry.get("timeout", False):
                timeouts += 1
            
            # Анализируем статистику поиска
            search_stats = entry.get("search_stats", {})
            if search_stats.get("has_relevant_results", False):
                successful_searches += 1
                total_search_results += search_stats.get("found_documents", 0)
            
            # Анализируем AI статистику
            ai_stats = entry.get("ai_stats", {})
            if ai_stats.get("ai_used", False):
                ai_responses += 1
                total_tokens += ai_stats.get("tokens_used", 0)
                total_ai_time += ai_stats.get("response_time", 0)
        
        return {
            "total_messages": len(chat_history),
            "languages": languages,
            "messages_with_sources": sources_used,
            "successful_searches": successful_searches,
            "success_rate": (successful_searches / len(chat_history) * 100) if chat_history else 0,
            "average_sources_per_message": sources_used / len(chat_history) if chat_history else 0,
            "average_results_per_successful_search": total_search_results / successful_searches if successful_searches > 0 else 0,
            
            # AI статистика
            "ai_responses": ai_responses,
            "ai_usage_rate": (ai_responses / len(chat_history) * 100) if chat_history else 0,
            "total_tokens_used": total_tokens,
            "average_tokens_per_ai_response": total_tokens / ai_responses if ai_responses > 0 else 0,
            "total_ai_time": total_ai_time,
            "average_ai_response_time": total_ai_time / ai_responses if ai_responses > 0 else 0,
            
            # Таймауты
            "timeouts": timeouts,
            "timeout_rate": (timeouts / len(chat_history) * 100) if chat_history else 0
        }
        
    except Exception as e:
        logger.error(f"Chat stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get chat stats: {str(e)}")

@router.get("/chat/timeout-test")
async def test_timeout_behavior():
    """Тестовый endpoint для проверки таймаутов"""
    try:
        # Симулируем длительную операцию
        await asyncio.sleep(5)
        
        return {
            "message": "Timeout test completed",
            "duration": "5 seconds",
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Timeout test error: {e}")
        return {
            "message": "Timeout test failed",
            "error": str(e)
        }