# backend/api/user/chat.py - УПРОЩЕННЫЙ ЧАТ API
"""
Упрощенный чат endpoint для минимальной RAG системы
Оптимизирован под FLAN-T5 Small и быстрые ответы
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

# Временное хранение для истории чатов (упрощенно)
chat_history: List[Dict[str, Any]] = []

# Упрощенные таймауты для быстрой модели
SEARCH_TIMEOUT = 5.0    # 5 секунд на поиск
LLM_TIMEOUT = 20.0      # 20 секунд на FLAN-T5
TOTAL_TIMEOUT = 30.0    # 30 секунд общий таймаут

@router.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(
    message: ChatMessage,
    document_service = Depends(get_document_service),
    llm_service = Depends(get_llm_service)
):
    """Основной endpoint для чата с быстрыми таймаутами"""
    
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
            "response": "⏰ Request timeout - please try a shorter question",
            "language": message.language,
            "sources": [],
            "timestamp": time.time(),
            "timeout": True
        }
        chat_history.append(timeout_entry)
        
        timeout_response = "⏰ Запит перевищив час очікування. Спробуйте коротше питання." if message.language == "uk" else "⏰ Request timeout. Please try a shorter question."
        
        return ChatResponse(
            response=timeout_response,
            sources=[]
        )
    except Exception as e:
        logger.error(f"❌ Critical chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat service error: {str(e)}")

async def _process_chat_message(message: ChatMessage, document_service, llm_service):
    """Обрабатывает сообщение чата с оптимизацией под FLAN-T5"""
    
    search_results = []
    sources = []
    search_start_time = time.time()
    
    # ====================================
    # ЭТАП 1: БЫСТРЫЙ ПОИСК ДОКУМЕНТОВ
    # ====================================
    try:
        logger.info(f"🔍 Searching for: '{message.message[:50]}...'")
        
        # Поиск с коротким таймаутом
        search_results = await asyncio.wait_for(
            document_service.search(
                query=message.message,
                limit=settings.MAX_CONTEXT_DOCUMENTS,  # Обычно 2 для FLAN-T5
                min_relevance=0.4  # Чуть выше порог для качества
            ),
            timeout=SEARCH_TIMEOUT
        )
        
        search_time = time.time() - search_start_time
        
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
    # ЭТАП 2: FLAN-T5 ГЕНЕРАЦИЯ
    # ====================================
    ai_response = None
    response_text = ""
    llm_start_time = time.time()
    
    try:
        logger.info("🤖 Generating FLAN-T5 response...")
        
        # Подготавливаем контекст для FLAN-T5 (короткий)
        context_documents = []
        if search_results:
            for result in search_results[:2]:  # Максимум 2 документа
                content = result.get('content', '')
                # Короткий контекст для T5 Small
                if len(content) > settings.CONTEXT_TRUNCATE_LENGTH:
                    content = content[:settings.CONTEXT_TRUNCATE_LENGTH] + "..."
                
                context_doc = {
                    "filename": result.get('filename', 'Unknown'),
                    "content": content,
                    "relevance_score": result.get('relevance_score', 0.0),
                    "metadata": result.get('metadata', {})
                }
                context_documents.append(context_doc)
        
        # Генерируем ответ через FLAN-T5 с таймаутом
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
            logger.info(f"✅ FLAN-T5 response generated: {len(response_text)} chars in {llm_time:.2f}s")
        else:
            # AI не смог ответить, используем fallback
            logger.warning(f"AI response failed: {getattr(ai_response, 'error', 'Unknown error')}")
            response_text = _generate_quick_fallback_response(
                message.message, search_results, message.language
            )
                
    except asyncio.TimeoutError:
        llm_time = time.time() - llm_start_time
        logger.error(f"❌ FLAN-T5 timeout after {LLM_TIMEOUT}s")
        response_text = _generate_timeout_fallback_response(
            message.message, search_results, message.language
        )
        
    except Exception as e:
        llm_time = time.time() - llm_start_time
        logger.error(f"❌ FLAN-T5 generation error after {llm_time:.2f}s: {e}")
        response_text = _generate_quick_fallback_response(
            message.message, search_results, message.language
        )
    
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
            "search_time": search_time if 'search_time' in locals() else 0.0
        },
        "ai_stats": {
            "ai_used": ai_response is not None and getattr(ai_response, 'success', False),
            "model": "google/flan-t5-small",
            "tokens_used": getattr(ai_response, 'tokens_used', 0),
            "response_time": getattr(ai_response, 'response_time', 0.0),
            "llm_time": llm_time if 'llm_time' in locals() else 0.0
        }
    }
    chat_history.append(chat_entry)
    
    # Ограничиваем историю для экономии памяти
    if len(chat_history) > 50:  # Меньше лимит
        chat_history.pop(0)
    
    # ====================================
    # ЭТАП 4: ВОЗВРАТ РЕЗУЛЬТАТА
    # ====================================
    total_time = time.time() - search_start_time
    
    if ai_response and getattr(ai_response, 'success', False):
        logger.info(f"💬 Chat completed with FLAN-T5: '{message.message[:30]}...', "
                   f"sources={len(sources)}, total_time={total_time:.2f}s")
    else:
        logger.info(f"💬 Chat completed with fallback: '{message.message[:30]}...', "
                   f"sources={len(sources)}, total_time={total_time:.2f}s")
    
    return ChatResponse(
        response=response_text,
        sources=sources if sources else None
    )

def _generate_quick_fallback_response(query: str, search_results: List[Dict], language: str) -> str:
    """Быстрый fallback ответ с найденным контекстом"""
    
    if search_results:
        # Краткий ответ с контекстом
        context_preview = search_results[0].get('content', '')[:150] + "..." if search_results[0].get('content') else ""
        filename = search_results[0].get('filename', 'Unknown')
        
        if language == "uk":
            response_text = f"""🔍 **Знайдено релевантну інформацію**

**Питання:** {query}

📄 **Джерело:** {filename}
📝 **Витяг:** {context_preview}

⚠️ FLAN-T5 тимчасово недоступний. Інформація знайдена в документах."""
        else:
            response_text = f"""🔍 **Found Relevant Information**

**Question:** {query}

📄 **Source:** {filename}
📝 **Extract:** {context_preview}

⚠️ FLAN-T5 temporarily unavailable. Information found in documents."""
    else:
        # Нет документов
        if language == "uk":
            response_text = f"""🔍 **Результат пошуку**

**Питання:** {query}

❌ Релевантних документів не знайдено.
⚠️ FLAN-T5 сервіс недоступний.

💡 Спробуйте інші ключові слова або додайте документи."""
        else:
            response_text = f"""🔍 **Search Result**

**Question:** {query}

❌ No relevant documents found.
⚠️ FLAN-T5 service unavailable.

💡 Try different keywords or add documents."""
    
    return response_text

def _generate_timeout_fallback_response(query: str, search_results: List[Dict], language: str) -> str:
    """Ответ при таймауте FLAN-T5"""
    
    if language == "uk":
        response_text = f"""⏰ **Таймаут FLAN-T5**

**Питання:** {query}

⚠️ FLAN-T5 Small модель перевищила час очікування.

📚 Знайдено документів: {len(search_results)}

💡 **Рекомендації:**
• Спробуйте коротше питання
• Перефразуйте простіше
• Модель може бути перевантажена"""
    else:
        response_text = f"""⏰ **FLAN-T5 Timeout**

**Question:** {query}

⚠️ FLAN-T5 Small model exceeded timeout.

📚 Documents found: {len(search_results)}

💡 **Recommendations:**
• Try a shorter question
• Rephrase more simply  
• Model may be overloaded"""
    
    return response_text

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
    """Получить упрощенную статистику чатов"""
    try:
        if not chat_history:
            return {
                "total_messages": 0,
                "ai_responses": 0,
                "average_response_time": 0,
                "languages": {},
                "success_rate": 0
            }
        
        # Анализируем языки
        languages = {}
        sources_used = 0
        successful_searches = 0
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
            
            # Статистика поиска
            search_stats = entry.get("search_stats", {})
            if search_stats.get("found_documents", 0) > 0:
                successful_searches += 1
            
            # AI статистика
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
            
            # FLAN-T5 статистика
            "ai_responses": ai_responses,
            "ai_usage_rate": (ai_responses / len(chat_history) * 100) if chat_history else 0,
            "total_tokens_used": total_tokens,
            "average_tokens_per_response": total_tokens / ai_responses if ai_responses > 0 else 0,
            "total_ai_time": total_ai_time,
            "average_ai_response_time": total_ai_time / ai_responses if ai_responses > 0 else 0,
            
            # Таймауты
            "timeouts": timeouts,
            "timeout_rate": (timeouts / len(chat_history) * 100) if chat_history else 0,
            
            # Модель информация
            "model_info": {
                "llm_model": "google/flan-t5-small",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "max_context_docs": 2,
                "memory_optimized": True
            }
        }
        
    except Exception as e:
        logger.error(f"Chat stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get chat stats: {str(e)}")

@router.get("/chat/model-info")
async def get_model_info():
    """Информация о моделях"""
    try:
        from app.dependencies import get_llm_service
        llm_service = get_llm_service()
        llm_status = await llm_service.get_service_status()
        
        return {
            "llm": {
                "model": "google/flan-t5-small",
                "type": "text2text-generation",
                "parameters": "80M",
                "memory_usage": "~300 MB",
                "ready": llm_status.get("ready", False),
                "service_type": llm_status.get("service_type", "unknown")
            },
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimensions": 384,
                "memory_usage": "~90 MB",
                "type": "sentence-transformer"
            },
            "vector_db": {
                "type": "ChromaDB",
                "memory_usage": "~20 MB per 10K docs",
                "features": ["semantic_search", "similarity_threshold"]
            },
            "system": {
                "total_memory_target": "<1GB RAM",
                "optimizations": [
                    "Small model sizes",
                    "Efficient embeddings", 
                    "Short context windows",
                    "Fast inference"
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return {
            "error": str(e),
            "fallback_info": {
                "llm_model": "google/flan-t5-small",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "status": "Model information unavailable"
            }
        }

@router.post("/chat/test")
async def test_chat():
    """Тестовый endpoint для проверки чата"""
    test_message = ChatMessage(
        message="What is law?",
        language="en"
    )
    
    try:
        from app.dependencies import get_document_service, get_llm_service
        doc_service = get_document_service()
        llm_service = get_llm_service()
        
        start_time = time.time()
        result = await _process_chat_message(test_message, doc_service, llm_service)
        end_time = time.time()
        
        return {
            "test_successful": True,
            "response": result.response[:200] + "..." if len(result.response) > 200 else result.response,
            "sources_found": len(result.sources) if result.sources else 0,
            "response_time": end_time - start_time,
            "model": "google/flan-t5-small"
        }
        
    except Exception as e:
        return {
            "test_successful": False,
            "error": str(e),
            "model": "google/flan-t5-small",
            "recommendations": [
                "Check FLAN-T5 model availability",
                "Verify transformers installation",
                "Check HuggingFace Hub access"
            ]
        }