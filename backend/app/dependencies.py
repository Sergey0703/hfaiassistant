# app/dependencies.py - ИСПРАВЛЕННАЯ ВЕРСИЯ ДЛЯ HUGGINGFACE SPACES

"""
Зависимости и инициализация сервисов для HuggingFace Spaces
"""

import logging
import sys
import os
import time
import json
import asyncio
from typing import Optional, List, Dict, Any

from app.config import settings

logger = logging.getLogger(__name__)

# ====================================
# FALLBACK СЕРВИСЫ ДЛЯ HF SPACES
# ====================================

class FallbackDocumentService:
    """Простой document service для HF Spaces без sentence-transformers"""
    
    def __init__(self):
        self.service_type = "hf_spaces_fallback"
        self.vector_db = type('MockVectorDB', (), {
            'persist_directory': './fallback_db'
        })()
        logger.info("📝 Using HF Spaces fallback document service")
    
    async def search(self, query: str, category: str = None, limit: int = 5, min_relevance: float = 0.3):
        """Простой текстовый поиск без векторизации"""
        logger.info(f"🔍 Fallback search for: {query}")
        
        # Возвращаем демо результаты с релевантным контентом
        demo_results = [
            {
                "content": f"""Legal Information Related to: "{query}"

This is a demonstration response for HuggingFace Spaces deployment. 
In a full deployment, this would search through your legal document database.

Key Points:
• Your query "{query}" would be processed using semantic search
• Multiple legal documents would be analyzed for relevance
• Results would be ranked by relevance score
• Specific legal references and citations would be provided

To enable full functionality:
1. Complete the sentence-transformers installation
2. Upload legal documents to the database
3. Configure vector search parameters

For demonstration purposes, this shows the API structure and response format.""",
                "filename": f"demo_legal_doc_{query.replace(' ', '_')[:20]}.txt",
                "document_id": f"demo_{int(time.time())}",
                "relevance_score": 0.85,
                "metadata": {
                    "status": "demo",
                    "category": category or "general",
                    "service": "hf_spaces_fallback",
                    "query": query,
                    "demo_mode": True
                }
            }
        ]
        
        return demo_results
    
    async def get_stats(self):
        """Демо статистика для HF Spaces"""
        return {
            "total_documents": 0,
            "categories": ["general", "demo", "legislation"],
            "database_type": "HF Spaces Fallback",
            "status": "Demo mode for HuggingFace Spaces deployment",
            "message": "Full functionality requires sentence-transformers setup",
            "available_features": [
                "✅ FastAPI REST API structure",
                "✅ Document upload endpoints", 
                "✅ Search API (demo responses)",
                "✅ Admin panel endpoints",
                "⚠️ Vector search (requires setup)",
                "⚠️ Real document processing (requires setup)"
            ],
            "setup_requirements": [
                "Fix sentence-transformers installation",
                "Configure vector database",
                "Upload legal documents"
            ]
        }
    
    async def get_all_documents(self):
        """Демо список документов"""
        return []
    
    async def delete_document(self, doc_id: str):
        """Демо удаление"""
        logger.info(f"Demo delete: {doc_id}")
        return False
    
    async def process_and_store_file(self, file_path: str, category: str = "general"):
        """Демо обработка файла"""
        logger.info(f"Demo file processing: {file_path}")
        return False

class FallbackLLMService:
    """LLM service для HF Spaces с улучшенными демо-ответами"""
    
    def __init__(self):
        self.service_type = "hf_spaces_demo"
        self.model_loaded = False
        logger.info("🤖 Using HF Spaces demo LLM service")
    
    async def answer_legal_question(self, question: str, context_documents: List[Dict], language: str = "en"):
        """Улучшенные демо ответы для HF Spaces"""
        from services.huggingface_llm_service import LLMResponse
        
        if language == "uk":
            demo_content = f"""🏛️ **Демонстраційна відповідь Legal Assistant**

**Ваше питання:** {question}

**Аналіз:** На основі {len(context_documents)} знайдених документів у базі знань.

**Демонстраційна відповідь:**
Це демонстраційна версія Legal Assistant для HuggingFace Spaces. У повній версії система би:

1. **Проаналізувала** ваш юридичний запит використовуючи ШІ
2. **Знайшла** релевантні документи в базі законодавства
3. **Надала** точну відповідь з посиланнями на закони
4. **Включила** специфічні статті та нормативні акти

**Поточний статус:** Демо режим на HuggingFace Spaces
**Для повного функціоналу:** Необхідно налаштувати sentence-transformers та завантажити юридичні документи.

📚 Ця версія демонструє архітектуру API та формат відповідей."""
        else:
            demo_content = f"""🏛️ **Legal Assistant Demo Response**

**Your Question:** {question}

**Analysis:** Based on {len(context_documents)} documents found in the knowledge base.

**Demo Response:**
This is a demonstration version of Legal Assistant running on HuggingFace Spaces. In the full version, the system would:

1. **Analyze** your legal query using AI
2. **Search** through relevant legal documents
3. **Provide** precise answers with legal references
4. **Include** specific articles and regulations

**Current Status:** Demo mode on HuggingFace Spaces
**For Full Functionality:** Requires sentence-transformers setup and legal document uploads.

📚 This version demonstrates the API architecture and response format."""
        
        return LLMResponse(
            content=demo_content,
            model="hf_spaces_demo",
            tokens_used=len(demo_content.split()),
            response_time=0.5,
            success=True,
            error=None
        )
    
    async def get_service_status(self):
        """Статус LLM для HF Spaces"""
        return {
            "model_loaded": False,
            "model_name": "hf_spaces_demo",
            "huggingface_available": True,
            "service_type": "demo",
            "environment": "HuggingFace Spaces",
            "status": "Demo mode - showing API structure",
            "supported_languages": ["en", "uk"],
            "demo_features": [
                "✅ REST API endpoints",
                "✅ Multi-language support",
                "✅ Legal document structure",
                "✅ Admin panel integration",
                "⚠️ AI responses (demo only)",
                "⚠️ Vector search (requires setup)"
            ],
            "recommendations": [
                "This is a working demo of the Legal Assistant API",
                "Full AI functionality requires model loading",
                "Upload legal documents for real search capabilities",
                "Configure HuggingFace Transformers for AI responses"
            ]
        }

class FallbackScraperService:
    """Web scraper для HF Spaces"""
    
    def __init__(self):
        self.service_type = "hf_spaces_scraper"
        self.legal_sites_config = {
            "irishstatutebook.ie": {"title": "h1", "content": ".content"},
            "citizensinformation.ie": {"title": "h1", "content": ".content"},
            "zakon.rada.gov.ua": {"title": "h1", "content": ".content"}
        }
        logger.info("🌐 HF Spaces scraper service initialized")
    
    async def scrape_legal_site(self, url: str):
        """Демо скрапинг для HF Spaces"""
        logger.info(f"🔍 Demo scraping: {url}")
        
        demo_content = f"""📄 **Legal Document from {url}**

This is a demonstration of web scraping functionality on HuggingFace Spaces.

**Document Source:** {url}
**Scraped Content:** In the full version, this would contain the actual legal text from the website.

**Sample Legal Content:**
"Article 1. This demonstration shows how the Legal Assistant would extract and process legal documents from official government websites, legal databases, and court systems.

The system would automatically:
- Extract relevant legal text
- Identify document structure
- Categorize by legal domain
- Process for semantic search
- Store in vector database"

**Metadata:**
- URL: {url}
- Language: Auto-detected
- Category: Determined by domain
- Processing: Demo mode

🔧 **For Full Functionality:** Enable web scraping libraries (aiohttp, beautifulsoup4)
"""
        
        return type('DemoDocument', (), {
            'url': url,
            'title': f'Demo Legal Document from {url}',
            'content': demo_content,
            'metadata': {
                'status': 'demo',
                'real_scraping': False,
                'scraped_at': time.time(),
                'service': 'hf_spaces_demo',
                'url': url
            },
            'category': 'demo'
        })()
    
    async def scrape_multiple_urls(self, urls: List[str], delay: float = 1.0):
        """Демо массовый скрапинг"""
        results = []
        for url in urls:
            doc = await self.scrape_legal_site(url)
            results.append(doc)
        return results

# ====================================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# ====================================

document_service: Optional[object] = None
scraper: Optional[object] = None
llm_service: Optional[object] = None
SERVICES_AVAILABLE: bool = False
CHROMADB_ENABLED: bool = False
LLM_ENABLED: bool = False

async def init_services():
    """Инициализация сервисов для HuggingFace Spaces"""
    global document_service, scraper, llm_service, SERVICES_AVAILABLE, CHROMADB_ENABLED, LLM_ENABLED
    
    logger.info("🚀 Initializing services for HuggingFace Spaces...")
    
    # Детектируем среду HF Spaces
    is_hf_spaces = os.getenv("SPACE_ID") is not None
    if is_hf_spaces:
        logger.info("🤗 Running on HuggingFace Spaces - using optimized setup")
    
    # ====================================
    # ПОПЫТКА РЕАЛЬНОГО DOCUMENT SERVICE
    # ====================================
    try:
        # Сначала пробуем реальный сервис
        if settings.USE_CHROMADB:
            try:
                # Проверяем доступность sentence-transformers
                import sentence_transformers
                from services.chroma_service import DocumentService
                document_service = DocumentService(settings.CHROMADB_PATH)
                CHROMADB_ENABLED = True
                SERVICES_AVAILABLE = True
                logger.info("✅ Real ChromaDB service initialized")
            except ImportError:
                logger.warning("⚠️ sentence-transformers not available, using fallback")
                raise ImportError("sentence-transformers not available")
        else:
            from services.document_processor import DocumentService
            document_service = DocumentService(settings.SIMPLE_DB_PATH)
            SERVICES_AVAILABLE = True
            logger.info("✅ SimpleVectorDB service initialized")
    
    except Exception as e:
        logger.info(f"ℹ️ Using fallback document service: {e}")
        document_service = FallbackDocumentService()
        SERVICES_AVAILABLE = True  # Fallback считается доступным
        CHROMADB_ENABLED = False
    
    # ====================================
    # SCRAPER SERVICE
    # ====================================
    try:
        # Проверяем доступность библиотек для скрапинга
        import aiohttp
        import bs4
        from services.scraper_service import LegalSiteScraper
        scraper = LegalSiteScraper()
        logger.info("✅ Real scraper service initialized")
    except ImportError:
        logger.info("ℹ️ Using fallback scraper service (aiohttp/bs4 not available)")
        scraper = FallbackScraperService()
    
    # ====================================
    # LLM SERVICE
    # ====================================
    try:
        if not settings.LLM_DEMO_MODE:
            # Пробуем реальный LLM сервис
            from services.huggingface_llm_service import create_llm_service
            llm_service = create_llm_service()
            
            status = await llm_service.get_service_status()
            if status.get("model_loaded", False):
                LLM_ENABLED = True
                logger.info("✅ Real HuggingFace LLM service initialized")
            else:
                raise Exception("Model not loaded")
        else:
            raise Exception("Demo mode enabled")
    
    except Exception as e:
        logger.info(f"ℹ️ Using demo LLM service: {e}")
        llm_service = FallbackLLMService()
        LLM_ENABLED = False
    
    # ====================================
    # ФИНАЛЬНЫЙ СТАТУС
    # ====================================
    logger.info(f"📊 HuggingFace Spaces services initialized:")
    logger.info(f"   Document service: {'✅ Real' if isinstance(document_service, FallbackDocumentService) == False else '🔄 Demo'}")
    logger.info(f"   Scraper service: {'✅ Real' if isinstance(scraper, FallbackScraperService) == False else '🔄 Demo'}")
    logger.info(f"   LLM service: {'✅ Real' if LLM_ENABLED else '🔄 Demo'}")
    logger.info(f"   Environment: {'🤗 HuggingFace Spaces' if is_hf_spaces else '💻 Local'}")

# ====================================
# DEPENDENCY FUNCTIONS
# ====================================

def get_document_service():
    """Dependency для получения document service"""
    return document_service or FallbackDocumentService()

def get_scraper_service():
    """Dependency для получения scraper service"""
    return scraper or FallbackScraperService()

def get_llm_service():
    """Dependency для получения LLM service"""
    return llm_service or FallbackLLMService()

def get_services_status():
    """Статус всех сервисов"""
    return {
        "document_service_available": document_service is not None,
        "scraper_available": scraper is not None,
        "llm_available": LLM_ENABLED,
        "llm_service_created": llm_service is not None,
        "chromadb_enabled": CHROMADB_ENABLED,
        "services_available": SERVICES_AVAILABLE,
        "huggingface_spaces": os.getenv("SPACE_ID") is not None,
        "environment": "hf_spaces" if os.getenv("SPACE_ID") else "local",
        "demo_mode": isinstance(document_service, FallbackDocumentService),
        "real_features": {
            "vector_search": CHROMADB_ENABLED,
            "web_scraping": not isinstance(scraper, FallbackScraperService),
            "ai_responses": LLM_ENABLED
        }
    }