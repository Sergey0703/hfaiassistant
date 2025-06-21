# ====================================
# ФАЙЛ: backend/services/scraper_service.py (БЕЗ ДЕМО РЕЖИМА)
# Заменить существующий файл полностью
# ====================================

"""
Legal Site Scraper Service - Реальный сервис для парсинга юридических сайтов
"""

import aiohttp
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

@dataclass
class ScrapedDocument:
    """Структура спарсенного документа"""
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]
    category: str = "scraped"

class LegalSiteScraper:
    """Сервис для парсинга юридических сайтов"""
    
    def __init__(self):
        self.session = None
        self.legal_sites_config = {
            # Украинские сайты
            "zakon.rada.gov.ua": {
                "title": "h1, .page-title, .document-title, .field-name-title",
                "content": ".field-item, .document-content, .law-text, .content, main, .field-type-text-long",
                "exclude": "nav, footer, .sidebar, .menu, script, style, .breadcrumb, .tabs"
            },
            "court.gov.ua": {
                "title": "h1, .title, .page-title",
                "content": ".content, .text, main, .field-item",
                "exclude": "nav, footer, .menu, .sidebar"
            },
            
            # Ирландские сайты
            "irishstatutebook.ie": {
                "title": "h1, .act-title, .page-title",
                "content": ".act-text, .section, .content, .akn-akomaNtoso",
                "exclude": "nav, footer, .navigation, .sidebar"
            },
            "citizensinformation.ie": {
                "title": "h1, .page-title",
                "content": ".content, .article-content, .main-content",
                "exclude": "nav, footer, .sidebar, .navigation"
            },
            "courts.ie": {
                "title": "h1, .page-title",
                "content": ".content, .main-content",
                "exclude": "nav, footer, .menu, .sidebar"
            }
        }
        
        logger.info("🌐 Legal Site Scraper initialized (Real mode only)")
    
    async def _get_session(self):
        """Получает или создает HTTP сессию"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9,uk;q=0.8',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }
            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self.session
    
    async def scrape_legal_site(self, url: str) -> Optional[ScrapedDocument]:
        """Парсит юридический сайт - только реальный парсинг"""
        
        try:
            session = await self._get_session()
            
            logger.info(f"🔍 Scraping URL: {url}")
            
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"❌ HTTP {response.status} for {url}")
                    raise Exception(f"HTTP {response.status}")
                
                html_content = await response.text()
                
                # Парсим HTML
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Определяем конфигурацию для сайта
                domain = urlparse(url).netloc
                config = self._get_site_config(domain)
                
                # Извлекаем заголовок
                title = self._extract_title(soup, config)
                
                # Извлекаем контент
                content = self._extract_content(soup, config)
                
                if not content or len(content.strip()) < 100:
                    logger.error(f"❌ Insufficient content extracted from {url}")
                    raise Exception("Insufficient content extracted")
                
                # Метаданные
                metadata = {
                    "url": url,
                    "scraped_at": time.time(),
                    "status_code": response.status,
                    "content_length": len(content),
                    "word_count": len(content.split()),
                    "real_scraping": True,
                    "domain": domain,
                    "scraper": "LegalSiteScraper"
                }
                
                logger.info(f"✅ Successfully scraped {url}: {len(content)} chars")
                
                return ScrapedDocument(
                    url=url,
                    title=title,
                    content=content,
                    metadata=metadata,
                    category=self._categorize_by_domain(domain)
                )
                
        except aiohttp.ClientError as e:
            logger.error(f"❌ Network error scraping {url}: {e}")
            raise Exception(f"Network error: {e}")
        except Exception as e:
            logger.error(f"❌ Error scraping {url}: {e}")
            raise Exception(f"Scraping error: {e}")
    
    def _get_site_config(self, domain: str) -> Dict[str, str]:
        """Получает конфигурацию для конкретного домена"""
        # Проверяем точное совпадение
        if domain in self.legal_sites_config:
            return self.legal_sites_config[domain]
        
        # Проверяем поддомены
        for site_domain, config in self.legal_sites_config.items():
            if domain.endswith(site_domain) or site_domain in domain:
                return config
        
        # Дефолтная конфигурация
        return {
            "title": "h1, title, .title, .page-title",
            "content": "main, .content, .main-content, article, .article, body",
            "exclude": "nav, footer, .sidebar, .menu, script, style, .ads, .navigation"
        }
    
    def _extract_title(self, soup, config: Dict[str, str]) -> str:
        """Извлекает заголовок страницы"""
        title_selectors = config.get("title", "h1, title").split(", ")
        
        for selector in title_selectors:
            try:
                element = soup.select_one(selector.strip())
                if element and element.get_text(strip=True):
                    title = element.get_text(strip=True)
                    if len(title) > 5:  # Минимальная длина заголовка
                        return title[:200]  # Ограничиваем длину
            except Exception:
                continue
        
        # Fallback к title тегу
        title_tag = soup.find('title')
        if title_tag and title_tag.get_text(strip=True):
            return title_tag.get_text(strip=True)[:200]
        
        return "Untitled Document"
    
    def _extract_content(self, soup, config: Dict[str, str]) -> str:
        """Извлекает основной контент страницы"""
        try:
            # Удаляем нежелательные элементы
            exclude_selectors = config.get("exclude", "").split(", ")
            for selector in exclude_selectors:
                if selector.strip():
                    for element in soup.select(selector.strip()):
                        element.decompose()
            
            # Извлекаем контент
            content_selectors = config.get("content", "main, .content").split(", ")
            content_parts = []
            
            for selector in content_selectors:
                try:
                    elements = soup.select(selector.strip())
                    for element in elements:
                        text = element.get_text(separator='\n', strip=True)
                        if text and len(text) > 50:  # Минимальная длина
                            content_parts.append(text)
                except Exception:
                    continue
            
            if content_parts:
                content = '\n\n'.join(content_parts)
                # Очищаем лишние пробелы и переносы
                lines = []
                for line in content.split('\n'):
                    cleaned_line = line.strip()
                    if cleaned_line and len(cleaned_line) > 5:  # Убираем очень короткие строки
                        lines.append(cleaned_line)
                
                return '\n'.join(lines)
            
            # Fallback - весь текст страницы
            body_text = soup.get_text(separator='\n', strip=True)
            if body_text and len(body_text) > 100:
                return body_text
            
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            return ""
    
    def _categorize_by_domain(self, domain: str) -> str:
        """Определяет категорию документа по домену"""
        domain_lower = domain.lower()
        
        # Украинские сайты
        if any(ua_domain in domain_lower for ua_domain in [
            "zakon.rada.gov.ua", "rada.gov.ua", "court.gov.ua", 
            "minjust.gov.ua", "ccu.gov.ua", "npu.gov.ua"
        ]):
            return "ukraine_legal"
        
        # Ирландские сайты
        if any(ie_domain in domain_lower for ie_domain in [
            "irishstatutebook.ie", "courts.ie", "citizensinformation.ie",
            "justice.ie", "oireachtas.ie", "gov.ie"
        ]):
            return "ireland_legal"
        
        # Общие категории
        if any(keyword in domain_lower for keyword in ["law", "legal", "court", "justice"]):
            return "legislation"
        elif any(keyword in domain_lower for keyword in ["zakon", "pravo", "sud"]):
            return "legislation"
        elif "court" in domain_lower or "sud" in domain_lower:
            return "jurisprudence"
        elif any(keyword in domain_lower for keyword in ["gov", "government", "ministry"]):
            return "government"
        elif any(keyword in domain_lower for keyword in ["citizen", "immigration", "rights"]):
            return "civil_rights"
        else:
            return "scraped"
    
    async def scrape_multiple_urls(self, urls: List[str], delay: float = 1.0) -> List[Optional[ScrapedDocument]]:
        """Парсит несколько URL с задержкой"""
        results = []
        
        for i, url in enumerate(urls):
            if i > 0 and delay > 0:
                await asyncio.sleep(delay)
            
            try:
                document = await self.scrape_legal_site(url)
                results.append(document)
                logger.info(f"✅ Successfully processed {i+1}/{len(urls)}: {url}")
                
            except Exception as e:
                logger.error(f"❌ Failed {i+1}/{len(urls)}: {url} - {e}")
                results.append(None)
        
        successful = len([r for r in results if r is not None])
        logger.info(f"🎯 Bulk scrape completed: {successful}/{len(urls)} successful")
        
        return results
    
    async def validate_url(self, url: str) -> Dict[str, Any]:
        """Валидирует доступность URL"""
        try:
            session = await self._get_session()
            
            async with session.head(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                return {
                    "url": url,
                    "valid": True,
                    "status_code": response.status,
                    "reachable": response.status < 400,
                    "content_type": response.headers.get('content-type', 'unknown'),
                    "content_length": response.headers.get('content-length', 'unknown')
                }
                
        except Exception as e:
            return {
                "url": url,
                "valid": False,
                "reachable": False,
                "error": str(e)
            }
    
    def get_supported_sites(self) -> Dict[str, Any]:
        """Возвращает список поддерживаемых сайтов"""
        return {
            "sites": list(self.legal_sites_config.keys()),
            "total": len(self.legal_sites_config),
            "real_scraping_available": True  # Всегда True, так как нет демо режима
        }
    
    async def close(self):
        """Закрывает HTTP сессию"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.debug("🔒 Scraper session closed")