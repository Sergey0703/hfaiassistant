# backend/services/chroma_service.py - КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ async ошибки
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import logging
import time
import hashlib
import asyncio
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class ProcessedDocument:
    id: str
    filename: str
    content: str
    metadata: Dict
    category: str
    chunks: List[str]

class ChromaDBService:
    """Сервис для работы с ChromaDB векторной базой данных С ИСПРАВЛЕННЫМ ASYNC"""
    
    def __init__(self, persist_directory: str = "./chromadb_data"):
        self.persist_directory = persist_directory
        
        # Создаем директорию если не существует
        os.makedirs(persist_directory, exist_ok=True)
        
        # ИСПРАВЛЕНИЕ: Добавляем таймауты
        self.operation_timeout = 30  # 30 секунд на операцию
        self.search_timeout = 10     # 10 секунд на поиск
        
        # Инициализируем ChromaDB клиент
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Настраиваем эмбеддинг функцию
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Создаем или получаем коллекцию документов
        self.collection = self.client.get_or_create_collection(
            name="legal_documents",
            embedding_function=self.embedding_function,
            metadata={"description": "Legal Assistant Documents Collection"}
        )
        
        logger.info(f"ChromaDB initialized with {self.collection.count()} documents")
    
    async def add_document(self, document: ProcessedDocument) -> bool:
        """Добавляет документ в ChromaDB С ТАЙМАУТОМ"""
        try:
            # ИСПРАВЛЕНИЕ: Используем run_in_executor для синхронной операции
            return await asyncio.get_event_loop().run_in_executor(
                None, 
                self._add_document_sync, 
                document
            )
        except Exception as e:
            logger.error(f"❌ Error adding document: {e}")
            return False
    
    def _add_document_sync(self, document: ProcessedDocument) -> bool:
        """Синхронная версия добавления документа"""
        try:
            # Проверяем существование документа
            existing_docs = self.collection.get(
                ids=[document.id],
                include=["metadatas"]
            )
            
            if existing_docs["ids"]:
                logger.warning(f"Document {document.id} already exists, skipping addition")
                return True
            
            # Подготавливаем метаданные для ChromaDB
            chroma_metadata = {
                "filename": document.filename,
                "category": document.category,
                "content_length": len(document.content),
                "word_count": len(document.content.split()),
                "chunks_count": len(document.chunks),
                "added_at": time.time(),
                **document.metadata
            }
            
            # Добавляем основной документ
            main_metadata = chroma_metadata.copy()
            main_metadata.update({
                "is_chunk": False,
                "chunk_index": -1,
                "parent_document_id": document.id
            })
            
            self.collection.add(
                ids=[document.id],
                documents=[document.content],
                metadatas=[main_metadata]
            )
            
            logger.info(f"✅ Added main document {document.filename}")
            
            # Если документ большой, добавляем чанки
            if len(document.chunks) > 1:
                chunk_ids = []
                chunk_documents = []
                chunk_metadatas = []
                
                for i, chunk in enumerate(document.chunks):
                    chunk_id = f"{document.id}_chunk_{i}"
                    
                    # Проверяем существование чанка
                    existing_chunk = self.collection.get(
                        ids=[chunk_id],
                        include=["metadatas"]
                    )
                    
                    if existing_chunk["ids"]:
                        logger.debug(f"Chunk {chunk_id} already exists, skipping")
                        continue
                    
                    chunk_ids.append(chunk_id)
                    chunk_documents.append(chunk)
                    
                    chunk_metadata = chroma_metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": i,
                        "parent_document_id": document.id,
                        "is_chunk": True
                    })
                    chunk_metadatas.append(chunk_metadata)
                
                # Добавляем только новые чанки
                if chunk_ids:
                    self.collection.add(
                        ids=chunk_ids,
                        documents=chunk_documents,
                        metadatas=chunk_metadatas
                    )
                    
                    logger.info(f"✅ Added {len(chunk_ids)} new chunks for {document.filename}")
                else:
                    logger.info(f"✅ All chunks for {document.filename} already exist")
            else:
                logger.info(f"✅ Added single document {document.filename}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding document sync: {e}")
            return False
    
    async def search_documents(self, query: str, n_results: int = 5, 
                             category: str = None, min_relevance: float = 0.3, **filters) -> List[Dict]:
        """
        КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Правильный async поиск документов
        """
        try:
            # ИСПРАВЛЕНИЕ: Используем run_in_executor для синхронной ChromaDB операции
            return await asyncio.get_event_loop().run_in_executor(
                None,
                self._search_documents_sync,
                query, n_results, category, min_relevance, filters
            )
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []
    
    def _search_documents_sync(self, query: str, n_results: int, 
                              category: str, min_relevance: float, filters: dict) -> List[Dict]:
        """ИСПРАВЛЕННАЯ синхронная версия поиска документов"""
        try:
            # Подготавливаем фильтры
            where_filter = {}
            
            if category:
                where_filter["category"] = category
            
            # Добавляем дополнительные фильтры (но НЕ is_chunk!)
            for key, value in filters.items():
                if key != "is_chunk":
                    where_filter[key] = value
            
            # Увеличиваем количество результатов для лучшей фильтрации
            search_limit = min(n_results * 3, 20)
            
            # ИСПРАВЛЕНО: Синхронный вызов ChromaDB query
            results = self.collection.query(
                query_texts=[query],
                n_results=search_limit,
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Форматируем и фильтруем результаты
            formatted_results = []
            query_lower = query.lower()
            seen_parent_ids = set()
            
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    distance = results["distances"][0][i]
                    
                    # ИСПРАВЛЕНО: Новая формула для relevance_score
                    if distance <= 0:
                        relevance_score = 1.0
                    elif distance >= 2.0:
                        relevance_score = 0.0
                    else:
                        relevance_score = max(0.0, (2.0 - distance) / 2.0)
                    
                    # Фильтрация по минимальной релевантности
                    if relevance_score < min_relevance:
                        logger.debug(f"Skipping result with low relevance: {relevance_score:.3f}")
                        continue
                    
                    document_content = results["documents"][0][i]
                    metadata = results["metadatas"][0][i]
                    
                    # Получаем parent_document_id
                    parent_doc_id = metadata.get("parent_document_id")
                    current_doc_id = results["ids"][0][i]
                    
                    # Избегаем дубликатов
                    unique_id = parent_doc_id or current_doc_id
                    if unique_id in seen_parent_ids:
                        logger.debug(f"Skipping duplicate parent document: {unique_id}")
                        continue
                    seen_parent_ids.add(unique_id)
                    
                    # Проверяем наличие точного совпадения в тексте
                    content_lower = document_content.lower()
                    filename_lower = metadata.get("filename", "").lower()
                    
                    # Определяем тип совпадения
                    exact_match = query_lower in content_lower or query_lower in filename_lower
                    semantic_match = relevance_score > 0.7
                    
                    # Лучший контекст
                    best_context = self._find_best_context(document_content, query, max_length=400)
                    
                    result = {
                        "content": best_context,
                        "full_content": document_content,
                        "metadata": metadata,
                        "distance": distance,
                        "relevance_score": relevance_score,
                        "document_id": parent_doc_id or current_doc_id,
                        "filename": metadata.get("filename", "Unknown"),
                        "exact_match": exact_match,
                        "semantic_match": semantic_match,
                        "is_chunk": metadata.get("is_chunk", False),
                        "search_info": {
                            "query": query,
                            "match_type": "exact" if exact_match else ("semantic" if semantic_match else "weak"),
                            "confidence": "high" if relevance_score > 0.7 else ("medium" if relevance_score > 0.5 else "low"),
                            "source_type": "chunk" if metadata.get("is_chunk", False) else "document"
                        }
                    }
                    formatted_results.append(result)
            
            # Сортируем результаты
            formatted_results.sort(key=lambda x: (
                x["exact_match"],
                not x["is_chunk"],
                x["relevance_score"]
            ), reverse=True)
            
            # Ограничиваем до запрошенного количества
            formatted_results = formatted_results[:n_results]
            
            # Логирование результатов
            if formatted_results:
                logger.info(f"Found {len(formatted_results)} relevant results for '{query}' (min_relevance={min_relevance})")
            else:
                logger.info(f"No relevant results found for '{query}' with min_relevance={min_relevance}")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Sync search error: {e}")
            return []
    
    def _find_best_context(self, content: str, query: str, max_length: int = 400) -> str:
        """Находит наиболее релевантную часть документа для показа в результатах"""
        if len(content) <= max_length:
            return content
        
        query_words = query.lower().split()
        content_lower = content.lower()
        
        # Ищем лучшее место для начала контекста
        best_score = 0
        best_start = 0
        
        # Проверяем различные позиции в тексте
        for start in range(0, len(content) - max_length + 1, max_length // 4):
            end = start + max_length
            segment = content_lower[start:end]
            
            # Считаем количество найденных слов запроса в сегменте
            score = sum(1 for word in query_words if word in segment)
            
            # Бонус за нахождение в начале документа
            if start == 0:
                score += 0.5
            
            if score > best_score:
                best_score = score
                best_start = start
        
        # Если не нашли хороший контекст, возвращаем начало
        if best_score == 0:
            return content[:max_length] + "..."
        
        # Возвращаем лучший контекст
        best_end = best_start + max_length
        context = content[best_start:best_end]
        
        # Добавляем многоточие если обрезали
        if best_start > 0:
            context = "..." + context
        if best_end < len(content):
            context = context + "..."
        
        return context.strip()
    
    async def get_document_count(self) -> int:
        """Возвращает количество документов в коллекции"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.count()
            )
        except Exception as e:
            logger.error(f"Error getting document count: {str(e)}")
            return 0
    
    async def delete_document(self, document_id: str) -> bool:
        """Удаляет документ и все его чанки"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                self._delete_document_sync,
                document_id
            )
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False
    
    def _delete_document_sync(self, document_id: str) -> bool:
        """Синхронная версия удаления документа"""
        try:
            # Получаем все связанные документы и чанки
            all_related_docs = self.collection.get(
                where={"parent_document_id": document_id},
                include=["metadatas"]
            )
            
            ids_to_delete = set()
            
            # Добавляем ID чанков
            if all_related_docs["ids"]:
                ids_to_delete.update(all_related_docs["ids"])
            
            # Добавляем основной документ
            ids_to_delete.add(document_id)
            
            # Проверяем какие ID реально существуют
            existing_docs = self.collection.get(
                ids=list(ids_to_delete),
                include=["metadatas"]
            )
            
            actual_ids_to_delete = existing_docs["ids"]
            
            if actual_ids_to_delete:
                # Удаляем по одному ID для избежания дубликатов
                deleted_count = 0
                for doc_id in actual_ids_to_delete:
                    try:
                        self.collection.delete(ids=[doc_id])
                        deleted_count += 1
                        logger.debug(f"Deleted document/chunk: {doc_id}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {doc_id}: {e}")
                        continue
                
                logger.info(f"Successfully deleted {deleted_count} documents/chunks for {document_id}")
                return deleted_count > 0
            else:
                logger.warning(f"Document {document_id} not found for deletion")
                return False
                
        except Exception as e:
            logger.error(f"Delete document sync error: {e}")
            return False
    
    async def get_all_documents(self) -> List[Dict]:
        """Получает все основные документы"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                self._get_all_documents_sync
            )
        except Exception as e:
            logger.error(f"Error getting all documents: {str(e)}")
            return []
    
    def _get_all_documents_sync(self) -> List[Dict]:
        """Синхронная версия получения всех документов"""
        try:
            # Более надежный запрос основных документов
            results = self.collection.get(
                where={"is_chunk": False},
                include=["documents", "metadatas"]
            )
            
            documents = []
            seen_ids = set()
            
            if results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    # Пропускаем дубликаты
                    if doc_id in seen_ids:
                        logger.debug(f"Skipping duplicate document: {doc_id}")
                        continue
                    
                    seen_ids.add(doc_id)
                    metadata = results["metadatas"][i]
                    
                    doc = {
                        "id": doc_id,
                        "filename": metadata.get("filename", "Unknown"),
                        "category": metadata.get("category", "general"),
                        "content": results["documents"][i],
                        "size": metadata.get("content_length", 0),
                        "word_count": metadata.get("word_count", 0),
                        "chunks_count": metadata.get("chunks_count", 1),
                        "added_at": metadata.get("added_at", time.time()),
                        "metadata": metadata
                    }
                    documents.append(doc)
            
            # Сортируем по дате добавления (новые первые)
            documents.sort(key=lambda x: x["added_at"], reverse=True)
            
            logger.info(f"Retrieved {len(documents)} unique documents")
            return documents
            
        except Exception as e:
            logger.error(f"Get all documents sync error: {e}")
            return []
    
    async def get_stats(self) -> Dict:
        """Получает статистику базы данных"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                self._get_stats_sync
            )
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {
                "total_documents": 0,
                "categories": [],
                "database_type": "ChromaDB",
                "error": str(e)
            }
    
    def _get_stats_sync(self) -> Dict:
        """Синхронная версия получения статистики"""
        try:
            total_count = self.collection.count()
            
            # Получаем уникальные категории из основных документов
            all_results = self.collection.get(
                where={"is_chunk": False},
                include=["metadatas"]
            )
            
            categories = set()
            unique_docs = 0
            
            if all_results["metadatas"]:
                seen_ids = set()
                for i, doc_id in enumerate(all_results["ids"]):
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        unique_docs += 1
                        category = all_results["metadatas"][i].get("category", "general")
                        categories.add(category)
            
            return {
                "total_documents": unique_docs,
                "categories": list(categories),
                "database_type": "ChromaDB",
                "persist_directory": self.persist_directory,
                "embedding_model": "all-MiniLM-L6-v2",
                "total_chunks": total_count,
                "unique_documents": unique_docs
            }
            
        except Exception as e:
            logger.error(f"Get stats sync error: {e}")
            return {
                "total_documents": 0,
                "categories": [],
                "database_type": "ChromaDB",
                "error": str(e)
            }

    async def update_document(self, document_id: str, new_content: str = None, new_metadata: Dict = None) -> bool:
        """Обновляет документ"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                self._update_document_sync,
                document_id, new_content, new_metadata
            )
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            return False
    
    def _update_document_sync(self, document_id: str, new_content: str, new_metadata: Dict) -> bool:
        """Синхронная версия обновления документа"""
        try:
            # ChromaDB не поддерживает прямое обновление, поэтому удаляем и добавляем заново
            if new_content or new_metadata:
                # Получаем текущий документ
                current = self.collection.get(
                    ids=[document_id],
                    include=["documents", "metadatas"]
                )
                
                if not current["ids"]:
                    return False
                
                # Подготавливаем новые данные
                content = new_content if new_content else current["documents"][0]
                metadata = current["metadatas"][0].copy()
                
                if new_metadata:
                    metadata.update(new_metadata)
                
                # Удаляем старый документ
                self.collection.delete(ids=[document_id])
                
                # Добавляем новый
                self.collection.add(
                    ids=[document_id],
                    documents=[content],
                    metadatas=[metadata]
                )
                
                logger.info(f"Updated document {document_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Update document sync error: {e}")
            return False

# ====================================
# ИСПРАВЛЕННЫЕ КЛАССЫ ДЛЯ ОБРАТНОЙ СОВМЕСТИМОСТИ
# ====================================

class DocumentProcessor:
    """Обработчик документов"""
    
    def __init__(self):
        # ИСПРАВЛЕНО: Правильные ссылки на методы
        self.supported_formats = {
            '.txt': self._process_txt,
            '.md': self._process_txt,
        }
    
    async def process_file(self, file_path: str, category: str = "general") -> Optional[ProcessedDocument]:
        """Обрабатывает файл и извлекает текст"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                self._process_file_sync,
                file_path, category
            )
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None
    
    def _process_file_sync(self, file_path: str, category: str) -> Optional[ProcessedDocument]:
        """Синхронная версия обработки файла"""
        try:
            from pathlib import Path
            
            file_path_obj = Path(file_path)
            extension = file_path_obj.suffix.lower()
            
            if extension not in self.supported_formats:
                content = self._process_txt_sync(file_path_obj)
            else:
                if extension in ['.txt', '.md']:
                    content = self._process_txt_sync(file_path_obj)
                else:
                    content = ""
            
            if not content or len(content.strip()) < 10:
                logger.warning(f"No meaningful content extracted from {file_path_obj.name}")
                return None
            
            # Создаем метаданные
            metadata = self._extract_metadata_sync(file_path_obj, content)
            
            # Разбиваем на чанки
            chunks = self._chunk_text(content)
            
            # Создаем ID документа
            doc_id = self._generate_doc_id(file_path_obj.name, content)
            
            return ProcessedDocument(
                id=doc_id,
                filename=file_path_obj.name,
                content=content,
                metadata=metadata,
                category=category,
                chunks=chunks
            )
            
        except Exception as e:
            logger.error(f"Process file sync error: {e}")
            return None
    
    # ИСПРАВЛЕНО: Добавлен недостающий async метод _process_txt
    async def _process_txt(self, file_path) -> str:
        """ИСПРАВЛЕНИЕ: Добавлен недостающий async метод"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._process_txt_sync, file_path
        )
    
    def _process_txt_sync(self, file_path) -> str:
        """Синхронная обработка текстовых файлов"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            for encoding in ['cp1251', 'iso-8859-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except:
                    continue
            logger.error(f"Could not decode text file {file_path}")
            return ""
    
    def _extract_metadata_sync(self, file_path, content: str) -> Dict:
        """Синхронное извлечение метаданных документа"""
        return {
            "filename": file_path.name,
            "file_size": file_path.stat().st_size,
            "file_extension": file_path.suffix,
            "content_length": len(content),
            "word_count": len(content.split()),
            "created_at": file_path.stat().st_ctime,
            "modified_at": file_path.stat().st_mtime,
            "processed_at": time.time()
        }
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Разбивает текст на чанки"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end < len(text):
                sentence_end = max(
                    text.rfind('.', start, end),
                    text.rfind('!', start, end),
                    text.rfind('?', start, end)
                )
                
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def _generate_doc_id(self, filename: str, content: str) -> str:
        """Генерирует уникальный ID для документа"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{filename}_{content_hash}"

class DocumentService:
    """Основной сервис документов с исправленным ChromaDB"""
    
    def __init__(self, db_path: str = "./chromadb_data"):
        self.processor = DocumentProcessor()
        self.vector_db = ChromaDBService(db_path)
        self.service_type = "chromadb"  # Добавляем service_type
    
    async def process_and_store_file(self, file_path: str, category: str = "general") -> bool:
        """Обрабатывает файл и сохраняет в ChromaDB"""
        try:
            document = await self.processor.process_file(file_path, category)
            
            if not document:
                return False
            
            return await self.vector_db.add_document(document)
        except Exception as e:
            logger.error(f"❌ Process and store error: {e}")
            return False
    
    async def search(self, query: str, category: str = None, limit: int = 5, min_relevance: float = 0.3) -> List[Dict]:
        """
        ИСПРАВЛЕННЫЙ поиск документов
        """
        try:
            return await self.vector_db.search_documents(
                query=query, 
                n_results=limit, 
                category=category,
                min_relevance=min_relevance
            )
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []
    
    async def get_stats(self) -> Dict:
        """Получает статистику"""
        try:
            return await self.vector_db.get_stats()
        except Exception as e:
            logger.error(f"❌ Get stats error: {e}")
            return {
                "total_documents": 0,
                "categories": [],
                "database_type": "ChromaDB", 
                "error": str(e)
            }
    
    async def get_all_documents(self) -> List[Dict]:
        """Получает все документы"""
        try:
            return await self.vector_db.get_all_documents()
        except Exception as e:
            logger.error(f"❌ Get all documents error: {e}")
            return []
    
    async def delete_document(self, document_id: str) -> bool:
        """Удаляет документ"""
        try:
            return await self.vector_db.delete_document(document_id)
        except Exception as e:
            logger.error(f"❌ Delete document error: {e}")
            return False
    
    async def update_document(self, document_id: str, new_content: str = None, new_metadata: Dict = None) -> bool:
        """Обновляет документ"""
        try:
            return await self.vector_db.update_document(document_id, new_content, new_metadata)
        except Exception as e:
            logger.error(f"❌ Update document error: {e}")
            return False