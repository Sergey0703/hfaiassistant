# Dockerfile для минимальной RAG системы - ИСПРАВЛЕННАЯ ВЕРСИЯ
# Single-stage build: Python + React статика + минимальные зависимости
# Целевое потребление: <1GB RAM

# ====================================
# БАЗОВЫЙ ОБРАЗ - PYTHON SLIM
# ====================================
FROM python:3.11-slim

# Установка только критических системных зависимостей
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Создание пользователя для HuggingFace Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Создание рабочей директории
WORKDIR $HOME/app

# ====================================
# УСТАНОВКА PYTHON ЗАВИСИМОСТЕЙ
# ====================================

# Копирование requirements и установка зависимостей
COPY --chown=user requirements.txt .

# Установка PyTorch CPU версии (оптимизированная)
RUN pip install --user --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Установка остальных зависимостей
RUN pip install --user --no-cache-dir -r requirements.txt

# Проверяем что критические импорты работают
RUN python -c "import fastapi, transformers, sentence_transformers, chromadb; print('✅ All critical imports OK')"

# ====================================
# КОПИРОВАНИЕ ПРИЛОЖЕНИЯ
# ====================================

# Копирование кода бэкенда
COPY --chown=user backend/ .

# Создание необходимых директорий
RUN mkdir -p logs chromadb_data uploads temp .cache

# ====================================
# REACT СТАТИКА (УПРОЩЕННАЯ БЕЗОПАСНАЯ ВЕРСИЯ)
# ====================================

# Создаем директории для статических файлов
RUN mkdir -p static frontend/build

# Создаем заглушку HTML файла для API-only режима
RUN echo '<!DOCTYPE html>\
<html>\
<head>\
    <title>Minimal RAG System</title>\
    <meta charset="utf-8">\
    <meta name="viewport" content="width=device-width, initial-scale=1">\
    <style>\
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }\
        h1 { color: #2c3e50; }\
        a { color: #3498db; text-decoration: none; }\
        a:hover { text-decoration: underline; }\
        .links { background: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }\
        .status { color: #7f8c8d; font-style: italic; }\
    </style>\
</head>\
<body>\
    <h1>🏛️ Minimal RAG System</h1>\
    <p>Legal Assistant API with FLAN-T5 Small and Sentence Transformers</p>\
    \
    <div class="links">\
        <h3>🔗 Available Endpoints:</h3>\
        <ul>\
            <li><a href="/docs">📖 Interactive API Documentation</a></li>\
            <li><a href="/health">💚 System Health Check</a></li>\
            <li><a href="/model-status">🤖 Model Status</a></li>\
            <li><a href="/api-status">⚙️ API Status</a></li>\
        </ul>\
    </div>\
    \
    <div class="links">\
        <h3>🚀 Quick Start:</h3>\
        <ul>\
            <li>POST <code>/api/user/chat</code> - Chat with legal assistant</li>\
            <li>POST <code>/api/user/search</code> - Search documents</li>\
            <li>GET <code>/api/admin/documents</code> - Manage documents</li>\
        </ul>\
    </div>\
    \
    <hr>\
    <p class="status">React frontend not built. API-only mode active.</p>\
    <p class="status">Memory target: &lt;1GB RAM | Models: FLAN-T5 Small + all-MiniLM-L6-v2</p>\
</body>\
</html>' > static/index.html

# Копируем в оба места для совместимости
RUN cp static/index.html frontend/build/index.html

# Примечание: Если вы хотите использовать React frontend,
# соберите его командой 'npm run build' в папке frontend/
# и пересоберите Docker образ

# ====================================
# НАСТРОЙКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ
# ====================================
ENV PYTHONPATH=$HOME/app
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV USE_CHROMADB=true

# HuggingFace Spaces settings
ENV HF_SPACES=true
ENV TRANSFORMERS_CACHE=$HOME/app/.cache
ENV HF_HOME=$HOME/app/.cache

# Минимальная RAG конфигурация
ENV LLM_MODEL=google/flan-t5-small
ENV LLM_MAX_TOKENS=150
ENV LLM_TEMPERATURE=0.3
ENV LLM_TIMEOUT=20

# Embedding настройки
ENV EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV MAX_CONTEXT_DOCUMENTS=2
ENV CONTEXT_TRUNCATE_LENGTH=300

# React frontend settings
ENV REACT_BUILD_PATH=$HOME/app/static
ENV SERVE_REACT=true

# Порт для HuggingFace Spaces
EXPOSE 7860

# ====================================
# HEALTHCHECK
# ====================================
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ====================================
# КОМАНДА ЗАПУСКА
# ====================================
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]