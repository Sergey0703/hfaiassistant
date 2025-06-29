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
# REACT СТАТИКА (ВОССТАНОВЛЕНА ПОДДЕРЖКА ВАШЕГО FRONTEND)
# ====================================

# Создаем директории для статических файлов
RUN mkdir -p static frontend/build

# Сначала создаем временную заглушку (будет перезаписана вашим React build)
RUN echo '<!DOCTYPE html><html><head><title>Loading...</title></head><body><h1>Loading React App...</h1></body></html>' > static/index.html

# Копируем ваш React build если он существует
# Используем отдельные команды для безопасности
RUN echo "Checking for React build..."
COPY --chown=user frontend/ ./frontend_source/
RUN if [ -d "frontend_source/build" ] && [ -f "frontend_source/build/index.html" ]; then \
    echo "✅ React build found - installing your frontend..."; \
    cp -r frontend_source/build/* static/; \
    cp -r frontend_source/build/* frontend/build/; \
    echo "✅ Your React frontend installed successfully"; \
    else \
    echo "⚠️ No React build found in frontend/build/"; \
    echo "📝 To use your React frontend:"; \
    echo "   1. cd frontend"; \
    echo "   2. npm install"; \
    echo "   3. npm run build"; \
    echo "   4. Rebuild Docker image"; \
    echo "⚙️ Using API-only mode for now"; \
    echo '<!DOCTYPE html>\
<html>\
<head>\
    <title>Legal Assistant API</title>\
    <meta charset="utf-8">\
    <meta name="viewport" content="width=device-width, initial-scale=1">\
</head>\
<body>\
    <h1>🏛️ Legal Assistant API</h1>\
    <p><strong>Your React frontend will appear here after building</strong></p>\
    <h3>📖 API Documentation:</h3>\
    <ul>\
        <li><a href="/docs">Interactive API Docs</a></li>\
        <li><a href="/health">Health Check</a></li>\
    </ul>\
    <h3>🔧 To enable your React frontend:</h3>\
    <ol>\
        <li>cd frontend</li>\
        <li>npm install</li>\
        <li>npm run build</li>\
        <li>Rebuild Docker image</li>\
    </ol>\
</body>\
</html>' > static/index.html; \
    cp static/index.html frontend/build/index.html; \
    fi

# Очищаем временные файлы
RUN rm -rf frontend_source

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