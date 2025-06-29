# Dockerfile для минимальной RAG системы - ОПТИМИЗИРОВАННАЯ ВЕРСИЯ
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
# REACT СТАТИКА (УПРОЩЕННО)
# ====================================

# Копируем React build если есть, иначе создаем заглушку
COPY --chown=user frontend/build ./static 2>/dev/null || \
    (mkdir -p static && echo '<!DOCTYPE html><html><head><title>Minimal RAG</title></head><body><h1>Minimal RAG API</h1><p>API Documentation: <a href="/docs">/docs</a></p></body></html>' > static/index.html)

# Альтернативный путь для совместимости
RUN cp -r static frontend/build 2>/dev/null || mkdir -p frontend/build

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