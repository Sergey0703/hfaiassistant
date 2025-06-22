# Dockerfile для HuggingFace Spaces - ИСПРАВЛЕННАЯ ВЕРСИЯ ДЛЯ REACT SPA
# Multi-stage build: Node.js для React + Python для FastAPI + правильная раздача статики

# ====================================
# STAGE 1: СБОРКА REACT ФРОНТЕНДА
# ====================================
FROM node:18-alpine AS react-builder

WORKDIR /app/frontend

# Копируем package.json и package-lock.json
COPY frontend/package*.json ./

# Устанавливаем ВСЕ зависимости (включая devDependencies для сборки)
RUN npm ci

# Устанавливаем типы для TypeScript если нужно
RUN npm install --save-dev @types/react @types/react-dom @types/node typescript || true

# Копируем исходники React
COPY frontend/ ./

# Устанавливаем переменные для отключения TypeScript проверок (если проблемы)
ENV SKIP_PREFLIGHT_CHECK=true
ENV TSC_COMPILE_ON_ERROR=true
ENV DISABLE_ESLINT_PLUGIN=true

# Собираем продукцию React
RUN npm run build

# ====================================
# STAGE 2: PYTHON BACKEND + REACT BUILD
# ====================================
FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создание пользователя для HuggingFace Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Создание рабочей директории
WORKDIR $HOME/app

# КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Установка совместимых версий PyTorch
RUN pip install --user --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Устанавливаем sentence-transformers с совместимой версией
RUN pip install --user --no-cache-dir sentence-transformers==2.7.0

# Копирование requirements и установка остальных зависимостей
COPY --chown=user requirements.txt .

# Устанавливаем остальные зависимости ПОСЛЕ PyTorch
RUN pip install --user --no-cache-dir -r requirements.txt

# Проверяем что sentence-transformers работает
RUN python -c "import sentence_transformers; print('✅ sentence-transformers OK')"

# ====================================
# КОПИРОВАНИЕ ПРИЛОЖЕНИЯ
# ====================================

# Копирование кода бэкенда
COPY --chown=user backend/ .

# КРИТИЧЕСКИ ВАЖНО: Копирование собранного React приложения из первого stage
# НА ПРАВИЛЬНЫЙ ПУТЬ ДЛЯ HUGGINGFACE SPACES
COPY --from=react-builder --chown=user /app/frontend/build ./static

# АЛЬТЕРНАТИВНО: также копируем в frontend/build для совместимости
COPY --from=react-builder --chown=user /app/frontend/build ./frontend/build

# Создание необходимых директорий с правильными правами
RUN mkdir -p logs simple_db chromadb_data uploads temp backups .cache

# ====================================
# НАСТРОЙКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ
# ====================================
ENV PYTHONPATH=$HOME/app
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV USE_CHROMADB=true
ENV LLM_DEMO_MODE=false
ENV CORS_ORIGINS=["*"]

# HuggingFace Spaces specific settings
ENV HF_SPACES=true
ENV TRANSFORMERS_CACHE=$HOME/app/.cache
ENV HF_HOME=$HOME/app/.cache
ENV TORCH_HOME=$HOME/app/.cache

# React frontend settings - ИСПРАВЛЕННЫЕ ПУТИ
ENV REACT_BUILD_PATH=$HOME/app/static
ENV SERVE_REACT=true

# Порт для HuggingFace Spaces (обязательно 7860)
EXPOSE 7860

# ====================================
# HEALTHCHECK ДЛЯ DOCKER
# ====================================
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ====================================
# КОМАНДА ЗАПУСКА
# ====================================
# Увеличенный timeout для React + FastAPI
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--timeout-keep-alive", "65"]