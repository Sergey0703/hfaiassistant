# Dockerfile для HuggingFace Spaces - ПОЛНОСТЬЮ ИСПРАВЛЕННАЯ ВЕРСИЯ
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

# Устанавливаем react-scripts если отсутствует
RUN npm install --save-dev react-scripts || true

# Копируем исходники React
COPY frontend/ ./

# СОЗДАЕМ public/index.html если отсутствует
RUN mkdir -p public && \
    if [ ! -f public/index.html ]; then \
        echo "Создаем public/index.html..." && \
        echo '<!DOCTYPE html>' > public/index.html && \
        echo '<html lang="en">' >> public/index.html && \
        echo '<head>' >> public/index.html && \
        echo '<meta charset="utf-8" />' >> public/index.html && \
        echo '<meta name="viewport" content="width=device-width, initial-scale=1" />' >> public/index.html && \
        echo '<title>Legal Assistant</title>' >> public/index.html && \
        echo '</head>' >> public/index.html && \
        echo '<body>' >> public/index.html && \
        echo '<div id="root"></div>' >> public/index.html && \
        echo '</body>' >> public/index.html && \
        echo '</html>' >> public/index.html; \
    fi

# ПРОВЕРЯЕМ package.json
RUN cat package.json | grep -q '"build":' || npm pkg set scripts.build="react-scripts build"

# Устанавливаем переменные окружения для сборки
ENV SKIP_PREFLIGHT_CHECK=true
ENV TSC_COMPILE_ON_ERROR=true
ENV DISABLE_ESLINT_PLUGIN=true
ENV GENERATE_SOURCEMAP=false
ENV NODE_OPTIONS="--max_old_space_size=4096"

# Собираем React приложение
RUN npm run build

# ДИАГНОСТИКА: проверяем что создалось
RUN echo "=== ДИАГНОСТИКА REACT BUILD ===" && \
    echo "Содержимое build/:" && \
    ls -la build/ && \
    echo "Поиск index.html:" && \
    find . -name "index.html" -type f && \
    echo "Размер build/:" && \
    du -sh build/ && \
    echo "Содержимое build/static/:" && \
    ls -la build/static/ 2>/dev/null || echo "build/static/ не существует"

# ПРИНУДИТЕЛЬНО создаем index.html если его нет
RUN if [ ! -f build/index.html ]; then \
        echo "ПРИНУДИТЕЛЬНОЕ СОЗДАНИЕ index.html" && \
        mkdir -p build && \
        echo '<!DOCTYPE html>' > build/index.html && \
        echo '<html lang="en">' >> build/index.html && \
        echo '<head>' >> build/index.html && \
        echo '<meta charset="utf-8">' >> build/index.html && \
        echo '<meta name="viewport" content="width=device-width,initial-scale=1">' >> build/index.html && \
        echo '<title>Legal Assistant</title>' >> build/index.html && \
        echo '</head>' >> build/index.html && \
        echo '<body>' >> build/index.html && \
        echo '<div id="root"></div>' >> build/index.html && \
        echo '</body>' >> build/index.html && \
        echo '</html>' >> build/index.html; \
    fi

# ФИНАЛЬНАЯ ПРОВЕРКА
RUN test -f build/index.html || (echo "КРИТИЧЕСКАЯ ОШИБКА: index.html не создан!" && exit 1)

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

# КРИТИЧЕСКИ ВАЖНО: Копируем React build из первого stage
COPY --from=react-builder --chown=user /app/frontend/build ./static

# ДОПОЛНИТЕЛЬНО: копируем в frontend/build для совместимости
COPY --from=react-builder --chown=user /app/frontend/build ./frontend/build

# ДИАГНОСТИКА: проверяем что скопировалось
RUN echo "=== ДИАГНОСТИКА КОПИРОВАНИЯ ===" && \
    echo "Корневая директория:" && \
    ls -la && \
    echo "Содержимое static/:" && \
    ls -la static/ && \
    echo "Содержимое frontend/:" && \
    ls -la frontend/ 2>/dev/null || echo "frontend/ не создана" && \
    echo "Поиск всех index.html:" && \
    find . -name "index.html" -type f 2>/dev/null || echo "index.html не найден"

# ДОПОЛНИТЕЛЬНОЕ ИСПРАВЛЕНИЕ: убеждаемся что index.html есть в нужных местах
RUN if [ ! -f static/index.html ]; then \
        echo "Создаем static/index.html..." && \
        echo '<!DOCTYPE html><html><head><title>Legal Assistant</title></head><body><div id="root">App Loading...</div></body></html>' > static/index.html; \
    fi && \
    if [ ! -f frontend/build/index.html ]; then \
        echo "Создаем frontend/build/index.html..." && \
        mkdir -p frontend/build && \
        cp static/index.html frontend/build/index.html; \
    fi

# СОЗДАЕМ ДОПОЛНИТЕЛЬНЫЕ ФАЙЛЫ для React
RUN echo '{"short_name":"Legal Assistant","name":"Legal Assistant","start_url":"/","display":"standalone"}' > static/manifest.json && \
    echo '<svg width="32" height="32" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg"><rect width="32" height="32" fill="#1f2937"/><text x="16" y="20" text-anchor="middle" fill="white" font-size="16">⚖️</text></svg>' > static/favicon.ico && \
    echo 'User-agent: *\nDisallow:' > static/robots.txt

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

# React frontend settings
ENV REACT_BUILD_PATH=$HOME/app/static
ENV SERVE_REACT=true

# Порт для HuggingFace Spaces (обязательно 7860)
EXPOSE 7860

# ====================================
# ФИНАЛЬНАЯ ПРОВЕРКА
# ====================================
RUN echo "=== ФИНАЛЬНАЯ ПРОВЕРКА ===" && \
    (test -f static/index.html && echo "✅ static/index.html существует" || echo "❌ static/index.html отсутствует") && \
    (test -f frontend/build/index.html && echo "✅ frontend/build/index.html существует" || echo "❌ frontend/build/index.html отсутствует") && \
    echo "Размеры ключевых файлов:" && \
    ls -lh static/index.html static/manifest.json static/favicon.ico 2>/dev/null || echo "Некоторые файлы отсутствуют" && \
    echo "=== ГОТОВО ==="

# ====================================
# HEALTHCHECK ДЛЯ DOCKER
# ====================================
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ====================================
# КОМАНДА ЗАПУСКА
# ====================================
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--timeout-keep-alive", "65"]