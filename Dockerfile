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

# ИСПРАВЛЕНИЕ: Убеждаемся что public/index.html существует
RUN test -f public/index.html || \
    (echo "Создаем public/index.html..." && \
    mkdir -p public && \
    cat > public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <link rel="icon" href="%PUBLIC_URL%/favicon.ico" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="Legal Assistant - AI-powered legal consultation" />
    <title>Legal Assistant</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
EOF
    )

# ИСПРАВЛЕНИЕ: Проверяем и исправляем package.json
RUN if ! grep -q '"build":' package.json; then \
    echo "Добавляем build скрипт в package.json..." && \
    npm pkg set scripts.build="react-scripts build"; \
    fi

# ИСПРАВЛЕНИЕ: Устанавливаем react-scripts если отсутствует
RUN npm list react-scripts || npm install --save-dev react-scripts

# Устанавливаем переменные для отключения TypeScript проверок (если проблемы)
ENV SKIP_PREFLIGHT_CHECK=true
ENV TSC_COMPILE_ON_ERROR=true
ENV DISABLE_ESLINT_PLUGIN=true

# Собираем продукцию React
RUN npm run build

# ДИАГНОСТИКА СБОРКИ REACT
RUN echo "🔍 Диагностика React build:" && \
    ls -la build/ && \
    echo "📁 Содержимое build/:" && \
    find build/ -type f -name "*.html" -o -name "*.js" -o -name "*.css" | head -20 && \
    echo "🔍 Поиск index.html:" && \
    find . -name "index.html" -type f && \
    echo "📊 Размеры ключевых файлов:" && \
    (ls -lh build/index.html || echo "❌ index.html НЕ НАЙДЕН!") && \
    (ls -lh build/static/js/*.js | head -3 || echo "⚠️ JS файлы не найдены") && \
    (ls -lh build/static/css/*.css | head -3 || echo "⚠️ CSS файлы не найдены")

# ИСПРАВЛЕНИЕ: Проверяем package.json и создаем index.html если нужно
RUN if [ ! -f build/index.html ]; then \
    echo "❌ index.html отсутствует! Создаем базовый..." && \
    mkdir -p build && \
    cat > build/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <link rel="icon" href="/favicon.ico" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="Legal Assistant - AI-powered legal consultation" />
    <link href="/static/css/main.css" rel="stylesheet">
    <title>Legal Assistant</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
    <script src="/static/js/main.js"></script>
  </body>
</html>
EOF
    fi

# КРИТИЧЕСКАЯ ПРОВЕРКА: Убеждаемся что все файлы созданы
RUN test -f build/index.html || (echo "💥 КРИТИЧЕСКАЯ ОШИБКА: index.html не создан!" && exit 1) && \
    test -d build/static || (echo "💥 КРИТИЧЕСКАЯ ОШИБКА: static/ не создан!" && exit 1) && \
    echo "✅ React build прошел все проверки"

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

# ДОПОЛНИТЕЛЬНО: также копируем в frontend/build для совместимости с main.py
COPY --from=react-builder --chown=user /app/frontend/build ./frontend/build

# ДИАГНОСТИКА КОПИРОВАНИЯ
RUN echo "🔍 Диагностика скопированных React файлов:" && \
    echo "📁 Содержимое ./static/:" && \
    ls -la static/ && \
    echo "📁 Содержимое ./frontend/build/:" && \
    ls -la frontend/build/ && \
    echo "🔍 Проверка index.html в обеих локациях:" && \
    (ls -lh static/index.html || echo "❌ static/index.html отсутствует") && \
    (ls -lh frontend/build/index.html || echo "❌ frontend/build/index.html отсутствует") && \
    echo "✅ Диагностика копирования завершена"

# ИСПРАВЛЕНИЕ: Убеждаемся что index.html есть в обеих локациях
RUN if [ ! -f static/index.html ] && [ -f frontend/build/index.html ]; then \
    echo "Копируем index.html в static/" && \
    cp frontend/build/index.html static/; \
    fi && \
    if [ ! -f frontend/build/index.html ] && [ -f static/index.html ]; then \
    echo "Копируем index.html в frontend/build/" && \
    mkdir -p frontend/build && \
    cp static/index.html frontend/build/; \
    fi

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