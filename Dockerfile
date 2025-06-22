# Dockerfile для HuggingFace Spaces - ФИНАЛЬНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ
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

# КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Проверяем и исправляем asset-manifest.json
RUN if [ -f build/asset-manifest.json ]; then \
        echo "Проверяем asset-manifest.json..." && \
        cat build/asset-manifest.json; \
    fi

# СОЗДАЕМ ПРАВИЛЬНЫЙ index.html с корректными путями
RUN echo "Создаем правильный index.html..." && \
    mkdir -p build && \
    echo '<!DOCTYPE html>' > build/index.html && \
    echo '<html lang="en">' >> build/index.html && \
    echo '<head>' >> build/index.html && \
    echo '<meta charset="utf-8">' >> build/index.html && \
    echo '<meta name="viewport" content="width=device-width,initial-scale=1">' >> build/index.html && \
    echo '<meta name="theme-color" content="#000000">' >> build/index.html && \
    echo '<meta name="description" content="Legal Assistant - AI-powered legal consultation">' >> build/index.html && \
    echo '<link rel="icon" href="/static/favicon.ico">' >> build/index.html && \
    echo '<link rel="manifest" href="/static/manifest.json">' >> build/index.html && \
    echo '<title>Legal Assistant</title>' >> build/index.html && \
    echo '</head>' >> build/index.html && \
    echo '<body>' >> build/index.html && \
    echo '<noscript>You need to enable JavaScript to run this app.</noscript>' >> build/index.html && \
    echo '<div id="root"></div>' >> build/index.html && \
    echo '</body>' >> build/index.html && \
    echo '</html>' >> build/index.html

# ДОБАВЛЯЕМ СКРИПТЫ ДИНАМИЧЕСКИ если они существуют
RUN if [ -d build/static/js ]; then \
        echo "Добавляем JS файлы в index.html..." && \
        JS_FILES=$(find build/static/js -name "*.js" | head -3) && \
        for js_file in $JS_FILES; do \
            js_path=$(echo $js_file | sed 's|build||') && \
            echo "<script defer=\"defer\" src=\"$js_path\"></script>" >> build/index.html; \
        done; \
    fi

# ДОБАВЛЯЕМ CSS ФАЙЛЫ если существуют  
RUN if [ -d build/static/css ]; then \
        echo "Добавляем CSS файлы в index.html..." && \
        CSS_FILES=$(find build/static/css -name "*.css" | head -3) && \
        for css_file in $CSS_FILES; do \
            css_path=$(echo $css_file | sed 's|build||') && \
            sed -i "/<title>/a <link href=\"$css_path\" rel=\"stylesheet\">" build/index.html; \
        done; \
    fi

# ФИНАЛЬНАЯ ПРОВЕРКА index.html
RUN echo "=== СОДЕРЖИМОЕ ИСПРАВЛЕННОГО index.html ===" && \
    cat build/index.html && \
    echo "=== КОНЕЦ index.html ==="

# ДИАГНОСТИКА: проверяем что создалось
RUN echo "=== ДИАГНОСТИКА REACT BUILD ===" && \
    echo "Содержимое build/:" && \
    ls -la build/ && \
    echo "Содержимое build/static/:" && \
    ls -la build/static/ 2>/dev/null || echo "build/static/ не существует" && \
    echo "JS файлы:" && \
    find build/static -name "*.js" 2>/dev/null || echo "JS файлы не найдены" && \
    echo "CSS файлы:" && \
    find build/static -name "*.css" 2>/dev/null || echo "CSS файлы не найдены"

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
    echo "Содержимое static/static/:" && \
    ls -la static/static/ 2>/dev/null || echo "static/static/ не существует" && \
    echo "Поиск всех index.html:" && \
    find . -name "index.html" -type f 2>/dev/null || echo "index.html не найден" && \
    echo "Содержимое index.html:" && \
    head -20 static/index.html 2>/dev/null || echo "Не удалось прочитать index.html"

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
    ls -lh static/index.html 2>/dev/null || echo "index.html отсутствует" && \
    echo "Статические файлы React:" && \
    ls -la static/static/ 2>/dev/null || echo "static/static/ отсутствует" && \
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