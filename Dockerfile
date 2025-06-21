# Dockerfile для HuggingFace Spaces с Llama-3 GPTQ
FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование requirements и установка зависимостей
COPY requirements.txt .

# Установка Python зависимостей
# Устанавливаем torch сначала для стабильности
RUN pip install --no-cache-dir torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu

# Остальные зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY backend/ .

# Создание необходимых директорий
RUN mkdir -p logs simple_db chromadb_data uploads temp backups

# Настройка переменных окружения для HuggingFace Spaces
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV USE_CHROMADB=true
ENV OLLAMA_ENABLED=false
ENV LLM_DEMO_MODE=false
ENV CORS_ORIGINS=["*"]

# HuggingFace Spaces specific settings
ENV HF_SPACES=true
ENV TRANSFORMERS_CACHE=/app/.cache
ENV HF_HOME=/app/.cache

# Порт для HuggingFace Spaces (обязательно 7860)
EXPOSE 7860

# Создание cache директории
RUN mkdir -p /app/.cache

# Команда запуска
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]