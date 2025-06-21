# Dockerfile для HuggingFace Spaces с исправлением прав доступа
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

# Копирование requirements и установка зависимостей
COPY --chown=user requirements.txt .

# Установка Python зависимостей
# Устанавливаем torch сначала для стабильности
RUN pip install --user --no-cache-dir torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu

# Фиксируем версию numpy для совместимости
RUN pip install --user --no-cache-dir "numpy<2.0.0"

# Остальные зависимости
RUN pip install --user --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY --chown=user backend/ .

# Создание необходимых директорий с правильными правами
RUN mkdir -p logs simple_db chromadb_data uploads temp backups .cache

# Настройка переменных окружения для HuggingFace Spaces
ENV PYTHONPATH=$HOME/app
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV USE_CHROMADB=true
ENV OLLAMA_ENABLED=false
ENV LLM_DEMO_MODE=false
ENV CORS_ORIGINS=["*"]

# HuggingFace Spaces specific settings
ENV HF_SPACES=true
ENV TRANSFORMERS_CACHE=$HOME/app/.cache
ENV HF_HOME=$HOME/app/.cache
ENV TORCH_HOME=$HOME/app/.cache

# Порт для HuggingFace Spaces (обязательно 7860)
EXPOSE 7860

# Команда запуска
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]