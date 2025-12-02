FROM python:3.11-slim

# Отключаем буферизацию вывода Python (важно для логов Cloud Run)
ENV PYTHONUNBUFFERED=1

# Рабочая директория внутри контейнера
WORKDIR /app

# Устанавливаем системные зависимости, нужные для opencv / PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Сначала копируем только requirements для кеширования layer'ов
COPY requirements.txt /app/requirements.txt

# Устанавливаем зависимости (CPU-версия PyTorch уже заложена в requirements.txt)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Копируем всё приложение
COPY . /app

# Cloud Run передаст порт через переменную окружения PORT
ENV PORT=8080

# Команда запуска — gunicorn с увеличенным timeout для инициализации EasyOCR
# --timeout 600: 10 минут на обработку запроса (включая первую инициализацию EasyOCR и обработку больших файлов)
# --graceful-timeout 30: время на graceful shutdown
CMD exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 600 --graceful-timeout 30 app:app



