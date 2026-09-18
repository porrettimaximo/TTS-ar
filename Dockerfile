FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app/api:$PYTHONPATH

WORKDIR /app

# Instalar dependencias del sistema requeridas por Piper, ONNX Runtime y espeak-ng
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    libgomp1 \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y cachear capas de pip
COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r ./api/requirements.txt

# Copiar código fuente y assets
COPY api/ ./api/
COPY voices/ ./voices/

# Descargar modelos durante la construcción para que la imagen quede lista para producción
RUN python voices/download_voices.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/audio/voices || exit 1

CMD ["uvicorn", "main:app", "--app-dir", "api", "--host", "0.0.0.0", "--port", "8000"]
