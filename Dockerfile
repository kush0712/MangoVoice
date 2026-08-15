FROM python:3.12-slim

# Install system deps for lancedb/tantivy native binaries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache-bust arg
ARG CACHE_BUST=7

# ── BAKE THE LANCEDB INDEX INTO THE DOCKER IMAGE! ────────────────────────────
# By doing this at build-time, we don't need a Railway Volume at all!
# This avoids the 434MB volume limit completely.
RUN mkdir -p /app/data && \
    echo "Downloading pre-built index into Docker image..." && \
    curl -sSL "https://github.com/kush0712/MangoVoice/releases/download/v1.0.0-index/mangovoice-index-v2.tar.gz" | tar -xzf - -C /app/data --warning=no-unknown-keyword && \
    echo "Index baked successfully!"

COPY . .

# Make startup script executable
RUN chmod +x startup.sh

ENV PYTHONUNBUFFERED=1
ENV APP_ENV=production
ENV INDEX_PATH=/app/data/lancedb
ENV FASTEMBED_CACHE_DIR=/tmp/fastembed_cache

EXPOSE 8000

CMD ["./startup.sh"]
