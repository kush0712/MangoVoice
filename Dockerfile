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

# Cache-bust arg — increment to force Railway to re-copy source files
ARG CACHE_BUST=3
COPY . .

# Make startup script executable
RUN chmod +x startup.sh

ENV PYTHONUNBUFFERED=1
ENV APP_ENV=production

EXPOSE 8000

CMD ["./startup.sh"]
