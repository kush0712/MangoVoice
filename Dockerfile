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

# Copy application source
COPY . .

# Make startup script executable
RUN chmod +x startup.sh

# Railway injects $PORT; uvicorn picks it up via startup.sh
ENV PYTHONUNBUFFERED=1
ENV APP_ENV=production

EXPOSE 8000

CMD ["./startup.sh"]
