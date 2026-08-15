#!/bin/bash
# MangoVoice — Railway startup script
#
# Strategy:
#   1. Start uvicorn IMMEDIATELY (health check passes right away)
#   2. In BACKGROUND: download pre-built LanceDB index from GitHub Releases
#      (avoids building 63k embeddings at runtime = no OOM on 512MB plan)
#
# Pre-built index: github.com/kush0712/MangoVoice/releases/tag/v1.0.0-index
# Contains: 63615 chunks, 384-dim multilingual embeddings (IVF_PQ + BM25)

set -e

PORT="${PORT:-8000}"
DATA_DIR="${DATA_DIR:-/data}"
INDEX_PATH="${DATA_DIR}/lancedb"
FASTEMBED_CACHE="${DATA_DIR}/fastembed_cache"
LOCK_FILE="${DATA_DIR}/.index_ready"

# GitHub Releases download URL for the pre-built index tarball
INDEX_URL="https://github.com/kush0712/MangoVoice/releases/download/v1.0.0-index/mangovoice-index.tar.gz"

echo "=== MangoVoice Startup ==="
echo "DATA_DIR=$DATA_DIR  PORT=$PORT"

# Create dirs on the volume
mkdir -p "$DATA_DIR"
mkdir -p "$FASTEMBED_CACHE"

# Export paths so the app reads from the volume
export INDEX_PATH="$INDEX_PATH"
export FASTEMBED_CACHE_DIR="$FASTEMBED_CACHE"

# ── Download pre-built index in BACKGROUND ───────────────────────────────────
if [ ! -f "$LOCK_FILE" ]; then
    echo ">>> Index not found — will download pre-built index in background."
    (
        sleep 5  # brief pause for uvicorn to bind the port
        echo ">>> [BG] Downloading pre-built LanceDB index (~346MB)..."

        TARBALL="${DATA_DIR}/index.tar.gz"
        curl -L --retry 3 --retry-delay 5 \
            -o "$TARBALL" \
            "$INDEX_URL"

        echo ">>> [BG] Extracting index to $DATA_DIR ..."
        tar -xzf "$TARBALL" -C "$DATA_DIR"
        rm -f "$TARBALL"

        touch "$LOCK_FILE"
        echo ">>> [BG] Index ready at $INDEX_PATH — app is now fully operational!"
    ) &
    disown
else
    echo ">>> Existing index found at $INDEX_PATH — skipping download."
fi

# ── Start FastAPI immediately ─────────────────────────────────────────────────
echo ">>> Starting uvicorn on 0.0.0.0:$PORT ..."
exec uvicorn api.index:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    --log-level info
