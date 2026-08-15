#!/bin/bash
# MangoVoice — Railway startup script
#
# Strategy:
#   1. Start uvicorn IMMEDIATELY (health check passes right away)
#   2. In BACKGROUND: stream pre-built LanceDB index from GitHub Releases directly to tar
#      (Zero intermediate disk usage, no OOM, clean unpack)

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
        sleep 3  # brief pause for uvicorn to bind the port
        echo ">>> [BG] Streaming and extracting pre-built LanceDB index directly..."

        # Clean any partial failed extractions
        rm -rf "$INDEX_PATH"

        # Stream directly from curl to tar (uses 0 extra temp disk space!)
        curl -sSL --retry 5 --retry-delay 3 "$INDEX_URL" | tar -xzf - -C "$DATA_DIR"

        if [ -d "$INDEX_PATH" ]; then
            touch "$LOCK_FILE"
            echo ">>> [BG] Index ready at $INDEX_PATH — MangoVoice is fully ready!"
        else
            echo ">>> [BG] Extraction failed or index folder not found" >&2
        fi
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
