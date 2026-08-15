#!/bin/bash
# MangoVoice — Railway startup script
#
# Strategy:
#   1. Start uvicorn IMMEDIATELY (health check passes right away)
#   2. In BACKGROUND: stream pre-built LanceDB index from GitHub Releases directly to tar

set -e

PORT="${PORT:-8000}"
DATA_DIR="${DATA_DIR:-/data}"
INDEX_PATH="${DATA_DIR}/lancedb"
FASTEMBED_CACHE="${DATA_DIR}/fastembed_cache"
LOCK_FILE="${DATA_DIR}/.index_ready"

INDEX_URL="https://github.com/kush0712/MangoVoice/releases/download/v1.0.0-index/mangovoice-index-v2.tar.gz"

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
        echo ">>> [BG] Pre-extraction disk usage (checking for hidden files taking up space):"
        df -h "$DATA_DIR"
        du -sh "$DATA_DIR"/* 2>/dev/null || true

        echo ">>> [BG] Aggressively wiping EVERYTHING in $DATA_DIR to free up 100% space..."
        # Wipe EVERYTHING in the volume EXCEPT what we need
        find "$DATA_DIR" -mindepth 1 -delete || true

        echo ">>> [BG] Disk space after wipe:"
        df -h "$DATA_DIR"

        echo ">>> [BG] Streaming and extracting pre-built LanceDB index directly..."
        
        # We suppress warnings about macOS xattrs, and stream directly
        curl -sSL --retry 5 --retry-delay 3 "$INDEX_URL" | tar -xzf - -C "$DATA_DIR" --warning=no-unknown-keyword

        if [ -d "$INDEX_PATH" ]; then
            touch "$LOCK_FILE"
            echo ">>> [BG] Final disk space:"
            df -h "$DATA_DIR"
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
