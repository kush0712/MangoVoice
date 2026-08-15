#!/bin/bash
# MangoVoice — Railway startup script
# Starts uvicorn FIRST so the health check passes immediately,
# then builds the LanceDB index in the background on first boot.

set -e

PORT="${PORT:-8000}"
DATA_DIR="${DATA_DIR:-/data}"
INDEX_PATH="${DATA_DIR}/lancedb"
FASTEMBED_CACHE="${DATA_DIR}/fastembed_cache"
LOCK_FILE="${DATA_DIR}/.index_ready"
RAW_PASSAGES="data/raw_passages_with_seeds.jsonl"

echo "=== MangoVoice Startup ==="
echo "DATA_DIR: $DATA_DIR"
echo "PORT: $PORT"

# Create dirs on the volume
mkdir -p "$INDEX_PATH"
mkdir -p "$FASTEMBED_CACHE"

# Export paths so the app reads from the volume
export INDEX_PATH="$INDEX_PATH"
export FASTEMBED_CACHE_DIR="$FASTEMBED_CACHE"

# ── Build index in BACKGROUND if not already done ────────────────────────────
if [ ! -f "$LOCK_FILE" ]; then
    echo ">>> Index not found — will build in background after uvicorn starts."
    (
        sleep 10  # give uvicorn time to start and pass health check
        echo ">>> [BG] Starting index build from $RAW_PASSAGES ..."

        if [ ! -f "$RAW_PASSAGES" ]; then
            echo "ERROR: $RAW_PASSAGES not found." >&2
            exit 1
        fi

        INDEX_PATH="$INDEX_PATH" \
        FASTEMBED_CACHE_DIR="$FASTEMBED_CACHE" \
        python -m ingestion.build_index \
            --input "$RAW_PASSAGES" \
            --output "$INDEX_PATH" \
            --strategy parent_child \
            --table chunks \
            --batch-size 256 \
            --max-passages 0

        touch "$LOCK_FILE"
        echo ">>> [BG] Index built successfully at $INDEX_PATH"
    ) &
else
    echo ">>> Existing index found at $INDEX_PATH — skipping build."
fi

# ── Start FastAPI immediately ─────────────────────────────────────────────────
echo ">>> Starting uvicorn on port $PORT ..."
exec uvicorn api.index:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    --log-level info
