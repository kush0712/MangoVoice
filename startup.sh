#!/bin/bash
# MangoVoice — Railway startup script
#
# Strategy:
#   1. Start uvicorn IMMEDIATELY so the health check passes
#   2. Build index in BACKGROUND with a passage limit that fits in 512MB RAM
#      (fastembed ONNX model = ~350MB, so we cap passages to stay under limit)
#
# Memory budget on Railway $5 plan (512MB):
#   fastembed model:  ~350MB
#   Python baseline:   ~50MB
#   5000 chunks data:  ~10MB
#   ──────────────────────────
#   Total:            ~410MB  ✅ fits

set -e

PORT="${PORT:-8000}"
DATA_DIR="${DATA_DIR:-/data}"
INDEX_PATH="${DATA_DIR}/lancedb"
FASTEMBED_CACHE="${DATA_DIR}/fastembed_cache"
LOCK_FILE="${DATA_DIR}/.index_ready"
RAW_PASSAGES="data/raw_passages_with_seeds.jsonl"

# Passage limit that fits in 512MB RAM
MAX_PASSAGES="${MAX_PASSAGES:-5000}"

echo "=== MangoVoice Startup ==="
echo "DATA_DIR=$DATA_DIR  PORT=$PORT  MAX_PASSAGES=$MAX_PASSAGES"

# Create dirs on the volume
mkdir -p "$INDEX_PATH"
mkdir -p "$FASTEMBED_CACHE"

# Export paths so the app reads from the volume
export INDEX_PATH="$INDEX_PATH"
export FASTEMBED_CACHE_DIR="$FASTEMBED_CACHE"

# ── Build index in BACKGROUND (after uvicorn has started) ────────────────────
if [ ! -f "$LOCK_FILE" ]; then
    echo ">>> Index not found — will build in background after uvicorn starts."
    (
        # Wait for uvicorn to be ready
        sleep 15
        echo ">>> [BG] Starting index build (max $MAX_PASSAGES passages)..."

        if [ ! -f "$RAW_PASSAGES" ]; then
            echo "[BG] ERROR: $RAW_PASSAGES not found." >&2
            exit 1
        fi

        INDEX_PATH="$INDEX_PATH" \
        FASTEMBED_CACHE_DIR="$FASTEMBED_CACHE" \
        python -m ingestion.build_index \
            --input "$RAW_PASSAGES" \
            --output "$INDEX_PATH" \
            --strategy parent_child \
            --table chunks \
            --batch-size 64 \
            --max-passages "$MAX_PASSAGES"

        touch "$LOCK_FILE"
        echo ">>> [BG] Index built successfully. App is now fully ready."
    ) &
    disown
else
    echo ">>> Existing index found — skipping build."
fi

# ── Start FastAPI immediately ─────────────────────────────────────────────────
echo ">>> Starting uvicorn on 0.0.0.0:$PORT ..."
exec uvicorn api.index:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    --log-level info
