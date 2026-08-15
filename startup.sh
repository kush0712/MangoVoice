#!/bin/bash
# MangoVoice — Railway startup script
# Mounts a Railway persistent volume at /data (configured in Railway dashboard).
# On first boot: builds the LanceDB index from committed JSONL.
# On subsequent boots: reuses the persisted index (fast).

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

# Create data dirs on the volume
mkdir -p "$INDEX_PATH"
mkdir -p "$FASTEMBED_CACHE"

# ── Build index if not already done ──────────────────────────────────────────
if [ ! -f "$LOCK_FILE" ]; then
    echo ">>> Index not found — building from $RAW_PASSAGES ..."

    if [ ! -f "$RAW_PASSAGES" ]; then
        echo "ERROR: $RAW_PASSAGES not found. Check git commit." >&2
        exit 1
    fi

    # Count passages (use all of them)
    PASSAGE_COUNT=$(wc -l < "$RAW_PASSAGES")
    echo ">>> Total passages: $PASSAGE_COUNT"

    INDEX_PATH="$INDEX_PATH" \
    FASTEMBED_CACHE_DIR="$FASTEMBED_CACHE" \
    python -m ingestion.build_index \
        --input "$RAW_PASSAGES" \
        --output "$INDEX_PATH" \
        --strategy parent_child \
        --table chunks \
        --batch-size 512 \
        --max-passages 0

    touch "$LOCK_FILE"
    echo ">>> Index built and saved to $INDEX_PATH"
else
    echo ">>> Existing index found at $INDEX_PATH — skipping build."
fi

# ── Export paths for the app ──────────────────────────────────────────────────
export INDEX_PATH="$INDEX_PATH"
export FASTEMBED_CACHE_DIR="$FASTEMBED_CACHE"

# ── Start FastAPI ─────────────────────────────────────────────────────────────
echo ">>> Starting uvicorn on port $PORT ..."
exec uvicorn api.index:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    --log-level info
