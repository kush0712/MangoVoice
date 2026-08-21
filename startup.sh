#!/bin/bash
# MangoVoice — Railway startup script
#
# The LanceDB index is now baked directly into the Docker image at build-time.
# We no longer need to download it or use a persistent volume.

set -e

PORT="${PORT:-8000}"

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "=== MangoVoice Startup ==="
echo "PORT=$PORT"
echo "INDEX_PATH=$INDEX_PATH"

# Start FastAPI
echo ">>> Starting uvicorn on 0.0.0.0:$PORT ..."
exec uvicorn api.index:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    --loop uvloop \
    --http httptools \
    --log-level warning
