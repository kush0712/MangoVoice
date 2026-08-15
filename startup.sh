#!/bin/bash
# MangoVoice — Railway startup script
#
# The LanceDB index is now baked directly into the Docker image at build-time.
# We no longer need to download it or use a persistent volume.

set -e

PORT="${PORT:-8000}"

echo "=== MangoVoice Startup ==="
echo "PORT=$PORT"
echo "INDEX_PATH=$INDEX_PATH"

# Start FastAPI
echo ">>> Starting uvicorn on 0.0.0.0:$PORT ..."
exec uvicorn api.index:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    --log-level info
