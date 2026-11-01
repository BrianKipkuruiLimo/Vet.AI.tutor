#!/usr/bin/env bash
# Launch the Vet AI Tutor web app (FastAPI + chat page).
#
# Usage (from the vet-ai-tutor directory):
#   bash scripts/start.sh
#
# This will:
#   1. Activate the virtual environment
#   2. Make sure Ollama is running
#   3. Warn if no knowledge index has been built
#   4. Start the API and open the chat page in your browser when it is ready

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
else
    echo "❌ No virtual environment found. Run: bash scripts/setup_env.sh"
    exit 1
fi

echo "🐄 Vet AI Tutor — Starting up..."
echo ""

if command -v ollama &>/dev/null; then
    if ollama list &>/dev/null; then
        echo "✅ Ollama is running"
    else
        echo "⚠️  Ollama is installed but not running. Starting it..."
        ollama serve &>/dev/null &
        sleep 2
    fi
else
    echo "⚠️  Ollama not found on PATH — make sure it's running separately"
fi

INDEX_PATH="${VET_VECTOR_PATH:-data/chroma_db}"
if [ -d "$INDEX_PATH" ] && [ "$(ls -A "$INDEX_PATH" 2>/dev/null)" ]; then
    echo "✅ Knowledge index found at $INDEX_PATH"
else
    echo "⚠️  No knowledge index found. Build it first:"
    echo "   python scripts/rebuild_vectorstore.py"
    echo ""
fi

# Host and port come from the app's settings, so VET_API_PORT works from the
# environment or from .env.
read -r HOST PORT < <(python -c "from vet_ai_tutor.config import API_HOST, API_PORT; print(API_HOST, API_PORT)")
URL="http://localhost:$PORT"

# Fail fast, before loading any models, if another program already uses the port.
if ! python - "$HOST" "$PORT" <<'PY'
import socket, sys
with socket.socket() as s:
    try:
        s.bind((sys.argv[1], int(sys.argv[2])))
    except OSError:
        sys.exit(1)
PY
then
    echo "❌ Port $PORT is already in use by another program."
    echo "   Use another port for this run:   VET_API_PORT=8001 bash scripts/start.sh"
    echo "   Or permanently, in .env:         echo VET_API_PORT=8001 >> .env"
    exit 1
fi

# Open the browser once the API answers /health (model loading takes a few seconds).
(
    for _ in $(seq 1 120); do
        if curl -fs "$URL/health" >/dev/null 2>&1; then
            echo "🌐 Opening $URL"
            if command -v xdg-open &>/dev/null; then
                xdg-open "$URL" >/dev/null 2>&1 || true
            elif command -v open &>/dev/null; then
                open "$URL" || true
            fi
            exit 0
        fi
        sleep 1
    done
    echo "⚠️  The API did not answer within 2 minutes; check the log above."
) &

echo ""
echo "🚀 Starting the API on $URL (Ctrl+C to stop)..."
echo ""
exec python -m vet_ai_tutor.api
