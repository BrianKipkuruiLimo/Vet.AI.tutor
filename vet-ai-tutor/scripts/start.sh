#!/usr/bin/env bash
# Launch the Vet AI Tutor Streamlit chat app.
#
# Usage (from repo root):
#   bash scripts/start.sh
#
# This will:
#   1. Activate the virtual environment
#   2. Check that Ollama is reachable
#   3. Launch the Streamlit app (auto-opens in your browser)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

VENV_PATH=".venv"

# Activate venv
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
elif [ -f "venv/bin/activate" ]; then
    source "venv/bin/activate"
else
    echo "❌ No virtual environment found. Run: bash scripts/setup_env.sh"
    exit 1
fi

echo "🐄 Vet AI Tutor — Starting up..."
echo ""

# Quick Ollama check
if command -v ollama &>/dev/null; then
    if ollama list &>/dev/null; then
        echo "✅ Ollama is running"
    else
        echo "⚠️  Ollama is installed but may not be running. Starting it..."
        ollama serve &>/dev/null &
        sleep 2
    fi
else
    echo "⚠️  Ollama not found on PATH — make sure it's running separately"
fi

# Check vectorstore exists
if [ -d "data/chroma_db" ] && [ "$(ls -A data/chroma_db 2>/dev/null)" ]; then
    echo "✅ Vectorstore found at data/chroma_db"
elif [ -d "data/vectorstore" ] && [ "$(ls -A data/vectorstore 2>/dev/null)" ]; then
    echo "✅ Vectorstore found at data/vectorstore"
else
    echo "⚠️  No vectorstore found. Build it first:"
    echo "   python scripts/rebuild_vectorstore.py"
    echo ""
fi

echo ""
echo "🚀 Launching Streamlit chat interface..."
echo "   The app will open in your browser automatically."
echo "   Press Ctrl+C to stop."
echo ""

# Launch Streamlit — auto-opens browser
streamlit run vet_ai_tutor_app.py \
    --server.port 8501 \
    --server.headless false \
    --browser.gatherUsageStats false
