#!/usr/bin/env bash
# Linux/macOS helper to create the virtual environment and install dependencies.
#
# Usage (from the vet-ai-tutor directory):
#   bash scripts/setup_env.sh
#
# With uv installed (recommended), this installs the exact versions from
# uv.lock, including the dev tools. Without uv it falls back to pip and the
# hash-pinned requirements.txt (runtime dependencies only).

set -e

if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: run this from the vet-ai-tutor directory (pyproject.toml not found)." >&2
    exit 1
fi

if command -v uv >/dev/null 2>&1; then
    echo "Installing locked dependencies with uv (this may take several minutes)..."
    uv sync --locked
else
    VENV_PATH=".venv"
    if [ ! -d "$VENV_PATH" ]; then
        echo "Creating virtual environment at $VENV_PATH"
        python3 -m venv "$VENV_PATH"
    fi
    echo "uv not found; installing runtime dependencies with pip."
    echo "(Install uv for the dev tools: https://docs.astral.sh/uv/)"
    "$VENV_PATH/bin/python" -m pip install --upgrade pip
    "$VENV_PATH/bin/python" -m pip install --require-hashes -r requirements.txt
fi

echo ""
echo "Setup complete. To activate the venv run:"
echo "  source .venv/bin/activate"
echo ""
echo "Then pull the models and start the app:"
echo "  ollama pull qwen3:1.7b"
echo "  ollama pull nomic-embed-text"
echo "  python -m vet_ai_tutor.api        # http://localhost:8000"
