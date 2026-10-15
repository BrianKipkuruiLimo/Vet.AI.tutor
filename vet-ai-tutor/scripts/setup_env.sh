#!/usr/bin/env bash
# Linux/macOS helper to create a virtual environment and install project requirements.
#
# Usage (from repo root):
#   bash scripts/setup_env.sh
#
# This will create `.venv`, upgrade packaging tools, and install packages from
# `requirements.txt` into the venv Python.

set -e

VENV_PATH=".venv"

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment at $VENV_PATH"
    python3 -m venv "$VENV_PATH"
else
    echo "Virtual environment already exists at $VENV_PATH"
fi

echo "Upgrading pip, setuptools, wheel in venv..."
"$VENV_PATH/bin/python" -m pip install --upgrade pip setuptools wheel

if [ ! -f "requirements.txt" ]; then
    echo "ERROR: requirements.txt not found in repo root. Aborting." >&2
    exit 1
fi

echo "Installing project requirements into venv (this may take several minutes)..."
"$VENV_PATH/bin/python" -m pip install -r requirements.txt

echo ""
echo "Setup complete. To activate the venv run:"
echo "  source .venv/bin/activate"
echo ""
echo "Then you can run:"
echo "  python scripts/smoke_check.py"
echo "  streamlit run vet_ai_tutor_app.py"
