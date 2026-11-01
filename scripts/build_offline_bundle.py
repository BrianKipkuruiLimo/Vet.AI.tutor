#!/usr/bin/env python3
"""Build a Linux, Python-version-specific offline deployment bundle.

Build on a connected staging machine with the target Python/platform and the
approved Ollama models already installed. The resulting ZIP can be moved by USB.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_ITEMS = ("requirements.txt", "vet_ai_tutor", "static", "scripts")
sys.path.insert(0, str(ROOT))
from vet_ai_tutor.config import (
    VECTOR_BACKEND, VECTOR_PATH, EMBEDDING_MODEL, LLM_MODEL, PATCH_PUBLIC_KEY_PATH, project_path,
)


def build(output: str, model_store: str, ollama_installer: str):
    model_store = Path(model_store).expanduser().resolve()
    installer = Path(ollama_installer).expanduser().resolve()
    index = Path(project_path(VECTOR_PATH))
    if not model_store.is_dir() or not any(model_store.rglob("*")):
        raise ValueError("Provide a populated Ollama models directory")
    if not installer.is_file():
        raise ValueError("Provide the approved offline Ollama installer for the target OS")
    if not index.is_dir() or not any(index.iterdir()):
        raise ValueError(f"Vector index is missing or empty: {index}")
    if VECTOR_BACKEND == "chroma":
        from scripts.create_data_patch import validate_index
        validate_index(index)
    elif not all((index / name).is_file() for name in ("index.faiss", "index.pkl")):
        raise ValueError("FAISS index is incomplete")

    manifest_root = model_store / "manifests" / "registry.ollama.ai" / "library"
    missing_models = []
    for model in (LLM_MODEL, EMBEDDING_MODEL):
        name, _, tag = model.partition(":")
        tag = tag or "latest"
        if not (manifest_root / name / tag).is_file():
            missing_models.append(model)
    if missing_models:
        raise ValueError(f"Ollama model store is missing configured models: {', '.join(missing_models)}")

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vet-ai-offline-") as tmp:
        bundle = Path(tmp) / "vet-ai-tutor"
        bundle.mkdir()
        for item in APP_ITEMS:
            src = ROOT / item
            if not src.exists():
                continue
            dst = bundle / item
            if src.is_dir():
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
            else:
                shutil.copy2(src, dst)
        shutil.copytree(index, bundle / "data" / "knowledge_index")
        # Offline installs verify knowledge patches with the developer's public key.
        public_key = Path(project_path(PATCH_PUBLIC_KEY_PATH))
        if public_key.is_file():
            shutil.copy2(public_key, bundle / "data" / "patch_signing_public.pem")
        else:
            print(f"WARNING: {public_key} not found; this bundle will refuse all knowledge patches. "
                  "Run scripts/generate_signing_key.py first.")
        shutil.copytree(model_store, bundle / "ollama-models")
        shutil.copy2(installer, bundle / f"ollama-installer{installer.suffix}")
        wheelhouse = bundle / "wheelhouse"
        wheelhouse.mkdir()
        subprocess.run([
            sys.executable, "-m", "pip", "download", "--only-binary=:all:",
            "--dest", str(wheelhouse), "-r", str(bundle / "requirements.txt"),
        ], check=True)
        (bundle / "install_offline.sh").write_text('''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --no-index --find-links wheelhouse -r requirements.txt
if ! command -v ollama >/dev/null 2>&1; then
  case "$(find . -maxdepth 1 -name 'ollama-installer*' -print -quit)" in
    *.deb) sudo dpkg -i ./ollama-installer*.deb ;;
    *) echo "Install the included platform installer, then run ./run_offline.sh"; exit 1 ;;
  esac
else
  echo "Dependencies installed. Run ./run_offline.sh"
fi
''', encoding="utf-8")
        (bundle / "run_offline.sh").write_text('''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export VET_VECTOR_PATH=data/knowledge_index
export VET_OFFLINE_ONLY=1
export OLLAMA_MODELS="$(pwd)/ollama-models"
export OLLAMA_HOST=127.0.0.1:11435
export VET_ADMIN_TOKEN="$(cat .admin_token)"
if ! ollama list >/dev/null 2>&1; then
  ollama serve >/dev/null 2>&1 &
  OLLAMA_PID=$!
  trap 'kill "$OLLAMA_PID" 2>/dev/null || true' EXIT
  for attempt in $(seq 1 30); do
    ollama list >/dev/null 2>&1 && break
    sleep 1
  done
fi
.venv/bin/python -m vet_ai_tutor.api
''', encoding="utf-8")
        token = subprocess.check_output([sys.executable, "-c", "import secrets; print(secrets.token_urlsafe(32))"], text=True).strip()
        (bundle / ".admin_token").write_text(token + "\n", encoding="utf-8")
        (bundle / "OFFLINE_DEPLOYMENT.txt").write_text(
            "Vet AI Tutor offline deployment\n"
            "1. Run ./install_offline.sh on the target Linux machine.\n"
            "2. Install the included Ollama installer if Ollama is absent.\n"
            "3. Run ./run_offline.sh and open http://localhost:8000.\n"
            "4. Keep .admin_token private; it authorizes knowledge updates.\n"
            "The Python wheelhouse and model store are platform/Python specific.\n",
            encoding="utf-8",
        )
        for filename in ("install_offline.sh", "run_offline.sh"):
            (bundle / filename).chmod(0o755)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
            for file in bundle.rglob("*"):
                if file.is_file():
                    zf.write(file, file.relative_to(bundle.parent))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="dist/vet-ai-tutor-offline.zip")
    parser.add_argument("--model-store", required=True, help="Ollama models directory")
    parser.add_argument("--ollama-installer", required=True, help="Offline Ollama installer for target OS")
    args = parser.parse_args()
    try:
        result = build(args.output, args.model_store, args.ollama_installer)
        print(f"Offline bundle created: {result} ({result.stat().st_size / 1024**3:.2f} GiB)")
    except Exception as exc:
        parser.exit(1, f"Offline bundle failed: {exc}\n")


if __name__ == "__main__":
    main()
