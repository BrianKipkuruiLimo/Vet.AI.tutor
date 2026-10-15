"""Normalize vectorstore locations by consolidating any root `vectorstore/`
contents into the canonical `data/vectorstore/` directory.

Run from repo root:
  python scripts/normalize_vectorstore.py
"""
import os
import shutil


def main():
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src = os.path.join(repo, "vectorstore")
    dst = os.path.join(repo, "data", "vectorstore")

    print(f"Source: {src}")
    print(f"Destination: {dst}")

    if not os.path.isdir(src):
        print("No root 'vectorstore' directory found; nothing to do.")
        return 0

    os.makedirs(dst, exist_ok=True)

    moved = []
    skipped = []
    for fname in os.listdir(src):
        s = os.path.join(src, fname)
        d = os.path.join(dst, fname)
        if os.path.exists(d):
            skipped.append(fname)
            print(f"Skipping {fname}: already exists in destination")
            continue
        print(f"Moving {fname} -> {dst}")
        shutil.move(s, d)
        moved.append(fname)

    # If src is empty after moves, remove it
    try:
        if not os.listdir(src):
            print(f"Removing empty source dir {src}")
            os.rmdir(src)
    except Exception as e:
        print(f"Could not remove {src}: {e}")

    print("Summary:")
    print("  Moved:", moved)
    print("  Skipped:", skipped)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
