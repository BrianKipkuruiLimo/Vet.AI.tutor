#!/usr/bin/env python3
"""Create the Ed25519 key pair used to sign knowledge patches.

The private key never leaves the developer machine. Commit the public key
(data/patch_signing_public.pem); offline installs use it to verify patches.

Usage:
    python scripts/generate_signing_key.py
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from vet_ai_tutor.config import PATCH_PUBLIC_KEY_PATH, PATCH_SIGNING_KEY_PATH, project_path
from vet_ai_tutor.signing import generate_keypair


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--private-key", default=PATCH_SIGNING_KEY_PATH)
    parser.add_argument("--public-key", default=project_path(PATCH_PUBLIC_KEY_PATH))
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing key (old patches stop verifying)")
    args = parser.parse_args()
    try:
        generate_keypair(args.private_key, args.public_key, overwrite=args.overwrite)
    except FileExistsError as exc:
        parser.exit(1, f"{exc}. Pass --overwrite to replace it.\n")
    print(f"Private key: {args.private_key}  (keep secret, back it up)")
    print(f"Public key:  {args.public_key}  (commit this; it ships with the app)")


if __name__ == "__main__":
    main()
