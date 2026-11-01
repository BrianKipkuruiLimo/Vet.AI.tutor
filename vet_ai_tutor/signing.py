"""Ed25519 signing for knowledge patches.

The patch manifest lists a SHA-256 checksum for every index file, so signing
the manifest bytes authenticates the whole patch. Checksums alone only detect
corruption: anyone able to build a ZIP can also compute matching checksums.

The private key stays on the developer machine (default:
~/.config/vet-ai-tutor/patch_signing_key.pem). The public key ships with the
app (data/patch_signing_public.pem) and is safe to commit.
"""
import base64
import os
from pathlib import Path

SIGNATURE_FILE = "manifest.sig"


def generate_keypair(private_path: str | Path, public_path: str | Path, overwrite: bool = False) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_path, public_path = Path(private_path), Path(public_path)
    if private_path.exists() and not overwrite:
        raise FileExistsError(f"Signing key already exists: {private_path}")
    key = Ed25519PrivateKey.generate()
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    private_bytes = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption(),
    )
    # Create the private key readable by the owner only.
    fd = os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(private_bytes)
    public_path.write_bytes(key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo,
    ))


def sign(data: bytes, private_key_path: str | Path) -> bytes:
    """Return a base64-encoded Ed25519 signature of `data`."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = serialization.load_pem_private_key(Path(private_key_path).read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("Patch signing key must be an Ed25519 private key")
    return base64.b64encode(key.sign(data))


def verify(data: bytes, signature_b64: bytes, public_key_path: str | Path) -> None:
    """Raise ValueError unless `signature_b64` is a valid signature of `data`."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    key = serialization.load_pem_public_key(Path(public_key_path).read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("Patch public key must be an Ed25519 public key")
    try:
        key.verify(base64.b64decode(signature_b64, validate=True), data)
    except (InvalidSignature, ValueError) as exc:
        raise ValueError("Patch signature is invalid") from exc
