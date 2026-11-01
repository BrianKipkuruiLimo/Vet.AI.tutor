"""Source whitelist and provenance validation. Everything here fails closed.

Used by the full rebuild (catalog entries for files in data/raw_notes/), the
staging pipeline (harvested and ingested JSON records) and the harvester.
"""
import json
import os
import re
from typing import Dict, Optional
from urllib.parse import urlparse

from vet_ai_tutor.config import WHITELIST_PATH, project_path

REQUIRED_FIELDS = ("source_title", "author", "journal", "publication_year", "source_url", "source_tier")
APPROVED_TIERS = {"tier_1", "tier_2"}
_YEAR_RE = re.compile(r"(?:18|19|20|21)\d{2}")


class ProvenanceError(ValueError):
    """A source is missing reviewed, complete, whitelisted provenance."""


def load_whitelist(path: Optional[str] = None) -> Dict[str, str]:
    """Load data/whitelist.json as {domain: tier}. Missing or empty lists are errors."""
    path = project_path(path or WHITELIST_PATH)
    if not os.path.exists(path):
        raise RuntimeError(f"Required source whitelist is missing: {path}")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    domain_tier = {}
    for tier_name in ("tier_1", "tier_2", "tier_3"):
        for domain in raw.get(tier_name, {}).get("domains", []):
            domain_tier[domain.lower().strip().strip(".")] = tier_name
    if not domain_tier:
        raise RuntimeError(f"Source whitelist contains no approved domains: {path}")
    return domain_tier


def check_domain(url: str, whitelist: Dict[str, str]) -> Optional[str]:
    """Return the tier of an HTTPS URL's domain (or a parent domain), else None."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return None
    parts = parsed.hostname.lower().split(".")
    for i in range(len(parts)):
        candidate = ".".join(parts[i:])
        if candidate in whitelist:
            return whitelist[candidate]
    return None


def validate_provenance(record: Optional[dict], name: str, whitelist: Dict[str, str]) -> dict:
    """Check one source's provenance; return the fields to store on its chunks."""
    if not isinstance(record, dict):
        raise ProvenanceError(f"{name}: no reviewed provenance entry")
    if record.get("_needs_review"):
        raise ProvenanceError(f"{name}: provenance entry is still a draft (_needs_review)")
    missing = [key for key in REQUIRED_FIELDS if not str(record.get(key, "")).strip()]
    if missing:
        raise ProvenanceError(f"{name}: incomplete provenance (missing {', '.join(missing)})")
    if not _YEAR_RE.fullmatch(str(record["publication_year"])):
        raise ProvenanceError(f"{name}: invalid publication_year {record['publication_year']!r}")
    tier = record.get("source_tier")
    if tier not in APPROVED_TIERS:
        raise ProvenanceError(f"{name}: only reviewed tier_1 and tier_2 sources are allowed")
    url = str(record["source_url"])
    scheme = urlparse(url).scheme
    if scheme not in {"https", "file"}:
        raise ProvenanceError(f"{name}: source_url must use https:// or file://")
    if scheme == "https" and check_domain(url, whitelist) != tier:
        raise ProvenanceError(f"{name}: source URL is not whitelisted at the declared tier")
    stored = {key: str(record[key]) for key in REQUIRED_FIELDS}
    stored["source_domain"] = urlparse(url).hostname or "local_document"
    return stored


def load_catalog(path: str) -> dict:
    """Load data/source_metadata.json (keyed by relative path or unique filename)."""
    path = project_path(path)
    if not os.path.isfile(path):
        raise ProvenanceError(
            f"Required provenance catalog missing: {path}. Draft it with "
            "scripts/draft_source_metadata.py and review every entry."
        )
    with open(path, encoding="utf-8") as f:
        catalog = json.load(f)
    if not isinstance(catalog, dict):
        raise ProvenanceError("Source metadata catalog must be a JSON object keyed by source filename.")
    return catalog


def catalog_entry(catalog: dict, raw_dir: str, path: str) -> Optional[dict]:
    rel = os.path.relpath(path, raw_dir).replace(os.sep, "/")
    return catalog.get(rel) or catalog.get(os.path.basename(path))
