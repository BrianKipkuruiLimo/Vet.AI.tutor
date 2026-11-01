"""Metadata review for data/raw_notes: drafts -> reviewed catalog.

Two files are involved:
- data/source_metadata.draft.json: machine-drafted entries awaiting review
- data/source_metadata.json: the reviewed catalog the rebuild trusts

Approval runs the same provenance validation the rebuild uses, so an entry
only reaches the catalog if the rebuild will accept it. Every catalog write
is atomic and keeps a timestamped backup.
"""
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from vet_ai_tutor.config import SKIP_EXTENSIONS, SUPPORTED_EXTENSIONS
from vet_ai_tutor.ingestion.provenance import ProvenanceError, load_whitelist, validate_provenance

EDITABLE_FIELDS = ("source_title", "author", "journal", "publication_year", "source_url", "source_tier")
_REVIEW_MARKERS = ("_needs_review", "_detected")
# Course codes have three digits ("VMED 344", "ANSC 460", "VMED-230"); "BVMS3" or
# "BVM-1" (year of study) is used only when no course code is present.
_COURSE_RE = re.compile(r"(?<![A-Z])([A-Z]{3,5})[\s_-]?(\d{3})(?!\d)")
_YEAR_RE = re.compile(r"(?<![A-Z])(BVMS?)[\s_-]?(\d)(?!\d)")
_COURSE_STOP = {"CAMSCANNER", "LECTURE", "GROUP", "NOTES", "PAGE", "WEEK", "TOPIC"}


def course_code(filename: str) -> str:
    """Best-effort course code from a filename ('' if none is found)."""
    name = filename.upper()
    for prefix, number in _COURSE_RE.findall(name):
        if prefix not in _COURSE_STOP:
            return f"{prefix} {number}"
    year = _YEAR_RE.search(name)
    return f"{year.group(1)} {year.group(2)}" if year else ""


def read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def backup(path: str, backups_dir: str) -> Optional[str]:
    """Copy `path` to backups_dir/catalog/<name>.<timestamp>.json before changing it."""
    if not os.path.exists(path):
        return None
    target_dir = os.path.join(backups_dir, "catalog")
    os.makedirs(target_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    target = os.path.join(target_dir, f"{os.path.splitext(os.path.basename(path))[0]}.{stamp}.json")
    shutil.copy2(path, target)
    return target


def raw_files(raw_dir: str) -> Dict[str, str]:
    """Supported files under raw_dir as {catalog key: absolute path}.

    The key is the filename when it is unique, otherwise the relative path,
    matching scripts/draft_source_metadata.py and the rebuild's lookup.
    """
    found = []
    for root, _, names in os.walk(raw_dir):
        for name in sorted(names):
            ext = os.path.splitext(name.lower())[1]
            if ext in SUPPORTED_EXTENSIONS and ext not in SKIP_EXTENSIONS:
                found.append(os.path.join(root, name))
    counts: Dict[str, int] = {}
    for path in found:
        counts[os.path.basename(path)] = counts.get(os.path.basename(path), 0) + 1
    return {(os.path.basename(p) if counts[os.path.basename(p)] == 1
             else os.path.relpath(p, raw_dir).replace(os.sep, "/")): p for p in found}


def _problem(entry: dict, key: str, whitelist: dict) -> str:
    candidate = {k: v for k, v in entry.items() if k not in _REVIEW_MARKERS}
    try:
        validate_provenance(candidate, key, whitelist)
        return ""
    except ProvenanceError as exc:
        return str(exc).split(": ", 1)[-1]


def review_rows(raw_dir: str, catalog_path: str, drafts_path: str, whitelist_path: str) -> List[dict]:
    """One row per source file with its status and what blocks approval."""
    catalog, drafts = read_json(catalog_path), read_json(drafts_path)
    whitelist = load_whitelist(whitelist_path)
    files = raw_files(raw_dir)
    rows = []
    for key in sorted(set(files) | set(catalog) | set(drafts), key=str.lower):
        if key in catalog:
            entry, status = catalog[key], "approved"
        elif key in drafts:
            entry, status = drafts[key], "draft"
        else:
            entry, status = {}, "no draft"
        problem = _problem(entry, key, whitelist) if entry else "no metadata yet (run the drafter)"
        if status == "approved" and problem:
            status = "invalid"
        rows.append({
            "key": key,
            "status": status,
            "course": course_code(key),
            **{f: str(entry.get(f, "")) for f in EDITABLE_FIELDS},
            "problem": "" if status == "approved" else problem,
            "file_exists": key in files,
            "path": files.get(key, ""),
        })
    return rows


def apply_edits(drafts_path: str, edits: Dict[str, Dict[str, str]]) -> int:
    """Save reviewer edits ({key: {field: value}}) to draft entries. Returns entries changed."""
    drafts = read_json(drafts_path)
    changed = 0
    for key, fields in edits.items():
        if key not in drafts:
            continue
        for field, value in fields.items():
            if field in EDITABLE_FIELDS and drafts[key].get(field, "") != value:
                drafts[key][field] = str(value).strip()
                changed += 1
    if changed:
        write_json(drafts_path, drafts)
    return changed


def bulk_set(drafts_path: str, keys: Iterable[str], field: str, value: str) -> int:
    if field not in EDITABLE_FIELDS:
        raise ValueError(f"{field} cannot be edited")
    return apply_edits(drafts_path, {key: {field: value} for key in keys})


def approve(keys: Iterable[str], drafts_path: str, catalog_path: str, whitelist_path: str,
            backups_dir: str) -> Tuple[List[str], Dict[str, str]]:
    """Move valid drafts into the catalog. Returns (approved keys, {key: reason} for rejected)."""
    drafts, catalog = read_json(drafts_path), read_json(catalog_path)
    whitelist = load_whitelist(whitelist_path)
    approved, rejected = [], {}
    for key in keys:
        entry = drafts.get(key)
        if entry is None:
            rejected[key] = "not in the draft file"
            continue
        clean = {k: v for k, v in entry.items() if k not in _REVIEW_MARKERS}
        try:
            validate_provenance(clean, key, whitelist)
        except ProvenanceError as exc:
            rejected[key] = str(exc).split(": ", 1)[-1]
            continue
        clean["reviewed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        catalog[key] = clean
        del drafts[key]
        approved.append(key)
    if approved:
        backup(catalog_path, backups_dir)
        write_json(catalog_path, catalog)
        write_json(drafts_path, drafts)
    return approved, rejected


def unapprove(keys: Iterable[str], drafts_path: str, catalog_path: str, backups_dir: str) -> List[str]:
    """Move catalog entries back to drafts (for correction). Returns keys moved."""
    drafts, catalog = read_json(drafts_path), read_json(catalog_path)
    moved = []
    for key in keys:
        if key in catalog:
            drafts[key] = {**catalog.pop(key), "_needs_review": True}
            drafts[key].pop("reviewed_at", None)
            moved.append(key)
    if moved:
        backup(catalog_path, backups_dir)
        write_json(catalog_path, catalog)
        write_json(drafts_path, drafts)
    return moved


def add_upload(name: str, content: bytes, raw_dir: str, drafts_path: str, course: str = "",
               default_author: str = "") -> str:
    """Save an uploaded document under raw_notes/uploaded/ and draft its metadata.

    Returns the catalog key. Identical content already uploaded is not duplicated.
    """
    from pathlib import Path

    from scripts.draft_source_metadata import draft_entry

    safe = os.path.basename(name).replace(os.sep, "_").strip() or "upload"
    if os.path.splitext(safe.lower())[1] not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {safe}")
    upload_dir = os.path.join(raw_dir, "uploaded")
    os.makedirs(upload_dir, exist_ok=True)
    digest = hashlib.sha256(content).hexdigest()
    target = os.path.join(upload_dir, safe)
    if os.path.exists(target):
        with open(target, "rb") as f:
            same = hashlib.sha256(f.read()).hexdigest() == digest
        if not same:
            stem, ext = os.path.splitext(safe)
            target = os.path.join(upload_dir, f"{stem}-{digest[:8]}{ext}")
    if not os.path.exists(target):
        with open(target, "wb") as f:
            f.write(content)
    rel = os.path.relpath(target, raw_dir).replace(os.sep, "/")
    key = next(k for k, p in raw_files(raw_dir).items() if os.path.samefile(p, target))
    drafts = read_json(drafts_path)
    if key not in drafts:
        drafts[key] = draft_entry(Path(target), rel, course, default_author)
        write_json(drafts_path, drafts)
    return key
