"""Read-only summaries for the developer dashboard (cheap file reads only)."""
import base64
import glob
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from typing import List, Optional

from vet_ai_tutor.embeddings import read_index_info


def _load(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def dir_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return round(total / 1024 / 1024, 1)


def index_stats(index_dir: str, cache_dir: Optional[str] = None) -> dict:
    """Chunk count, size and embedding profile of the active index."""
    stats = {"exists": os.path.isdir(index_dir) and bool(os.listdir(index_dir)), "chunks": 0,
             "size_mb": dir_size_mb(index_dir) if os.path.isdir(index_dir) else 0.0,
             "profile": read_index_info(index_dir), "error": ""}
    if stats["exists"] and os.path.isfile(os.path.join(index_dir, "chroma.sqlite3")):
        try:
            import chromadb
            client = chromadb.PersistentClient(path=index_dir)
            stats["chunks"] = sum(c.count() for c in client.list_collections())
        except Exception as exc:  # a locked or damaged index should not break the dashboard
            stats["error"] = str(exc)
    if cache_dir:
        stats["embedding_cache_mb"] = dir_size_mb(os.path.join(cache_dir, "embeddings"))
    return stats


def rebuild_report(summary_path: str) -> Optional[dict]:
    report = _load(summary_path)
    if isinstance(report, dict):
        report["written_at"] = datetime.fromtimestamp(os.path.getmtime(summary_path), timezone.utc)
    return report if isinstance(report, dict) else None


def _timestamp(path: str) -> datetime:
    digits = "".join(c for c in os.path.basename(path).rsplit("_", 1)[-1] if c.isdigit())
    if len(digits) >= 9:
        return datetime.fromtimestamp(int(digits[:10]), timezone.utc)
    return datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)


def retrieval_history(eval_dir: str) -> List[dict]:
    """One row per retrieval evaluation run (results_*.json), oldest first."""
    rows = []
    for path in glob.glob(os.path.join(eval_dir, "results_*.json")):
        data = _load(path)
        if not isinstance(data, dict) or "summary" not in data:
            continue  # results from the old evaluator have no summary block
        summary, cfg = data["summary"], data.get("config", {})
        rows.append({"time": _timestamp(path), "mode": cfg.get("mode", "hybrid"), "k": cfg.get("k"),
                     "questions": summary.get("total"), "recall_at_k": summary.get("recall_at_k"),
                     "mrr": summary.get("mrr"), "chunks": cfg.get("chunks"),
                     "embedding_model": cfg.get("embedding_model", "")})
    return sorted(rows, key=lambda r: r["time"])


def benchmark_history(eval_dir: str) -> List[dict]:
    """One row per benchmark run (benchmark_*.json), oldest first."""
    rows = []
    for path in glob.glob(os.path.join(eval_dir, "benchmark_*.json")):
        data = _load(path)
        if not isinstance(data, dict):
            continue
        summary, checks = data.get("summary", {}), data.get("answer_checks", {})
        ragas = (data.get("ragas") or {}).get("summary", {})
        rows.append({
            "time": _timestamp(path), "questions": summary.get("total_questions"),
            "llm_model": (data.get("system_specs") or {}).get("llm_model", ""),
            "avg_seconds": summary.get("avg_time_per_question"),
            "grounding": checks.get("mean_grounding"), "key_term_recall": checks.get("mean_key_term_recall"),
            "citation_rate": checks.get("citation_rate"), "not_found_rate": checks.get("not_found_rate"),
            "faithfulness": ragas.get("faithfulness"), "context_recall": ragas.get("context_recall"),
        })
    return sorted(rows, key=lambda r: r["time"])


def qa_stats(qa_path: str) -> dict:
    items = _load(qa_path) or []
    return {"questions": len(items),
            "with_source": sum(1 for i in items if i.get("source")),
            "retrieval_missed": sum(1 for i in items if i.get("retrieval_missed"))}


def quiz_stats(pending_path: str, bank_path: str) -> dict:
    pending = _load(pending_path) or []
    bank = (_load(bank_path) or {}).get("questions", [])
    by_status: dict = {}
    for q in pending:
        by_status[q.get("status", "?")] = by_status.get(q.get("status", "?"), 0) + 1
    return {"drafts": by_status, "published": sum(1 for q in bank if q.get("status") == "approved")}


def key_fingerprint(public_key_path: str) -> Optional[str]:
    """SHA-256 fingerprint of a PEM public key (first 16 hex chars), or None."""
    try:
        with open(public_key_path, "rb") as f:
            pem = f.read()
    except OSError:
        return None
    body = b"".join(line for line in pem.splitlines() if line and not line.startswith(b"-----"))
    try:
        der = base64.b64decode(body)
    except ValueError:
        return None
    digest = hashlib.sha256(der).hexdigest()
    return ":".join(digest[i:i + 4] for i in range(0, 16, 4))


def release_status(paths) -> dict:
    patches = []
    for path in sorted(glob.glob(os.path.join(paths.patches, "*.zip")), key=os.path.getmtime, reverse=True):
        patches.append({"file": os.path.basename(path), "size_mb": round(os.path.getsize(path) / 1024 / 1024, 1),
                        "created": datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)})
    return {
        "private_key": os.path.isfile(paths.signing_key),
        "public_key": os.path.isfile(paths.public_key),
        "fingerprint": key_fingerprint(paths.public_key),
        "patches": patches,
    }


def recent_log(path: str, limit: int = 20) -> List[dict]:
    entries = _load(path)
    return list(reversed(entries[-limit:])) if isinstance(entries, list) else []


def tool_status() -> dict:
    """Whether the optional system tools for OCR and legacy Office files are present."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    components = None
    if soffice and shutil.which("dpkg-query"):
        result = subprocess.run(["dpkg-query", "-W", "-f=${Status}\n", "libreoffice-writer-nogui",
                                 "libreoffice-writer"], capture_output=True, text=True)
        components = "install ok installed" in result.stdout
    return {"tesseract": bool(shutil.which("tesseract")), "libreoffice": bool(soffice),
            "libreoffice_writer": components}
