"""Run long developer tasks as detached background processes.

A rebuild takes hours and OCR takes minutes, so they cannot run inside a
Streamlit page (closing the tab would kill them). Each job runs one of the
existing scripts through a small runner process that records the exit code
in data/jobs/<id>.json and the output in data/jobs/<id>.log, even if the
dashboard is closed. Jobs that write the index are mutually exclusive.

    python -m vet_ai_tutor.devtools.jobs _run <job.json>   (internal: the runner)
"""
import json
import os
import signal
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from vet_ai_tutor.devtools import paths as _paths

# kind -> (script relative to the project root, human label)
JOB_KINDS: Dict[str, tuple] = {
    "draft_metadata": ("scripts/draft_source_metadata.py", "Draft metadata for new files"),
    "dry_run": ("scripts/rebuild_vectorstore.py", "Rebuild dry run"),
    "rebuild": ("scripts/rebuild_vectorstore.py", "Rebuild index"),
    "ocr": ("scripts/ocr_scanned_pdfs.py", "OCR scanned PDFs"),
    "harvest": ("scripts/harvester.py", "Harvest web page"),
    "ingest_staged": ("scripts/ingest_staged.py", "Ingest approved staged records"),
    "evaluate_retrieval": ("scripts/evaluate_retrieval.py", "Retrieval evaluation"),
    "benchmark": ("scripts/benchmark.py", "Answer benchmark"),
    "generate_quiz": ("scripts/generate_quiz.py", "Generate quiz drafts"),
    "create_patch": ("scripts/create_data_patch.py", "Create knowledge patch"),
}

# Kinds in the same group must not run at the same time (they write the index).
_GROUPS = {"rebuild": "index", "ingest_staged": "index", "create_patch": "index"}
_DEFAULT_ARGS = {"dry_run": ["--dry-run"]}


class JobConflict(RuntimeError):
    """A job of this kind (or one that writes the same data) is already running."""


@dataclass
class Job:
    id: str
    kind: str
    label: str
    argv: List[str]
    status: str = "starting"          # starting | running | succeeded | failed | cancelled
    pid: Optional[int] = None
    started: str = ""
    finished: str = ""
    returncode: Optional[int] = None
    log_path: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.status in ("starting", "running")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _status_path(job_id: str, jobs_dir: Optional[str] = None) -> str:
    return os.path.join(jobs_dir or _paths.get_paths().jobs, f"{job_id}.json")


def _write(job: Job, jobs_dir: Optional[str] = None) -> None:
    path = _status_path(job.id, jobs_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(asdict(job), f, indent=2)
    os.replace(tmp, path)


def _read(path: str) -> Optional[Job]:
    try:
        with open(path, encoding="utf-8") as f:
            return Job(**json.load(f))
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A finished child of this process stays a zombie until reaped.
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def list_jobs(kind: Optional[str] = None, limit: int = 20, jobs_dir: Optional[str] = None) -> List[Job]:
    """Newest first. A job whose runner died without reporting is marked failed."""
    jobs_dir = jobs_dir or _paths.get_paths().jobs
    if not os.path.isdir(jobs_dir):
        return []
    jobs = []
    for name in os.listdir(jobs_dir):
        if not name.endswith(".json"):
            continue
        job = _read(os.path.join(jobs_dir, name))
        if job is None or (kind and job.kind != kind):
            continue
        if job.active and not _alive(job.pid) and job.status == "running":
            job = _read(os.path.join(jobs_dir, name)) or job  # the runner may have just finished
            if job.active:
                job.status, job.finished = "failed", _now()
                job.meta["error"] = "runner stopped without reporting (machine restarted or killed)"
                _write(job, jobs_dir)
        jobs.append(job)
    jobs.sort(key=lambda j: j.started, reverse=True)
    return jobs[:limit]


def get_job(job_id: str, jobs_dir: Optional[str] = None) -> Optional[Job]:
    return next((j for j in list_jobs(limit=10_000, jobs_dir=jobs_dir) if j.id == job_id), None)


def active_conflict(kind: str, jobs_dir: Optional[str] = None) -> Optional[Job]:
    group = _GROUPS.get(kind)
    for job in list_jobs(limit=10_000, jobs_dir=jobs_dir):
        if job.active and (job.kind == kind or (group and _GROUPS.get(job.kind) == group)):
            return job
    return None


def start_job(kind: str, args: Optional[List[str]] = None, jobs_dir: Optional[str] = None,
              meta: Optional[dict] = None, command: Optional[List[str]] = None) -> Job:
    """Start a background job. `command` overrides the script (used by tests)."""
    if kind not in JOB_KINDS:
        raise ValueError(f"Unknown job kind: {kind}")
    paths = _paths.get_paths()
    jobs_dir = jobs_dir or paths.jobs
    os.makedirs(jobs_dir, exist_ok=True)
    conflict = active_conflict(kind, jobs_dir)
    if conflict:
        raise JobConflict(f"{conflict.label} is already running (started {conflict.started}).")
    script, label = JOB_KINDS[kind]
    argv = command or [sys.executable, os.path.join(paths.root, script),
                       *_DEFAULT_ARGS.get(kind, []), *(args or [])]
    job_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    job = Job(id=job_id, kind=kind, label=label, argv=argv, started=_now(),
              log_path=os.path.join(jobs_dir, f"{job_id}.log"), meta=meta or {})
    _write(job, jobs_dir)
    runner = subprocess.Popen(
        [sys.executable, "-m", "vet_ai_tutor.devtools.jobs", "_run", _status_path(job_id, jobs_dir)],
        cwd=paths.root, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,  # survives the dashboard; cancel signals the whole group
        env={**os.environ, "PYTHONPATH": paths.root},
    )
    job.pid, job.status = runner.pid, "running"
    _write(job, jobs_dir)
    return job


def cancel_job(job_id: str, jobs_dir: Optional[str] = None) -> bool:
    job = get_job(job_id, jobs_dir)
    if job is None or not job.active or not job.pid:
        return False
    try:
        os.killpg(job.pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    return True


def read_log(job: Job, tail_lines: int = 200) -> str:
    try:
        with open(job.log_path, encoding="utf-8", errors="replace") as f:
            lines = f.read().replace("\r", "\n").splitlines()
    except OSError:
        return ""
    return "\n".join(line for line in lines[-tail_lines:])


def last_line(job: Job) -> str:
    """The most recent non-empty output line (scripts print progress there)."""
    lines = [line for line in read_log(job, 20).splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _run(status_file: str) -> int:
    """Runner process: execute the job command, then record how it ended."""
    job = _read(status_file)
    if job is None:
        return 2
    jobs_dir = os.path.dirname(status_file)
    with open(job.log_path, "ab") as log:
        child = subprocess.Popen(job.argv, cwd=_paths.get_paths().root, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        cancelled = []

        def _terminate(signum, frame):
            cancelled.append(True)
            child.terminate()

        signal.signal(signal.SIGTERM, _terminate)
        while True:
            try:
                code = child.wait()
                break
            except InterruptedError:  # pragma: no cover - wait() restarts on signals in py3.5+
                continue
    job = _read(status_file) or job
    job.returncode = code
    job.finished = _now()
    job.status = "cancelled" if cancelled else ("succeeded" if code == 0 else "failed")
    _write(job, jobs_dir)
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "_run":
        sys.exit(_run(sys.argv[2]))
    print(__doc__)
