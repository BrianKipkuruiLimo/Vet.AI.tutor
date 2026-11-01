"""File locations used by the developer dashboard, resolved from settings.

Pages call `get_paths()` at run time, so tests can point the whole dashboard
at a temporary directory by replacing this function.
"""
import os
from dataclasses import dataclass

from vet_ai_tutor.config import (
    BACKUPS_DIR, PATCH_PUBLIC_KEY_PATH, PATCH_SIGNING_KEY_PATH, PATCHES_DIR, RAW_DIR,
    SOURCE_CATALOG_PATH, STAGING_DIR, VECTOR_PATH, WHITELIST_PATH, project_path,
)


@dataclass(frozen=True)
class DevPaths:
    root: str
    raw_dir: str
    catalog: str
    drafts: str
    whitelist: str
    staging: str
    index: str
    eval_dir: str
    qa_pairs: str
    quiz_pending: str
    jobs: str
    rebuild_summary: str
    patches: str
    public_key: str
    signing_key: str
    backups: str


def get_paths() -> DevPaths:
    root = project_path(".")
    return DevPaths(
        root=os.path.abspath(root),
        raw_dir=project_path(RAW_DIR),
        catalog=project_path(SOURCE_CATALOG_PATH),
        drafts=project_path("data/source_metadata.draft.json"),
        whitelist=project_path(WHITELIST_PATH),
        staging=project_path(STAGING_DIR),
        index=project_path(VECTOR_PATH),
        eval_dir=project_path("data/eval"),
        qa_pairs=project_path("data/eval/qa_pairs.json"),
        quiz_pending=project_path("data/quiz/pending.json"),
        jobs=project_path("data/jobs"),
        rebuild_summary=project_path("archive/rebuild_summary.json"),
        patches=project_path(PATCHES_DIR),
        public_key=project_path(PATCH_PUBLIC_KEY_PATH),
        signing_key=PATCH_SIGNING_KEY_PATH,
        backups=project_path(BACKUPS_DIR),
    )
