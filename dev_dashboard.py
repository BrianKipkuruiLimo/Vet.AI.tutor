"""Developer dashboard for the Vet AI Tutor knowledge pipeline.

DEVELOPER-ONLY: it edits the source catalog, rebuilds the index and signs
releases. It is bound to 127.0.0.1 (see .streamlit/config.toml) and is not
part of the student app.

    streamlit run dev_dashboard.py      # http://localhost:8502
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vet_ai_tutor.devtools import catalog, jobs, status  # noqa: E402
from vet_ai_tutor.devtools.paths import get_paths  # noqa: E402

st.set_page_config(page_title="Vet AI Tutor · dev dashboard", page_icon=":material/pets:", layout="wide")


@st.cache_data(ttl=15, show_spinner=False)
def _summary(index_dir: str, catalog_path: str, drafts_path: str, staging_dir: str) -> dict:
    staged = [f for f in os.listdir(staging_dir) if f.endswith(".json")] if os.path.isdir(staging_dir) else []
    return {
        "chunks": status.index_stats(index_dir)["chunks"],
        "approved": len(catalog.read_json(catalog_path)),
        "drafts": len(catalog.read_json(drafts_path)),
        "staged": len([f for f in staged if f not in ("harvest_log.json", "ingest_log.json",
                                                      "ingested_hashes.json")]),
    }


page = st.navigation(
    {
        "Knowledge": [
            st.Page("app_pages/sources.py", title="Sources", icon=":material/library_books:", default=True),
            st.Page("app_pages/index.py", title="Index", icon=":material/database:"),
            st.Page("app_pages/staging.py", title="Web staging", icon=":material/travel_explore:"),
        ],
        "Quality": [
            st.Page("app_pages/evaluation.py", title="Evaluation", icon=":material/monitoring:"),
            st.Page("app_pages/quiz.py", title="Quiz", icon=":material/quiz:"),
        ],
        "Ship": [
            st.Page("app_pages/release.py", title="Release", icon=":material/rocket_launch:"),
        ],
    },
    position="top",
)

paths = get_paths()
summary = _summary(paths.index, paths.catalog, paths.drafts, paths.staging)
running = [j for j in jobs.list_jobs(limit=50) if j.active]

with st.container(horizontal=True):
    st.metric("Chunks in index", f"{summary['chunks']:,}", border=True)
    st.metric("Approved sources", summary["approved"], border=True)
    st.metric("Drafts to review", summary["drafts"], border=True)
    st.metric("Staged web records", summary["staged"], border=True)
    st.metric("Running jobs", len(running), border=True,
              help=", ".join(j.label for j in running) or "No background jobs are running")

st.title(page.title, anchor=False)
page.run()
