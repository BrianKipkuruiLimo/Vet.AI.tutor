"""Shared Streamlit widgets for the developer dashboard."""
from typing import List, Optional

import streamlit as st

from vet_ai_tutor.devtools import jobs

_STATUS_BADGE = {
    "starting": ("Starting", "blue", ":material/hourglass_top:"),
    "running": ("Running", "blue", ":material/autorenew:"),
    "succeeded": ("Finished", "green", ":material/check_circle:"),
    "failed": ("Failed", "red", ":material/error:"),
    "cancelled": ("Cancelled", "gray", ":material/cancel:"),
}


def start_job(kind: str, args: Optional[List[str]] = None, meta: Optional[dict] = None) -> None:
    """Start a background job from a button handler and show the outcome."""
    try:
        job = jobs.start_job(kind, args, meta=meta)
    except jobs.JobConflict as exc:
        st.error(str(exc), icon=":material/block:")
        return
    st.toast(f"Started: {job.label}", icon=":material/play_arrow:")
    st.rerun()


def job_panel(kind: str, title: Optional[str] = None) -> Optional[jobs.Job]:
    """Show the latest job of `kind`; refreshes itself every 2 s while it runs."""
    latest = jobs.list_jobs(kind, limit=1)
    job = latest[0] if latest else None

    @st.fragment(run_every="2s" if job and job.active else None)
    def _panel():
        current = jobs.get_job(job.id) if job else None
        if current is None:
            st.caption(f"No {title or jobs.JOB_KINDS[kind][1].lower()} has run yet.")
            return
        if job and job.active and not current.active:
            st.rerun()  # finished: refresh the whole page once to show new results
        label, color, icon = _STATUS_BADGE.get(current.status, (current.status, "gray", None))
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.badge(label, color=color, icon=icon)
                st.caption(f"{current.label} · started {current.started.replace('T', ' ')} UTC"
                           + (f" · finished {current.finished[11:19]}" if current.finished else ""))
                if current.active and st.button("Cancel", key=f"cancel_{kind}", icon=":material/stop:",
                                                type="tertiary"):
                    jobs.cancel_job(current.id)
                    st.rerun()
            line = jobs.last_line(current)
            if line:
                st.text(line)
            if current.meta.get("error"):
                st.error(current.meta["error"])
            with st.expander("Output", icon=":material/terminal:"):
                st.code(jobs.read_log(current) or "(no output yet)", language=None, height=300)

    _panel()
    return job
