import pandas as pd
import streamlit as st

from vet_ai_tutor.config import CHUNK_OVERLAP, CHUNK_SIZE, INGEST_CACHE_DIR, project_path
from vet_ai_tutor.devtools import status, ui
from vet_ai_tutor.devtools.paths import get_paths

paths = get_paths()


@st.cache_data(ttl=20, show_spinner=False)
def index_stats(index_dir: str, cache_dir: str) -> dict:
    return status.index_stats(index_dir, cache_dir)


@st.cache_data(ttl=300, show_spinner=False)
def tools() -> dict:
    return status.tool_status()


stats = index_stats(paths.index, project_path(INGEST_CACHE_DIR))
profile = stats["profile"] or {}

with st.container(horizontal=True):
    st.metric("Chunks", f"{stats['chunks']:,}", border=True)
    st.metric("Index size", f"{stats['size_mb']} MB", border=True)
    st.metric("Embedding model", profile.get("embedding_model", "unknown"), border=True,
              help=f"Document prefix: {profile.get('document_prefix', '?')!r}")
    st.metric("Embedding cache", f"{stats.get('embedding_cache_mb', 0)} MB", border=True,
              help="Vectors reused by later rebuilds")
if stats["error"]:
    st.error(stats["error"], title="Could not read the index")
if stats["exists"] and not stats["profile"]:
    st.warning("This index predates embedding prefixes. Rebuild it for better retrieval.",
               title="Rebuild recommended")

st.subheader("Rebuild", anchor=False)
st.caption(f"Indexes every approved source in data/raw_notes ({CHUNK_SIZE}-character chunks, "
           f"{CHUNK_OVERLAP} overlap). Start with a dry run: it checks every file in about a minute "
           "without embedding anything.")
with st.container(horizontal=True, vertical_alignment="bottom"):
    if st.button("Dry run", icon=":material/fact_check:"):
        ui.start_job("dry_run")
    confirm = st.checkbox("I understand a first full rebuild takes hours on this laptop", key="confirm_rebuild")
    if st.button("Rebuild index", icon=":material/build:", type="primary", disabled=not confirm,
                 help="Embedding is cached: later rebuilds only embed new or changed chunks"):
        ui.start_job("rebuild")
ui.job_panel("dry_run")
ui.job_panel("rebuild")

report = status.rebuild_report(paths.rebuild_summary)
if report:
    kind = "Dry run" if report.get("dry_run") else "Rebuild"
    st.subheader(f"Last report: {kind.lower()}", anchor=False)
    st.caption(f"Written {report['written_at']:%Y-%m-%d %H:%M} UTC")
    with st.container(horizontal=True):
        st.metric("Indexed files", len(report.get("indexed_files", [])), border=True)
        st.metric("Chunks", f"{report.get('chunks', 0):,}", border=True)
        st.metric("Not approved", len(report.get("not_approved", [])), border=True)
        st.metric("Unreadable", len(report.get("unreadable", [])), border=True)
        st.metric("Need OCR", len(report.get("needs_ocr", [])), border=True)
        st.metric("No text", len(report.get("empty", [])), border=True)
    detail = st.segmented_control("Details", ["Unreadable", "Need OCR", "No text", "Not approved"],
                                  key="report_detail", label_visibility="collapsed")
    if detail == "Unreadable":
        st.dataframe(pd.DataFrame(report.get("unreadable", []), columns=["file", "error"]), hide_index=True,
                     alt="Files that could not be read")
    elif detail == "Need OCR":
        st.dataframe(pd.DataFrame({"file": report.get("needs_ocr", [])}), hide_index=True,
                     alt="Mostly scanned PDFs")
    elif detail == "No text":
        st.dataframe(pd.DataFrame({"file": report.get("empty", [])}), hide_index=True,
                     alt="Files without extractable text")
    elif detail == "Not approved":
        st.dataframe(pd.DataFrame(report.get("not_approved", []), columns=["file", "reason"]),
                     hide_index=True, alt="Files without a reviewed catalog entry")

st.subheader("Scanned PDFs and legacy files", anchor=False)
available = tools()
with st.container(horizontal=True):
    st.badge("Tesseract installed" if available["tesseract"] else "Tesseract missing",
             color="green" if available["tesseract"] else "red", icon=":material/document_scanner:")
    writer = available["libreoffice_writer"]
    st.badge("LibreOffice document components installed" if writer else
             "LibreOffice can't open documents yet" if available["libreoffice"] else "LibreOffice missing",
             color="green" if writer else "orange", icon=":material/description:")
if not available["tesseract"]:
    st.code("sudo apt install tesseract-ocr", language="bash")
if available["libreoffice"] and writer is False:
    st.caption("Needed to convert legacy .doc and .ppt files:")
    st.code("sudo apt install libreoffice-writer-nogui libreoffice-impress-nogui", language="bash")

st.caption("OCR reads pages that have no text layer. Pages below the confidence threshold "
           "(handwriting, poor scans) are recorded but not indexed. Rebuild afterwards.")
with st.container(horizontal=True, vertical_alignment="bottom"):
    all_empty = st.toggle("Also image-only pages inside text PDFs", key="ocr_all_empty")
    if st.button("Run OCR", icon=":material/document_scanner:", disabled=not available["tesseract"]):
        ui.start_job("ocr", ["--all-empty"] if all_empty else [])
ui.job_panel("ocr")
