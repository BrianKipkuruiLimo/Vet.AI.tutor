"""Developer Ingestion Dashboard — Streamlit UI for managing the data pipeline.

This is a DEVELOPER-ONLY page for managing data ingestion.
It is NOT exposed to end users of the offline Vet AI Tutor.

Features:
  - Upload PDFs/Word docs via drag & drop
  - Enter a URL to scrape from whitelisted sources
  - View & review the staging area (approve/reject files)
  - Trigger ingestion into the vector database
  - View ingestion logs and database stats

Usage:
    streamlit run dev_dashboard.py --server.port 8502
"""

import os
import sys
import json
import glob
import shutil
import hashlib
import streamlit as st
from datetime import datetime, timezone

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from vet_ai_tutor.config import (
    VECTOR_PATH, VECTOR_BACKEND, EMBEDDING_MODEL, EMBEDDING_BACKEND,
    STAGING_DIR, WHITELIST_PATH, CHUNK_SIZE, CHUNK_OVERLAP,
)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Vet AI — Dev Dashboard",
    page_icon="⚙️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .main-header {
        background: linear-gradient(135deg, #1a1d27, #1e2233);
        padding: 20px 24px;
        border-radius: 12px;
        border: 1px solid #2a2d3a;
        margin-bottom: 24px;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #f59e0b, #ef4444);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        margin: 0;
    }
    .stat-card {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .stat-card h3 { color: #4ade80; font-size: 2rem; margin: 0; }
    .stat-card p { color: #9aa0a6; font-size: 0.85rem; margin: 4px 0 0 0; }
    .tier-1 { color: #4ade80; font-weight: 600; }
    .tier-2 { color: #f59e0b; font-weight: 600; }
    .tier-3 { color: #ef4444; font-weight: 600; }
    .status-pending { color: #f59e0b; }
    .status-approved { color: #4ade80; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
STAGING_PATH = os.path.abspath(STAGING_DIR)
INGESTED_PATH = os.path.join(STAGING_PATH, "ingested")
HARVEST_LOG_PATH = os.path.join(STAGING_PATH, "harvest_log.json")
INGEST_LOG_PATH = os.path.join(STAGING_PATH, "ingest_log.json")


def get_db_doc_count() -> int:
    """Get number of documents in the vector database."""
    if VECTOR_BACKEND == "chroma" and os.path.exists(VECTOR_PATH):
        try:
            import chromadb
            client = chromadb.PersistentClient(path=VECTOR_PATH)
            collections = client.list_collections()
            return sum(c.count() for c in collections)
        except Exception:
            return -1
    return 0


def get_staging_files() -> list[dict]:
    """List all JSON files in the staging area."""
    if not os.path.exists(STAGING_PATH):
        return []
    files = []
    for fp in sorted(glob.glob(os.path.join(STAGING_PATH, "*.json"))):
        basename = os.path.basename(fp)
        if basename in ("harvest_log.json", "ingest_log.json"):
            continue
        try:
            with open(fp, "r", encoding="utf-8") as f:
                doc = json.load(f)
            if not isinstance(doc, dict):
                continue
            files.append({
                "path": fp,
                "filename": basename,
                "species": doc.get("species", "?"),
                "category": doc.get("category", "?"),
                "source_url": doc.get("source_url", "?"),
                "source_tier": doc.get("source_tier", "?"),
                "source_title": doc.get("source_title", ""),
                "author": doc.get("author", ""),
                "journal": doc.get("journal", ""),
                "publication_year": doc.get("publication_year", ""),
                "source_domain": doc.get("source_domain", "?"),
                "status": doc.get("status", "unknown"),
                "date": doc.get("date_harvested", "?"),
                "topics": _count_topics(doc.get("extracted_data", {})),
            })
        except (json.JSONDecodeError, KeyError, AttributeError):
            continue
    return files


def _count_topics(extracted: dict) -> int:
    """Count how many topic fields have content."""
    topics = ["taxonomy", "biology", "development", "diseases",
              "nutrition", "behavior", "conservation", "veterinary_notes"]
    return sum(1 for t in topics if extracted.get(t) and extracted[t] != "null")


def get_log(path: str) -> list:
    """Load a JSON log file."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def load_whitelist() -> dict:
    """Load whitelist for display."""
    if not os.path.exists(WHITELIST_PATH):
        return {}
    with open(WHITELIST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>⚙️ Developer Ingestion Dashboard</h1>
    <p style="color: #9aa0a6; margin: 4px 0 0 0;">
        Manage the data pipeline — harvest, stage, review, and ingest veterinary knowledge.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Stats Row
# ---------------------------------------------------------------------------
doc_count = get_db_doc_count()
staging_files = get_staging_files()
pending = [f for f in staging_files if f["status"] == "pending_review"]
approved = [f for f in staging_files if f["status"] == "approved"]
harvest_log = get_log(HARVEST_LOG_PATH)
ingest_log = get_log(INGEST_LOG_PATH)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f'<div class="stat-card"><h3>{doc_count}</h3><p>Vectors in DB</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="stat-card"><h3>{len(staging_files)}</h3><p>Staged Files</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="stat-card"><h3>{len(pending)}</h3><p>Pending Review</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="stat-card"><h3>{len(harvest_log)}</h3><p>Total Harvests</p></div>', unsafe_allow_html=True)
with col5:
    st.markdown(f'<div class="stat-card"><h3>{len(ingest_log)}</h3><p>Total Ingestions</p></div>', unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_upload, tab_scrape, tab_staging, tab_ingest, tab_logs = st.tabs([
    "📄 Upload Documents", "🌐 Scrape URL", "📂 Staging Area", "🚀 Ingest to DB", "📊 Logs & Stats"
])

# ---------------------------------------------------------------------------
# Tab 1: Upload Documents
# ---------------------------------------------------------------------------
with tab_upload:
    st.subheader("Upload PDFs, Word Docs, or Text Files")
    st.caption("Files are processed and placed in the staging area for review before ingestion.")

    ucol1, ucol2 = st.columns([2, 1])

    with ucol2:
        species = st.text_input("Species Name", placeholder="e.g. giraffe", key="upload_species")
        category = st.selectbox("Category", ["wild", "domestic", "exotic", "aquatic", "avian", "unknown"], key="upload_category")
        source_tier = st.selectbox("Source Credibility Tier", ["tier_1", "tier_2"], key="upload_tier")
        source_title = st.text_input("Source title", key="upload_title")
        author = st.text_input("Author / institution", key="upload_author")
        journal = st.text_input("Journal / publisher / course", key="upload_journal")
        publication_year = st.text_input("Publication year", max_chars=4, key="upload_year")
        is_book = st.checkbox("📖 Treat as a book (detect chapters)", key="upload_book")
        use_llm = st.checkbox("🤖 Use LLM for structured extraction", value=True, key="upload_llm")

    with ucol1:
        uploaded_files = st.file_uploader(
            "Drag & drop files here",
            type=["pdf", "docx", "doc", "txt", "pptx", "json", "csv", "xlsx"],
            accept_multiple_files=True,
            key="file_uploader",
        )

    if uploaded_files and species:
        if st.button("🔄 Process Uploaded Files", type="primary", key="process_upload"):
            for uploaded in uploaded_files:
                tmp_dir = os.path.join(os.path.dirname(__file__), "data", "raw_notes", "uploaded")
                os.makedirs(tmp_dir, exist_ok=True)
                safe_name = os.path.basename(uploaded.name)
                content = uploaded.getbuffer()
                tmp_path = os.path.join(tmp_dir, f"{hashlib.sha256(content).hexdigest()[:10]}_{safe_name}")
                with open(tmp_path, "wb") as f:
                    f.write(content)

                with st.spinner(f"Processing {uploaded.name}..."):
                    try:
                        from scripts.ingest_documents import process_document
                        count = process_document(
                            tmp_path, species, category, source_tier,
                            use_llm=use_llm, is_book=is_book,
                            source_title=source_title, author=author, journal=journal,
                            publication_year=publication_year,
                        )
                        st.success(f"✅ {uploaded.name} → {count} staging file(s) created")
                    except Exception as e:
                        st.error(f"❌ Failed to process {uploaded.name}: {e}")

            st.rerun()
    elif uploaded_files and not species:
        st.warning("⚠️ Please enter a species name before processing.")

# ---------------------------------------------------------------------------
# Tab 2: Scrape URL
# ---------------------------------------------------------------------------
with tab_scrape:
    st.subheader("Scrape a Web Page for Veterinary Data")
    st.caption("The harvester will scrape the page, extract text, use an LLM to structure it, and save to staging.")

    scol1, scol2 = st.columns([2, 1])

    with scol1:
        url_input = st.text_input("URL to scrape", placeholder="https://en.wikipedia.org/wiki/Giraffe", key="scrape_url")

    with scol2:
        scrape_species = st.text_input("Species Name", placeholder="e.g. giraffe", key="scrape_species")
        scrape_category = st.selectbox("Category", ["wild", "domestic", "exotic", "aquatic", "avian", "unknown"], key="scrape_category")
        scrape_title = st.text_input("Source title", key="scrape_title")
        scrape_author = st.text_input("Author", key="scrape_author")
        scrape_journal = st.text_input("Journal / publisher", key="scrape_journal")
        scrape_year = st.text_input("Publication year", max_chars=4, key="scrape_year")

    # Show whitelist
    with st.expander("📋 Approved Domains (Whitelist)"):
        wl = load_whitelist()
        for tier_name in ("tier_1", "tier_2", "tier_3"):
            tier = wl.get(tier_name, {})
            desc = tier.get("description", "")
            domains = tier.get("domains", [])
            tier_class = tier_name.replace("_", "-")
            st.markdown(f'<span class="{tier_class}">{tier_name.upper()}</span>: {desc}', unsafe_allow_html=True)
            st.write(", ".join(domains) if domains else "None")

    if url_input and scrape_species:
        if st.button("🕷️ Scrape & Extract", type="primary", key="scrape_btn"):
            with st.spinner("Scraping and extracting structured data..."):
                try:
                    from scripts.harvester import harvest_url
                    ok = harvest_url(
                        url_input, scrape_species, scrape_category,
                        source_title=scrape_title, author=scrape_author,
                        journal=scrape_journal, publication_year=scrape_year,
                    )
                    if ok:
                        st.success("✅ Scraped and saved to staging area!")
                    else:
                        st.error("❌ Harvesting failed. Check the HTTPS URL and approved source whitelist.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            st.rerun()

# ---------------------------------------------------------------------------
# Tab 3: Staging Area
# ---------------------------------------------------------------------------
with tab_staging:
    st.subheader("Review Staged Files")
    st.caption("Approve or reject files before they are ingested into the vector database.")

    staging_files = get_staging_files()  # refresh

    if not staging_files:
        st.info("No files in the staging area. Upload a document or scrape a URL first.")
    else:
        # Filters
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            filter_status = st.selectbox("Filter by Status", ["All", "pending_review", "approved"], key="filter_status")
        with fcol2:
            all_species = sorted(set(f["species"] for f in staging_files))
            filter_species = st.selectbox("Filter by Species", ["All"] + all_species, key="filter_species")
        with fcol3:
            st.write("")  # spacer
            st.caption("Approve records individually after checking their source and extracted content.")

        # Display files
        filtered = staging_files
        if filter_status != "All":
            filtered = [f for f in filtered if f["status"] == filter_status]
        if filter_species != "All":
            filtered = [f for f in filtered if f["species"] == filter_species]

        for i, f in enumerate(filtered):
            tier_class = f["source_tier"].replace("_", "-")
            status_class = "status-approved" if f["status"] == "approved" else "status-pending"

            with st.expander(
                f"{'✅' if f['status'] == 'approved' else '⏳'} {f['filename']}  —  "
                f"{f['species']} · {f['source_tier']} · {f['topics']} topics"
            ):
                dcol1, dcol2 = st.columns([2, 1])
                with dcol1:
                    st.markdown(f"**Species:** {f['species']}")
                    st.markdown(f"**Category:** {f['category']}")
                    st.markdown(f"**Source:** {f['source_url'][:80]}...")
                    with open(f["path"], "r", encoding="utf-8") as fh:
                        full_doc = json.load(fh)
                    for label, key in (("Title", "source_title"), ("Author", "author"),
                                       ("Journal / publisher", "journal"), ("Year", "publication_year")):
                        value = st.text_input(label, value=str(full_doc.get(key, "")), key=f"{key}_{i}")
                        full_doc[key] = value
                    if st.button("💾 Save provenance", key=f"save_meta_{i}"):
                        with open(f["path"], "w", encoding="utf-8") as fh:
                            json.dump(full_doc, fh, indent=2, ensure_ascii=False)
                        st.success("Saved source metadata. Review the extracted content before approving.")
                    st.markdown(f"**Domain:** {f['source_domain']}")
                    st.markdown(f"**Date:** {f['date'][:19]}")
                    st.markdown(f"**Topics extracted:** {f['topics']}")

                with dcol2:
                    st.markdown(f'**Status:** <span class="{status_class}">{f["status"]}</span>', unsafe_allow_html=True)
                    st.markdown(f'**Tier:** <span class="{tier_class}">{f["source_tier"]}</span>', unsafe_allow_html=True)

                    if f["status"] == "pending_review":
                        if st.button("✅ Approve", key=f"approve_{i}"):
                            with open(f["path"], "r", encoding="utf-8") as fh:
                                doc = json.load(fh)
                            missing = [key for key in ("source_title", "author", "journal", "publication_year", "source_url")
                                       if not str(doc.get(key, "")).strip()]
                            if missing or doc.get("source_tier") not in {"tier_1", "tier_2"}:
                                st.error(f"Complete verified provenance and an approved tier before approval: {', '.join(missing) or 'tier_1/tier_2 required'}")
                                st.stop()
                            year = str(doc.get("publication_year", ""))
                            if not (len(year) == 4 and year.isdigit()):
                                st.error("Publication year must be four digits.")
                                st.stop()
                            doc["status"] = "approved"
                            doc["approved_at"] = datetime.now(timezone.utc).isoformat()
                            with open(f["path"], "w", encoding="utf-8") as fh:
                                json.dump(doc, fh, indent=2, ensure_ascii=False)
                            st.rerun()

                    if st.button("🗑️ Delete", key=f"delete_{i}"):
                        os.remove(f["path"])
                        st.rerun()

                # Show extracted data preview
                try:
                    with open(f["path"], "r", encoding="utf-8") as fh:
                        full_doc = json.load(fh)
                    extracted = full_doc.get("extracted_data", {})
                    preview_topics = {k: v for k, v in extracted.items()
                                     if v and v != "null" and k not in ("_extraction_failed", "raw_text")}
                    if full_doc.get("raw_source_text"):
                        st.text_area("Captured source text", full_doc["raw_source_text"], height=220, key=f"raw_source_{i}")
                    if preview_topics:
                        st.json(preview_topics)
                except Exception:
                    pass

# ---------------------------------------------------------------------------
# Tab 4: Ingest to DB
# ---------------------------------------------------------------------------
with tab_ingest:
    st.subheader("Ingest Approved Files Into the Vector Database")

    staging_files = get_staging_files()
    approved_files = [f for f in staging_files if f["status"] == "approved"]

    icol1, icol2 = st.columns([2, 1])

    with icol1:
        st.markdown(f"**Approved files ready:** {len(approved_files)}")
        st.markdown(f"**Current DB documents:** {get_db_doc_count()}")
        st.markdown(f"**Vector backend:** {VECTOR_BACKEND}")
        st.markdown(f"**Embedding model:** {EMBEDDING_MODEL} ({EMBEDDING_BACKEND})")

    with icol2:
        st.markdown(f"**Chunk size:** {CHUNK_SIZE}")
        st.markdown(f"**Chunk overlap:** {CHUNK_OVERLAP}")
        st.markdown(f"**DB path:** `{VECTOR_PATH}`")

    if not approved_files:
        st.info("No approved files to ingest. Go to the Staging Area tab to approve files first.")
    else:
        st.markdown("**Files to ingest:**")
        for f in approved_files:
            st.write(f"  • {f['filename']} ({f['species']} · {f['source_tier']})")

        if st.button("🚀 Ingest All Approved Files", type="primary", key="ingest_btn"):
            progress = st.progress(0, text="Starting ingestion...")
            status_area = st.empty()

            try:
                from scripts.ingest_staged import (
                    load_embeddings, get_vectorstore, ingest_file,
                )

                # Load embeddings
                status_area.info("Loading embedding model...")
                embeddings = load_embeddings()
                if embeddings is None:
                    st.error("❌ Failed to load embeddings. Is Ollama running?")
                    st.stop()

                # Open vectorstore
                os.makedirs(VECTOR_PATH, exist_ok=True)
                db = get_vectorstore(embeddings)

                total_docs = 0
                for idx, f in enumerate(approved_files):
                    progress.progress(
                        (idx + 1) / len(approved_files),
                        text=f"Ingesting {f['filename']}..."
                    )
                    status_area.info(f"Processing: {f['filename']} ({f['species']})")
                    db, count = ingest_file(f["path"], embeddings, db)
                    total_docs += count

                progress.progress(1.0, text="Done!")
                st.success(f"✅ Ingested {total_docs} documents from {len(approved_files)} files!")

            except Exception as e:
                st.error(f"❌ Ingestion failed: {e}")
                import traceback
                st.code(traceback.format_exc())

            st.rerun()

    st.markdown("---")
    st.subheader("📦 Create Data Patch")
    st.caption("Package the current vector database for offline distribution.")

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        patch_version = st.text_input("Version tag", placeholder="e.g. v1.0", key="patch_version")
    with pcol2:
        st.write("")
        if st.button("📦 Create Patch", key="create_patch_btn"):
            if not patch_version:
                st.warning("Enter a version tag first.")
            else:
                with st.spinner("Creating data patch..."):
                    try:
                        from scripts.create_data_patch import create_patch
                        create_patch(version=patch_version)
                        patches_dir = os.path.join(os.path.dirname(__file__), "data", "patches")
                        st.success(f"✅ Patch created! Check: `{patches_dir}`")
                    except Exception as e:
                        st.error(f"❌ Failed: {e}")

# ---------------------------------------------------------------------------
# Tab 5: Logs & Stats
# ---------------------------------------------------------------------------
with tab_logs:
    lcol1, lcol2 = st.columns(2)

    with lcol1:
        st.subheader("🕷️ Harvest Log")
        harvest_log = get_log(HARVEST_LOG_PATH)
        if harvest_log:
            for entry in reversed(harvest_log[-20:]):
                status_emoji = "✅" if entry.get("status") == "staged" else "❌"
                ts = entry.get("timestamp", "")[:19]
                st.markdown(
                    f"{status_emoji} **{entry.get('species', '?')}** — "
                    f"`{entry.get('status', '?')}` — {ts}"
                )
                st.caption(f"  {entry.get('url', '?')[:60]}")
        else:
            st.info("No harvest log entries yet.")

    with lcol2:
        st.subheader("📥 Ingest Log")
        ingest_log = get_log(INGEST_LOG_PATH)
        if ingest_log:
            for entry in reversed(ingest_log[-20:]):
                status_emoji = "✅" if entry.get("status") == "ingested" else "⚠️"
                ts = entry.get("timestamp", "")[:19]
                st.markdown(
                    f"{status_emoji} **{entry.get('species', '?')}** — "
                    f"{entry.get('documents_created', 0)} docs — {ts}"
                )
                st.caption(f"  {entry.get('file', '?')}")
        else:
            st.info("No ingest log entries yet.")

    st.markdown("---")

    st.subheader("🗄️ Database Info")
    st.markdown(f"- **Backend:** {VECTOR_BACKEND}")
    st.markdown(f"- **Path:** `{os.path.abspath(VECTOR_PATH)}`")
    st.markdown(f"- **Embedding model:** {EMBEDDING_MODEL} ({EMBEDDING_BACKEND})")
    st.markdown(f"- **Document count:** {get_db_doc_count()}")

    db_exists = os.path.exists(VECTOR_PATH)
    st.markdown(f"- **Database exists:** {'✅ Yes' if db_exists else '❌ No'}")

    if db_exists:
        # Show size
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(VECTOR_PATH):
            for fn in filenames:
                total_size += os.path.getsize(os.path.join(dirpath, fn))
        st.markdown(f"- **Database size:** {total_size / (1024*1024):.1f} MB")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("⚙️ Developer Dashboard · Not for end users · Vet AI Tutor")
