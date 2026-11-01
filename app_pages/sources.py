import os

import pandas as pd
import streamlit as st

from vet_ai_tutor.devtools import catalog, ui
from vet_ai_tutor.devtools.paths import get_paths
from vet_ai_tutor.ingestion.loaders import load_file

paths = get_paths()
EDITABLE = list(catalog.EDITABLE_FIELDS)


@st.cache_data(max_entries=64, show_spinner="Reading document…")
def preview(path: str, mtime: float) -> dict:
    """Extracted text of one source file (cached per file version)."""
    try:
        loaded = load_file(path)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"pages": loaded.total_pages, "with_text": len(loaded.documents), "empty": loaded.empty_pages,
            "needs_ocr": loaded.needs_ocr, "notes": loaded.notes,
            "text": "\n\n".join(d.page_content for d in loaded.documents)[:4000]}


st.caption("Review the metadata drafted for each file in data/raw_notes. Only approved sources are "
           "indexed; approval runs the same checks as the rebuild.")

rows = catalog.review_rows(paths.raw_dir, paths.catalog, paths.drafts, paths.whitelist)
df = pd.DataFrame(rows, columns=["key", "status", "course", *EDITABLE, "problem", "file_exists", "path"])
counts = df["status"].value_counts().to_dict() if len(df) else {}

with st.container(horizontal=True, vertical_alignment="bottom"):
    with st.popover("Add documents", icon=":material/upload_file:"):
        uploads = st.file_uploader("Lecture notes, slides or PDFs", accept_multiple_files=True,
                                   type=[e.lstrip(".") for e in catalog.SUPPORTED_EXTENSIONS])
        up_course = st.text_input("Course or institution (journal field)", key="upload_course")
        up_author = st.text_input("Default author", key="upload_author")
        if st.button("Save to raw notes", disabled=not uploads, type="primary"):
            for up in uploads or []:
                try:
                    key = catalog.add_upload(up.name, up.getvalue(), paths.raw_dir, paths.drafts,
                                             up_course, up_author)
                    st.toast(f"Added {key}", icon=":material/check:")
                except ValueError as exc:
                    st.error(str(exc))
            st.rerun()
    with st.popover("Draft new files", icon=":material/auto_fix_high:",
                    help="Draft metadata for files that have neither a draft nor a catalog entry"):
        d_course = st.text_input("Course or institution", key="draft_course")
        d_author = st.text_input("Default author", key="draft_author")
        st.caption("Existing drafts and your edits are kept.")
        if st.button("Run drafter", type="primary"):
            args = (["--course", d_course] if d_course else []) + \
                   (["--default-author", d_author] if d_author else [])
            ui.start_job("draft_metadata", args)
    st.caption(" · ".join(f"{counts.get(s, 0)} {s}" for s in ("approved", "draft", "invalid", "no draft")))

ui.job_panel("draft_metadata")

view = st.segmented_control("View", ["Drafts", "Approved"], default="Drafts", key="sources_view",
                            label_visibility="collapsed")

with st.container(horizontal=True, vertical_alignment="bottom"):
    courses = sorted(c for c in df["course"].unique() if c) if len(df) else []
    course = st.selectbox("Course", ["All courses", *courses, "No course code"], key="sources_course")
    search = st.text_input("Search file, title or author", key="sources_search", icon=":material/search:")
    only_ready = view == "Drafts" and st.toggle("Ready to approve only", key="sources_ready")

shown = df[df["status"].isin(["draft", "no draft"] if view == "Drafts" else ["approved", "invalid"])]
if course == "No course code":
    shown = shown[shown["course"] == ""]
elif course != "All courses":
    shown = shown[shown["course"] == course]
if search:
    needle = search.lower()
    shown = shown[shown[["key", "source_title", "author"]].apply(
        lambda r: needle in " ".join(r).lower(), axis=1)]
if only_ready:
    shown = shown[(shown["problem"] == "") & (shown["status"] == "draft")]
shown = shown.reset_index(drop=True)

if view == "Drafts":
    ready = shown[(shown["status"] == "draft") & (shown["problem"] == "")]
    st.caption(f"{len(shown)} files in this view · {len(ready)} ready to approve. "
               "Edit cells, save, then approve. The tier must be tier_1 or tier_2.")
    edited = st.data_editor(
        shown[["key", "course", *EDITABLE, "problem"]], key="drafts_editor", hide_index=True,
        num_rows="fixed", disabled=["key", "course", "problem"], height=420,
        alt="Draft source metadata, one row per document",
        column_config={
            "key": st.column_config.TextColumn("File", width="medium", pinned=True),
            "source_title": st.column_config.TextColumn("Title", width="large"),
            "publication_year": st.column_config.TextColumn("Year", width="small", max_chars=4),
            "source_tier": st.column_config.SelectboxColumn("Tier", options=["", "tier_1", "tier_2"],
                                                            width="small"),
            "source_url": st.column_config.TextColumn("Source URL", width="medium"),
            "problem": st.column_config.TextColumn("Blocking approval", width="large"),
        },
    )
    changes = {
        row["key"]: {f: row[f] for f in EDITABLE if row[f] != shown.at[i, f]}
        for i, row in edited.iterrows()
    }
    changes = {k: v for k, v in changes.items() if v}

    with st.container(horizontal=True, vertical_alignment="bottom"):
        if st.button(f"Save {len(changes)} edited row(s)", icon=":material/save:", disabled=not changes):
            catalog.apply_edits(paths.drafts, changes)
            st.toast("Edits saved", icon=":material/save:")
            st.rerun()
        if st.button(f"Approve {len(ready)} ready draft(s)", icon=":material/verified:", type="primary",
                     disabled=bool(changes) or ready.empty,
                     help="Save edits first" if changes else "Moves every ready draft in this view into the catalog"):
            approved, rejected = catalog.approve(list(ready["key"]), paths.drafts, paths.catalog,
                                                 paths.whitelist, paths.backups)
            st.toast(f"Approved {len(approved)} source(s)", icon=":material/verified:")
            for key, reason in rejected.items():
                st.error(f"{key}: {reason}")
            if not rejected:
                st.rerun()
        with st.popover("Set a field for this view", icon=":material/edit_note:", disabled=shown.empty):
            field = st.selectbox("Field", EDITABLE, format_func=lambda f: f.replace("_", " "), key="bulk_field")
            if field == "source_tier":
                value = st.selectbox("Value", ["tier_1", "tier_2"], key="bulk_tier")
            else:
                value = st.text_input("Value", key="bulk_value")
            drafts_in_view = list(shown[shown["status"] == "draft"]["key"])
            if st.button(f"Apply to {len(drafts_in_view)} draft(s)", type="primary", disabled=not value):
                catalog.bulk_set(paths.drafts, drafts_in_view, field, value)
                st.rerun()
else:
    st.caption(f"{len(shown)} approved sources in this view. Select rows to move them back to drafts "
               "for correction.")
    selection = st.dataframe(
        shown[["key", "status", "course", *EDITABLE, "problem"]], key="approved_table", hide_index=True,
        on_select="rerun", selection_mode="multi-row", height=420, alt="Approved source catalog",
        column_config={"key": st.column_config.TextColumn("File", pinned=True),
                       "problem": st.column_config.TextColumn("Problem (invalid entries)")},
    )
    picked = [shown.at[i, "key"] for i in selection.selection.rows]
    if st.button(f"Move {len(picked)} back to drafts", icon=":material/undo:", disabled=not picked):
        catalog.unapprove(picked, paths.drafts, paths.catalog, paths.backups)
        st.rerun()

st.subheader("Preview extracted text", anchor=False)
previewable = [k for k, ok in zip(shown["key"], shown["file_exists"], strict=True) if ok]
choice = st.selectbox("File", previewable, index=None, placeholder="Choose a file in the current view",
                      key="preview_file")
if choice:
    path = df.loc[df["key"] == choice, "path"].iloc[0]
    info = preview(path, os.path.getmtime(path))
    if "error" in info:
        st.error(info["error"], title="This file cannot be read")
    else:
        with st.container(horizontal=True):
            st.metric("Pages / slides", info["pages"], border=True)
            st.metric("With text", info["with_text"], border=True)
            st.metric("Empty", len(info["empty"]), border=True)
        if info["needs_ocr"]:
            st.warning("Most pages have no text layer. Run OCR from the Index page before rebuilding.",
                       title="Scanned document")
        for note in info["notes"]:
            st.info(note)
        st.code(info["text"] or "(no extractable text)", language=None, wrap_lines=True, height=320)
