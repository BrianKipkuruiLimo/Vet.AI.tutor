import json
import os

import pandas as pd
import streamlit as st

from vet_ai_tutor.devtools import catalog, status, ui
from vet_ai_tutor.devtools.paths import get_paths
from vet_ai_tutor.ingestion import staging
from vet_ai_tutor.ingestion.provenance import ProvenanceError

paths = get_paths()
CATEGORIES = ["domestic", "wild", "exotic", "aquatic", "avian", "unknown"]
TOPIC_KEYS = staging.TOPICS + ["summary", "common_name", "scientific_name"]

st.caption("Web pages from whitelisted HTTPS domains are scraped, structured by the local model and "
           "staged here. Each record needs complete provenance and a person's approval before ingestion.")

with st.expander("Harvest a web page", icon=":material/travel_explore:"):
    with st.form("harvest_form"):
        url = st.text_input("HTTPS URL on an approved domain", placeholder="https://www.merckvetmanual.com/...")
        with st.container(horizontal=True):
            species = st.text_input("Species", placeholder="cattle")
            category = st.selectbox("Category", CATEGORIES)
        with st.container(horizontal=True):
            title = st.text_input("Source title")
            author = st.text_input("Author")
        with st.container(horizontal=True):
            journal = st.text_input("Journal or publisher")
            year = st.text_input("Publication year", max_chars=4)
        if st.form_submit_button("Harvest", icon=":material/download:", type="primary"):
            if not url.startswith("https://") or not species:
                st.error("Enter an HTTPS URL and a species.")
            else:
                ui.start_job("harvest", ["--url", url, "--species", species, "--category", category,
                                         "--source-title", title, "--author", author,
                                         "--journal", journal, "--year", year])
    whitelist = catalog.read_json(paths.whitelist)
    for tier in ("tier_1", "tier_2"):
        group = whitelist.get(tier, {})
        st.caption(f"**{tier}** · {group.get('description', '')}: {', '.join(group.get('domains', []))}")
ui.job_panel("harvest")

records = staging.list_staging_files(staging_dir=paths.staging)
st.subheader("Staged records", anchor=False)
if not records:
    st.info("No staged records. Harvest a page above.", icon=":material/inbox:")
else:
    with st.container(horizontal=True, vertical_alignment="bottom"):
        show = st.segmented_control("Status", ["pending_review", "approved", "All"], default="pending_review",
                                    key="staging_status", format_func=lambda s: s.replace("_", " "))
        species_list = sorted({r["species"] for r in records})
        pick_species = st.selectbox("Species", ["All species", *species_list], key="staging_species")
    shown = [r for r in records if show in (None, "All") or r["status"] == show]
    if pick_species != "All species":
        shown = [r for r in shown if r["species"] == pick_species]
    st.caption(f"{len(shown)} of {len(records)} records")

    for rec in shown:
        icon = ":material/verified:" if rec["status"] == "approved" else ":material/pending:"
        exp = st.expander(f"{rec['source_title'] or rec['filename']} · {rec['species']} · {rec['source_tier']}",
                          icon=icon, key=f"rec_{rec['filename']}", on_change="rerun")
        if not exp.open:
            continue  # read and render the record only when it is open
        with exp:
            with open(rec["path"], encoding="utf-8") as f:
                doc = json.load(f)
            for warning in doc.get("injection_warnings") or []:
                st.warning(warning, title="Possible prompt-injection text")
            st.caption(f"{rec['source_url']} · harvested {str(rec['date'])[:19]} · status {rec['status']}")
            with st.form(f"prov_{rec['filename']}"):
                with st.container(horizontal=True):
                    new_title = st.text_input("Title", doc.get("source_title", ""))
                    new_author = st.text_input("Author", doc.get("author", ""))
                with st.container(horizontal=True):
                    new_journal = st.text_input("Journal or publisher", doc.get("journal", ""))
                    new_year = st.text_input("Year", str(doc.get("publication_year", "")), max_chars=4)
                    tiers = ["tier_1", "tier_2"]
                    new_tier = st.selectbox("Tier", tiers, index=tiers.index(doc["source_tier"])
                                            if doc.get("source_tier") in tiers else 0)
                reviewed = st.checkbox("I checked the flagged text; it is harmless",
                                       value=bool(doc.get("injection_warnings_reviewed")),
                                       disabled=not doc.get("injection_warnings"))
                if st.form_submit_button("Save provenance", icon=":material/save:"):
                    doc.update(source_title=new_title, author=new_author, journal=new_journal,
                               publication_year=new_year, source_tier=new_tier,
                               injection_warnings_reviewed=reviewed)
                    catalog.write_json(rec["path"], doc)
                    st.toast("Saved", icon=":material/save:")
                    st.rerun()
            with st.container(horizontal=True):
                if rec["status"] != "approved" and st.button("Approve", key=f"approve_{rec['filename']}",
                                                             icon=":material/verified:", type="primary"):
                    try:
                        staging.approve_file(rec["path"])
                        st.rerun()
                    except (ProvenanceError, ValueError) as exc:
                        st.error(str(exc))
                with st.popover("Delete", icon=":material/delete:"):
                    st.caption("The staged record is removed; the source web page is not affected.")
                    if st.button("Delete record", key=f"delete_{rec['filename']}", type="primary"):
                        os.remove(rec["path"])
                        st.rerun()
            topics = {k: v for k, v in (doc.get("extracted_data") or {}).items()
                      if k in TOPIC_KEYS and v and v != "null"}
            if topics:
                st.json(topics, expanded=False)
            if doc.get("raw_source_text"):
                st.code(doc["raw_source_text"][:6000], language=None, wrap_lines=True, height=240)

approved = [r for r in records if r["status"] == "approved"]
st.subheader("Ingest approved records", anchor=False)
st.caption("Adds approved records to the active index without a full rebuild.")
if st.button(f"Ingest {len(approved)} approved record(s)", icon=":material/input:", disabled=not approved):
    ui.start_job("ingest_staged")
ui.job_panel("ingest_staged")

with st.expander("Harvest and ingest logs", icon=":material/history:"):
    left, right = st.columns(2)
    with left:
        harvests = status.recent_log(os.path.join(paths.staging, "harvest_log.json"))
        st.dataframe(pd.DataFrame(harvests, columns=["timestamp", "status", "species", "url"]),
                     hide_index=True, alt="Recent harvests")
    with right:
        ingests = status.recent_log(os.path.join(paths.staging, "ingest_log.json"))
        st.dataframe(pd.DataFrame(ingests, columns=["timestamp", "status", "file", "documents_created"]),
                     hide_index=True, alt="Recent ingestions")
