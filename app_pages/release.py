import os

import pandas as pd
import streamlit as st

from vet_ai_tutor.devtools import jobs, status, ui
from vet_ai_tutor.devtools.paths import get_paths
from vet_ai_tutor.signing import generate_keypair

paths = get_paths()
release = status.release_status(paths)

st.subheader("Signing key", anchor=False)
st.caption("Offline installs accept only knowledge patches signed with this key.")
with st.container(horizontal=True):
    st.badge("Private key present" if release["private_key"] else "No private key",
             color="green" if release["private_key"] else "red", icon=":material/key:")
    st.badge("Public key present" if release["public_key"] else "No public key",
             color="green" if release["public_key"] else "red", icon=":material/verified_user:")
    if release["fingerprint"]:
        st.badge(f"Fingerprint {release['fingerprint']}", color="gray", icon=":material/fingerprint:")
st.caption(f"Private: `{paths.signing_key}` · public: `{os.path.relpath(paths.public_key, paths.root)}`")

if not release["private_key"]:
    if release["public_key"]:
        st.warning("The public key exists but its private key is missing on this machine, so new "
                   "patches cannot be signed. Restore the private key from your backup rather than "
                   "generating a new one: installed copies only trust the existing key.",
                   title="Private key missing")
    else:
        st.info("Generate a key pair once. Back up the private key: if it is lost, new patches can't "
                "be signed for existing installs.", title="No signing key yet")
        if st.button("Generate signing key", icon=":material/key:", type="primary"):
            generate_keypair(paths.signing_key, paths.public_key)
            st.toast("Key pair created", icon=":material/key:")
            st.rerun()

st.subheader("Knowledge patch", anchor=False)
st.caption("Packages the active index (and quiz bank) for USB transfer. Stop the tutor first.")
with st.form("patch_form", border=False):
    with st.container(horizontal=True, vertical_alignment="bottom"):
        version = st.text_input("Version", placeholder="v2.1")
        submitted = st.form_submit_button("Create patch", icon=":material/inventory_2:", type="primary",
                                          disabled=not release["private_key"])
    if submitted:
        if not version.strip():
            st.error("Enter a version.")
        else:
            ui.start_job("create_patch", ["--version", version.strip()])
ui.job_panel("create_patch")

if release["patches"]:
    st.dataframe(pd.DataFrame(release["patches"]), hide_index=True, alt="Created knowledge patches",
                 column_config={"created": st.column_config.DatetimeColumn("Created", format="YYYY-MM-DD HH:mm"),
                                "size_mb": st.column_config.NumberColumn("Size (MB)", format="%.1f")})

with st.expander("Offline bundle", icon=":material/usb:"):
    st.caption("Build on a connected machine with the same Linux and Python version as the target.")
    st.code("python scripts/build_offline_bundle.py \\\n  --model-store \"$HOME/.ollama/models\" \\\n"
            "  --ollama-installer ./ollama-linux-amd64.deb \\\n  --output ./dist/vet-ai-tutor-offline.zip",
            language="bash")

st.subheader("Recent activity", anchor=False)
history = [{"job": j.label, "status": j.status, "started": j.started, "finished": j.finished}
           for j in jobs.list_jobs(limit=25)]
if history:
    st.dataframe(pd.DataFrame(history), hide_index=True, alt="Recent background jobs")
updates = status.recent_log(os.path.join(paths.root, "data", "update_log.json"))
if updates:
    st.caption("Patches applied on this machine")
    st.dataframe(pd.DataFrame(updates), hide_index=True, alt="Applied knowledge patches")
