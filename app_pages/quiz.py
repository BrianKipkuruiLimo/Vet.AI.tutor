from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

from scripts.generate_quiz import load_pending, save_pending
from scripts.review_quiz import apply_decision, publish
from vet_ai_tutor.config import LLM_MODEL
from vet_ai_tutor.devtools import status, ui
from vet_ai_tutor.devtools.paths import get_paths
from vet_ai_tutor.quiz import bank_path, load_bank

paths = get_paths()
pending_path = Path(paths.quiz_pending)
bank_file = bank_path(paths.index)

stats = status.quiz_stats(paths.quiz_pending, bank_file)
with st.container(horizontal=True):
    st.metric("Drafts to review", stats["drafts"].get("pending_review", 0), border=True)
    st.metric("Approved drafts", stats["drafts"].get("approved", 0), border=True,
              help="Publishing copies approved drafts into the bank")
    st.metric("Rejected", stats["drafts"].get("rejected", 0), border=True)
    st.metric("Published questions", stats["published"], border=True)

with st.expander("Generate drafts", icon=":material/auto_awesome:"):
    st.caption("About 20–40 s per question. Drafts that fail the automatic checks (malformed options, "
               "unsupported answer, a second correct option) are discarded.")
    with st.form("generate_form", border=False):
        with st.container(horizontal=True):
            limit = st.number_input("Questions to attempt", 1, 200, 10)
            per_source = st.number_input("Per source document", 1, 10, 2)
            model = st.text_input("Model", LLM_MODEL, help="qwen3:4b writes better questions, at half the speed")
        sources = st.text_input("Only these sources (comma-separated words from file names or titles)")
        if st.form_submit_button("Generate", icon=":material/auto_awesome:", type="primary"):
            ui.start_job("generate_quiz", ["--limit", str(limit), "--per-source", str(per_source),
                                           "--model", model] + (["--sources", sources] if sources else []))
ui.job_panel("generate_quiz")

st.subheader("Review", anchor=False)
items = load_pending(pending_path)
queue = [q for q in items if q.get("status") == "pending_review"]
if not queue:
    st.info("Nothing to review. Generate drafts above.", icon=":material/task_alt:")
else:
    entry = queue[0]
    where = entry["source"] + (f" · p. {entry['page']}" if "page" in entry else "") + \
        (f" · slide {entry['slide']}" if "slide" in entry else "")
    st.caption(f"{len(queue)} left · {where}")
    passage_col, question_col = st.columns([1, 1])
    with passage_col, st.container(border=True):
        st.markdown("**Source passage**")
        st.code(entry["passage"][:2000], language=None, wrap_lines=True, height=360)
    with question_col, st.container(border=True):
        st.markdown(f"**{entry['question']}**")
        correct = st.radio("Correct option", range(4), index=entry["answer_index"], key=f"answer_{entry['id']}",
                           format_func=lambda i: f"{'ABCD'[i]}. {entry['options'][i]}")
        st.caption(f"Explanation: {entry['explanation']}")
        with st.container(horizontal=True):
            if st.button("Approve", icon=":material/check:", type="primary", key="quiz_approve"):
                apply_decision(entry, str(correct + 1))  # also records a corrected answer
                save_pending(items, pending_path)
                st.rerun()
            if st.button("Reject", icon=":material/close:", key="quiz_reject"):
                apply_decision(entry, "r")
                save_pending(items, pending_path)
                st.rerun()

st.subheader("Publish", anchor=False)
st.caption("Copies approved questions into the bank inside the index, so they ship with the next "
           "knowledge patch. Restart the tutor to serve them locally.")
if st.button("Publish approved questions", icon=":material/publish:",
             disabled=not any(q.get("status") in ("approved", "rejected") for q in items)):
    count = publish(items, bank_file)
    st.toast(f"The bank now has {count} questions", icon=":material/publish:")
    st.rerun()

published = load_bank(bank_file)
if published:
    coverage = Counter(q.get("source_title") or q["source"] for q in published)
    st.dataframe(pd.DataFrame(sorted(coverage.items()), columns=["source", "questions"]), hide_index=True,
                 alt="Published quiz questions per source document")
