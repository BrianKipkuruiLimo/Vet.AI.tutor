import pandas as pd
import streamlit as st

from vet_ai_tutor.config import RETRIEVER_K
from vet_ai_tutor.devtools import status, ui
from vet_ai_tutor.devtools.paths import get_paths

paths = get_paths()

qa = status.qa_stats(paths.qa_pairs)
with st.container(horizontal=True):
    st.metric("Evaluation questions", qa["questions"], border=True)
    st.metric("With a reviewed source", qa["with_source"], border=True,
              help="Items matched by source file are scored reliably; others fall back to answer words")
    st.metric("Retrieval missed", qa["retrieval_missed"], border=True,
              help="Reviewer found no correct passage among the candidates")
if qa["questions"] and qa["with_source"] < qa["questions"]:
    st.info("Attach sources so scores are reliable. Someone who knows the material runs:",
            title=f"{qa['questions'] - qa['with_source']} questions have no source yet")
    st.code("python scripts/review_qa_sources.py", language="bash")

left, right = st.columns(2)
with left, st.container(border=True):
    st.subheader("Retrieval", anchor=False)
    st.caption("Fast: no LLM calls. Recall@k and MRR of the hybrid search.")
    with st.form("retrieval_form", border=False):
        with st.container(horizontal=True):
            k = st.number_input("k", 1, 20, RETRIEVER_K)
            mode = st.selectbox("Search", ["hybrid", "vector"])
            limit = st.number_input("Questions (0 = all)", 0, 1000, 0)
        if st.form_submit_button("Evaluate retrieval", icon=":material/search_check:"):
            ui.start_job("evaluate_retrieval", ["--k", str(k), "--mode", mode]
                         + (["--limit", str(limit)] if limit else []))
    ui.job_panel("evaluate_retrieval")
with right, st.container(border=True):
    st.subheader("Answers", anchor=False)
    st.caption("Slow: about 30–45 s per question on the laptop. RAGAS adds ~2.5 min per question.")
    with st.form("benchmark_form", border=False):
        with st.container(horizontal=True):
            b_limit = st.number_input("Questions (0 = all)", 0, 1000, 10)
            ragas = st.toggle("Add RAGAS scores")
        judge = st.text_input("RAGAS judge model", "qwen3:4b", disabled=False)
        if st.form_submit_button("Run benchmark", icon=":material/speed:"):
            ui.start_job("benchmark", (["--limit", str(b_limit)] if b_limit else [])
                         + (["--ragas", "--judge-model", judge] if ragas else []))
    ui.job_panel("benchmark")

st.subheader("Retrieval over time", anchor=False)
retrieval = pd.DataFrame(status.retrieval_history(paths.eval_dir))
if retrieval.empty:
    st.caption("No retrieval evaluations yet.")
else:
    st.line_chart(retrieval, x="time", y=["recall_at_k", "mrr"], y_label="score",
                  alt="Retrieval recall at k and mean reciprocal rank over time")
    st.dataframe(retrieval.sort_values("time", ascending=False), hide_index=True,
                 alt="Retrieval evaluation runs",
                 column_config={"recall_at_k": st.column_config.NumberColumn("Recall@k", format="percent"),
                                "mrr": st.column_config.NumberColumn("MRR", format="%.3f"),
                                "time": st.column_config.DatetimeColumn("Run", format="YYYY-MM-DD HH:mm")})

st.subheader("Answer quality over time", anchor=False)
answers = pd.DataFrame(status.benchmark_history(paths.eval_dir))
if answers.empty:
    st.caption("No benchmark runs yet.")
else:
    quality = [c for c in ("grounding", "key_term_recall", "citation_rate", "faithfulness") if answers[c].notna().any()]
    if quality:
        st.line_chart(answers, x="time", y=quality, y_label="share",
                      alt="Grounding, key-term recall, citation rate and faithfulness over time")
    st.dataframe(answers.sort_values("time", ascending=False), hide_index=True, alt="Benchmark runs",
                 column_config={c: st.column_config.NumberColumn(format="percent") for c in
                                ("grounding", "key_term_recall", "citation_rate", "not_found_rate",
                                 "faithfulness", "context_recall")}
                 | {"time": st.column_config.DatetimeColumn("Run", format="YYYY-MM-DD HH:mm"),
                    "avg_seconds": st.column_config.NumberColumn("Avg s / question", format="%.1f")})
