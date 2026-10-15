import streamlit as st
import logging

logging.basicConfig(level=logging.INFO)

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Vet AI Tutor (Local)", page_icon="🐄", layout="centered")

st.title("🐄 Veterinary AI Tutor (Local)")
st.caption("Runs fully offline using Ollama + your local knowledge base")

# -----------------------------
# Load Embeddings & Vector Store
# -----------------------------
@st.cache_resource
def load_tutor():
    try:
        from langchain_ollama import OllamaLLM
        from vet_ai_tutor.core import Tutor
        from vet_ai_tutor.llm_adapter import OllamaAdapter
        from vet_ai_tutor.config import VECTOR_PATH, LLM_MODEL, OLLAMA_KEEP_ALIVE

        with st.spinner("Loading AI Tutor…"):
            base_llm = OllamaLLM(
                model=LLM_MODEL,
                num_predict=256,       # ~150 words — good speed/quality balance on CPU
                timeout=300,
                keep_alive=OLLAMA_KEEP_ALIVE,
                num_ctx=4096,          # must fit 5 chunks × 600 chars + prompt
            )
            adapter = OllamaAdapter(base_llm)
            tutor = Tutor(llm=adapter, vector_path=VECTOR_PATH)

        # Pre-warm: load model into Ollama memory so first real query is fast
        try:
            base_llm.invoke("Hi")
        except Exception:
            pass

        return tutor, None
    except Exception as e:
        return None, str(e)


tutor, load_error = load_tutor()

if load_error:
    st.error(f"Failed to initialize tutor: {load_error}")
    st.info("Make sure Ollama is running and the vectorstore has been built. "
            "Run: `python scripts/rebuild_vectorstore.py`")
    st.stop()

if tutor.db is None:
    st.error("Vectorstore not found or failed to load.")
    st.info("Build it first: `python scripts/rebuild_vectorstore.py`")
    st.stop()

# maintain chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "followups" not in st.session_state:
    st.session_state["followups"] = []
if "pending_question" not in st.session_state:
    st.session_state["pending_question"] = None

# Track docs for display
docs = []
retrieved_context = ""

# Reset button
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("🔄 Reset"):
        st.session_state["chat_history"] = []
        st.session_state["followups"] = []
        tutor.reset_history()
        st.rerun()

# -----------------------------
# User Input
# -----------------------------
question = None
typed_question = st.chat_input("Ask a veterinary question...")
if typed_question:
    question = typed_question
elif st.session_state["pending_question"]:
    question = st.session_state["pending_question"]
    st.session_state["pending_question"] = None

if question:
    st.session_state["followups"] = []  # clear old follow-ups
    # append user message immediately
    st.session_state["chat_history"].append(("You", question))

    # Display all previous messages first
    for who, text in st.session_state["chat_history"]:
        with st.chat_message("user" if who == "You" else "assistant"):
            st.markdown(text)

    # Stream the new response
    try:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            docs = []
            for token, token_docs in tutor.ask_stream(question):
                full_response += token
                docs = token_docs
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

        retrieved_context = "\n\n---\n\n".join(
            f"**{getattr(d, 'metadata', {}).get('source', 'unknown')}**\n{d.page_content}"
            for d in docs
        )
    except Exception as e:
        full_response = f"Error: {e}"
        retrieved_context = ""
        docs = []

    st.session_state["chat_history"].append(("Tutor", full_response))

    # Generate suggested follow-ups (runs in background while user reads)
    if full_response and not full_response.startswith("Error"):
        try:
            followups = tutor.suggest_followups(question, full_response, num=3)
            st.session_state["followups"] = followups
        except Exception:
            st.session_state["followups"] = []

else:
    # No new question — just display history
    for who, text in st.session_state["chat_history"]:
        with st.chat_message("user" if who == "You" else "assistant"):
            st.markdown(text)

# Show suggested follow-ups
if st.session_state.get("followups"):
    st.markdown("**💡 You might also ask:**")
    cols = st.columns(len(st.session_state["followups"]))
    for i, (col, fu) in enumerate(zip(cols, st.session_state["followups"])):
        with col:
            if st.button(fu, key=f"followup_{i}", use_container_width=True):
                st.session_state["pending_question"] = fu
                st.session_state["followups"] = []
                st.rerun()

# Show retrieved context if there are docs from the last question
if st.session_state["chat_history"] and docs:
    with st.expander("🔍 Retrieved Context"):
        st.markdown(retrieved_context)

st.markdown("---")
st.caption("Local • Private • No Internet Required")
