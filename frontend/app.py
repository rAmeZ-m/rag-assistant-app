import html
import re

import streamlit as st

import api_client

st.set_page_config(page_title="RAG Document Assistant", page_icon="📄", layout="centered")

st.markdown(
    """
    <style>
    .src-card {
        border: 1px solid rgba(128, 128, 128, 0.35);
        border-left: 4px solid #ff9f1c;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 6px 0;
        font-size: 0.92rem;
    }
    .src-page { opacity: 0.7; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLES = [
    "What is meta-reinforcement learning?",
    "What is the Q-learning update equation?",
    "What is a transformer architecture?",
    "How does emotion influence learning in RL agents?",
]
AVATARS = {"user": "🙋", "assistant": "📄"}
SRC_RE = re.compile(r"^\[(\d+)\]\s*(.*?)\s*\(page (\d+)\)$")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

with st.sidebar:
    st.header("📄 Document Assistant")
    info = api_client.health()
    if info:
        st.success("Backend connected")
        st.write(f"Chunks indexed: **{info['chunks']}**")
        st.write(f"Model: **{info['llm_model']}**")
    else:
        st.error("Backend not reachable")

    st.subheader("Try an example")
    for i, ex in enumerate(EXAMPLES):
        if st.button(ex, key=f"ex_{i}"):
            st.session_state.pending = ex

    st.subheader("Tips")
    st.caption("Ask complete questions. Very short ones may not match any passage.")
    st.caption("Each question is answered on its own; the assistant does not remember earlier messages.")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


def source_card(s: str) -> str:
    m = SRC_RE.match(s)
    if not m:
        return f'<div class="src-card">{html.escape(s)}</div>'
    n, title, page = m.groups()
    return (
        f'<div class="src-card"><b>[{n}]</b> {html.escape(title)} '
        f'<span class="src-page">· page {page}</span></div>'
    )


def render(msg: dict) -> None:
    st.markdown(msg["content"])
    if msg.get("sources"):
        st.markdown("**Sources**")
        st.markdown("".join(source_card(s) for s in msg["sources"]), unsafe_allow_html=True)
    elif msg.get("refused"):
        st.caption("No matching passage was found in the documents.")


st.title("📄 RAG Document Assistant")
st.caption("Ask questions about the AI research papers. Answers are grounded in the documents and cite their sources.")

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar=AVATARS[m["role"]]):
        render(m)

typed = st.chat_input("Ask a question about the documents...")
question = typed or st.session_state.pending
st.session_state.pending = None

if not st.session_state.messages and not question:
    st.info("👋 Ask a question about the papers, or pick an example from the sidebar.")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            with st.spinner("Searching the documents..."):
                result = api_client.ask(question)
            msg = {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "refused": not result["sources"],
            }
        except api_client.ApiError as e:
            msg = {"role": "assistant", "content": f"⚠️ {e}", "sources": [], "refused": False}
        render(msg)
    st.session_state.messages.append(msg)