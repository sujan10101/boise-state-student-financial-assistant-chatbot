"""
Streamlit chat UI for the Boise State SFS RAG assistant.
"""
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="BSU Financial Services Assistant",
    page_icon="🎓",
    layout="centered",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 BSU SFS RAG Assistant")
    st.markdown(
        "Ask any question about Boise State University's **Student Financial Services** — "
        "payments, deadlines, late fees, payment plans, 529 accounts, and more."
    )
    st.divider()
    st.markdown("**Contact SFS directly:**")
    st.markdown("📧 sfinfo@boisestate.edu")
    st.markdown("📞 (208) 426-1212")
    st.markdown("🏢 Admin Building, Room 101")
    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🎓 Boise State SFS Assistant")

# Guard: check setup
chroma_ready = os.path.exists("chroma_db")
raw_ready = os.path.exists("data/raw_documents.json")

if not raw_ready:
    st.error("No crawled data found. Run `python crawler.py` first.", icon="⚠️")
    st.stop()

if not chroma_ready:
    st.warning("Vector store not found. Building it now…", icon="⏳")
    with st.spinner("Ingesting documents and building vector store…"):
        try:
            from ingest import load_raw_documents, split_documents, build_vectorstore
            docs = load_raw_documents()
            chunks = split_documents(docs)
            build_vectorstore(chunks)
            st.success("Vector store built! Refreshing…")
            st.rerun()
        except Exception as e:
            st.error(f"Ingestion failed: {e}", icon="❌")
            st.stop()

if not os.getenv("OPENAI_API_KEY"):
    st.stop()


# ── Load chain (cached per session) ──────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base…")
def get_chain():
    from rag import load_vectorstore, build_chain
    vs = load_vectorstore()
    return build_chain(vs)


chain_tuple = get_chain()

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm your **Boise State Student Financial Services** assistant. "
                "Ask me about tuition payments, deadlines, payment plans, late fees, "
                "529 accounts, refunds, or anything else related to SFS. How can I help?"
            ),
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"- [{src}]({src})")

# ── Example buttons (only shown on fresh start) ───────────────────────────────
EXAMPLES = [
    "What are the payment options?",
    "What is the late fee penalty?",
    "How does the payment plan work?",
    "What 529 plans are accepted?",
    "What happens if my check bounces?",
]

if len(st.session_state.messages) == 1:
    st.markdown("**Try asking:**")
    cols = st.columns(len(EXAMPLES))
    for col, q in zip(cols, EXAMPLES):
        if col.button(q, use_container_width=True):
            st.session_state._prefill = q
            st.rerun()

# Handle example-button prefill: inject as a user message and process below
prefill = st.session_state.pop("_prefill", None)
if prefill:
    st.session_state._pending_prompt = prefill
    st.rerun()

pending = st.session_state.pop("_pending_prompt", None)

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := (pending or st.chat_input("Ask about SFS payments, deadlines, fees…")):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base…"):
            from rag import ask
            result = ask(chain_tuple, prompt)

        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("📄 Sources", expanded=False):
                for src in result["sources"]:
                    st.markdown(f"- [{src}]({src})")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )
