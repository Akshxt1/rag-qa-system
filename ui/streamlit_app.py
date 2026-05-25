import sys
import os
import uuid
import math
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="DocMind — RAG Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #0d1117; color: #e6edf3; }
.main .block-container { padding-top: 1.5rem; max-width: 1200px; }
[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] * { font-size: 0.88rem; }
.hero-title { font-size: 2.2rem; font-weight: 700;
  background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 60%, #ff7b72 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.hero-subtitle { color: #8b949e; font-size: 0.95rem; margin-bottom: 1.2rem; }
.badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }
.badge-ready { background: #1a3a2a; color: #3fb950; border: 1px solid #3fb950; }
.badge-empty { background: #2d1a1a; color: #f85149; border: 1px solid #f85149; }
.chat-user { background: #1c2128; border: 1px solid #30363d;
  border-radius: 12px 12px 4px 12px; padding: 0.75rem 1rem; margin: 0.4rem 0; color: #e6edf3; }
.chat-assistant { background: #0f1923; border: 1px solid #21262d;
  border-left: 3px solid #58a6ff; border-radius: 4px 12px 12px 12px;
  padding: 0.75rem 1.1rem; margin: 0.4rem 0; color: #e6edf3; line-height: 1.65; }
.resp-time { font-size: 0.72rem; color: #8b949e; font-family: 'JetBrains Mono', monospace; margin-top: 0.3rem; }
.source-card { background: #161b22; border: 1px solid #30363d;
  border-radius: 8px; padding: 0.65rem 0.9rem; margin: 0.35rem 0; font-size: 0.82rem; }
.source-card .fname { color: #58a6ff; font-weight: 600; font-size: 0.78rem; }
.source-card .sc { float: right; font-weight: 600; font-size: 0.75rem;
  padding: 0.08rem 0.4rem; border-radius: 10px; }
.source-card .ex { color: #8b949e; margin-top: 0.35rem; font-size: 0.79rem; line-height: 1.5; }
.stat-box { background: #161b22; border: 1px solid #30363d;
  border-radius: 8px; padding: 0.7rem; text-align: center; }
.stat-num { font-size: 1.4rem; font-weight: 700; color: #58a6ff; }
.stat-label { font-size: 0.68rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.07em; }
.section-label { color: #8b949e; font-size: 0.7rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.08em; margin: 0.8rem 0 0.4rem 0; }
.stButton > button { background: #1f6feb; color: white; border: none;
  border-radius: 6px; font-weight: 600; font-size: 0.85rem; }
.stButton > button:hover { background: #388bfd; }
.stTextInput > div > div > input { background: #161b22 !important;
  border: 1px solid #30363d !important; border-radius: 8px !important; color: #e6edf3 !important; }
div[data-testid="stForm"] { border: none; padding: 0; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


# ── Pipeline ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    from app.pipeline.rag import RAGPipeline
    return RAGPipeline()


# ── State ─────────────────────────────────────────────────────────────────────
def init_state():
    if "chats" not in st.session_state:
        cid = str(uuid.uuid4())[:8]
        st.session_state.chats = {
            cid: {"name": "Chat 1", "messages": [], "created_at": datetime.now()}
        }
        st.session_state.active_chat = cid

    if "selected_docs" not in st.session_state:
        st.session_state.selected_docs = set()

    if "settings" not in st.session_state:
        st.session_state.settings = {
            "model":           "gemini-3.1-flash-lite",
            "temperature":     0.1,
            "max_tokens":      1024,
            "top_k":           5,
            "score_threshold": 0.0,
            "use_hybrid":      True,
            "use_reranking":   True,
        }


init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────
def active_messages():
    return st.session_state.chats[st.session_state.active_chat]["messages"]


def create_chat():
    cid = str(uuid.uuid4())[:8]
    n = len(st.session_state.chats) + 1
    st.session_state.chats[cid] = {
        "name": f"Chat {n}", "messages": [], "created_at": datetime.now()
    }
    st.session_state.active_chat = cid


def delete_chat(cid):
    if len(st.session_state.chats) == 1:
        create_chat()
    del st.session_state.chats[cid]
    if st.session_state.active_chat == cid:
        st.session_state.active_chat = list(st.session_state.chats.keys())[-1]



def score_color(pct: int) -> str:
    if pct >= 70:
        return "#3fb950"
    elif pct >= 45:
        return "#d29922"
    return "#8b949e"


# ── Load pipeline ─────────────────────────────────────────────────────────────
try:
    pipeline = load_pipeline()
    status_data = pipeline.status()
    is_ready = status_data["ready"]
    pipeline_ok = True
except Exception as e:
    pipeline = None
    status_data = {}
    is_ready = False
    pipeline_ok = False
    st.error(f"Pipeline error: {e}")

# Sync model if changed
if pipeline_ok:
    s = st.session_state.settings
    if s["model"] != status_data.get("generative_model"):
        pipeline.set_model(s["model"])


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧠 DocMind")

    if is_ready:
        st.markdown('<span class="badge badge-ready">● READY</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-empty">○ NO DOCS</span>',
                    unsafe_allow_html=True)

    # ── CHATS ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">💬 Chats</div>',
                unsafe_allow_html=True)

    if st.button("＋ New Chat", use_container_width=True, key="new_chat_btn"):
        create_chat()
        st.rerun()

    for cid, chat in list(st.session_state.chats.items()):
        is_active = cid == st.session_state.active_chat
        msg_count = len(chat["messages"]) // 2
        label = f"{'▶ ' if is_active else ''}{chat['name']} ({msg_count})"
        col_name, col_del = st.columns([5, 1])
        with col_name:
            if st.button(label, key=f"chat_{cid}", use_container_width=True):
                st.session_state.active_chat = cid
                st.rerun()
        with col_del:
            if st.button("🗑", key=f"del_chat_{cid}"):
                delete_chat(cid)
                st.rerun()

    # ── DOCUMENTS ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">📁 Documents</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        if st.button("⚡ Ingest", use_container_width=True):
            with st.spinner("Ingesting..."):
                total_docs, total_chunks, errors = 0, 0, []
                prog = st.progress(0)
                for i, f in enumerate(uploaded):
                    prog.progress((i + 1) / len(uploaded))
                    try:
                        r = pipeline.ingest_from_bytes(f.read(), f.name)
                        if r.success:
                            total_docs += r.documents_loaded
                            total_chunks += r.chunks_created
                        else:
                            errors.extend(r.errors)
                    except Exception as e:
                        errors.append(f"{f.name}: {e}")
                prog.progress(1.0)
            if total_docs > 0:
                st.success(f"✅ {total_docs} doc(s) → {total_chunks} chunks")
                st.rerun()
            for err in errors:
                st.error(err)

    # Indexed documents with checkboxes + delete
    if is_ready and status_data.get("sources"):
        st.markdown("**Indexed — select to filter:**")
        for src in status_data["sources"]:
            ext = Path(src).suffix.lstrip(".").lower()
            icon = {"pdf": "📕", "docx": "📘", "txt": "📄", "md": "📝"}.get(ext, "📄")
            col_chk, col_name, col_del = st.columns([1, 5, 1])
            with col_chk:
                checked = st.checkbox(
                    "", key=f"doc_{src}",
                    value=src in st.session_state.selected_docs,
                    label_visibility="collapsed",
                )
                if checked:
                    st.session_state.selected_docs.add(src)
                else:
                    st.session_state.selected_docs.discard(src)
            with col_name:
                st.markdown(
                    f'<div style="font-size:0.78rem;color:#8b949e;padding-top:0.4rem;">'
                    f'{icon} {src[:22]}{"…" if len(src) > 22 else ""}</div>',
                    unsafe_allow_html=True,
                )
            with col_del:
                if st.button("✕", key=f"del_doc_{src}"):
                    with st.spinner(f"Removing {src}..."):
                        removed = pipeline.delete_document(src)
                    st.session_state.selected_docs.discard(src)
                    st.success(f"Removed {removed} chunks")
                    st.rerun()

        if st.session_state.selected_docs:
            st.markdown(
                f'<div style="font-size:0.75rem;color:#58a6ff;">🔍 Filtering: '
                f'{len(st.session_state.selected_docs)} doc(s)</div>',
                unsafe_allow_html=True,
            )

    # ── SETTINGS ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">⚙️ Settings</div>',
                unsafe_allow_html=True)

    with st.expander("Model & Generation", expanded=False):
        s = st.session_state.settings

        new_model = st.selectbox(
            "Gemini Model",
            options=[
                "gemini-3.1-flash-lite",
                "gemini-2.5-flash-lite",
                "gemini-2.5-flash",
                "gemini-2.5-pro",
            ],
            index=["gemini-3.1-flash-lite", "gemini-2.5-flash-lite",
                   "gemini-2.5-flash", "gemini-2.5-pro"].index(
                s.get("model", "gemini-3.1-flash-lite")
            ),
            help="Faster models respond quicker; Pro gives best quality",
        )
        if new_model != s.get("model"):
            s["model"] = new_model
            if pipeline_ok:
                pipeline.set_model(new_model)

        s["temperature"] = st.slider(
            "Temperature", 0.0, 1.0, s.get("temperature", 0.1), 0.05,
            help="Lower = more factual. Higher = more creative",
        )
        s["max_tokens"] = st.slider(
            "Max Response Length", 256, 2048, s.get("max_tokens", 1024), 128,
            help="Maximum tokens in the response",
        )

    with st.expander("Retrieval", expanded=False):
        s = st.session_state.settings

        s["top_k"] = st.slider(
            "Top-K Chunks", 1, 15, s.get("top_k", 5),
            help="Final chunks passed to Gemini after reranking",
        )
        s["score_threshold"] = st.slider(
            "Min Relevance Score", 0.0, 1.0, s.get("score_threshold", 0.0), 0.05,
            help="Filter out chunks below this score",
        )
        s["use_hybrid"] = st.toggle(
            "🔀 Hybrid Search (BM25 + Vector)",
            value=s.get("use_hybrid", True),
            help="Combines keyword + semantic search for better recall",
        )
        s["use_reranking"] = st.toggle(
            "🎯 CrossEncoder Reranking",
            value=s.get("use_reranking", True),
            help="Reranks retrieved chunks for higher precision",
        )
        st.markdown(
            f'<div style="font-size:0.75rem;color:#8b949e;margin-top:0.4rem;">'
            f'Embedding: <code>{status_data.get("embedding_model", "—").split("/")[-1]}</code><br>'
            f'Reranker: <code>{status_data.get("reranker_model", "—").split("/")[-1]}</code><br>'
            f'Candidates: <code>{status_data.get("candidate_k", 30)}</code>'
            f' → reranked to <code>{s.get("top_k", 5)}</code>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="hero-title">DocMind</div>'
    '<div class="hero-subtitle">Ask anything about your documents. '
    'Answers grounded in your data.</div>',
    unsafe_allow_html=True,
)

# Stats strip
if pipeline_ok and is_ready:
    c1, c2, c3, c4 = st.columns(4)
    for col, num, label in [
        (c1, status_data["unique_documents"], "Documents"),
        (c2, status_data["total_chunks"], "Chunks"),
        (c3, len(active_messages()) // 2, "Queries"),
        (c4, status_data["embedding_dim"], "Vector Dim"),
    ]:
        with col:
            st.markdown(
                f'<div class="stat-box">'
                f'<div class="stat-num">{num}</div>'
                f'<div class="stat-label">{label}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown("<br>", unsafe_allow_html=True)

# Chat title + rename
current_chat_name = st.session_state.chats[st.session_state.active_chat]["name"]
col_title, col_rename = st.columns([4, 1])
with col_title:
    st.markdown(
        f'<div style="font-size:1rem;font-weight:600;color:#58a6ff;margin-bottom:0.5rem;">'
        f'💬 {current_chat_name}</div>',
        unsafe_allow_html=True,
    )
with col_rename:
    new_name = st.text_input(
        "Rename", value=current_chat_name,
        label_visibility="collapsed", key="rename_input",
    )
    if new_name and new_name != current_chat_name:
        st.session_state.chats[st.session_state.active_chat]["name"] = new_name
        st.rerun()

# Chat + Sources layout
chat_col, src_col = st.columns([3, 2])

with chat_col:
    msgs = active_messages()
    if not msgs:
        if is_ready:
            st.markdown(
                '<div style="color:#8b949e;font-size:0.9rem;padding:1rem 0;">'
                '💬 Documents ready. Ask a question below.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#161b22;border:1px dashed #30363d;'
                'border-radius:10px;padding:2rem;text-align:center;color:#8b949e;">'
                '<div style="font-size:2rem;">📂</div>'
                '<div style="font-weight:600;color:#e6edf3;margin-bottom:0.3rem;">'
                'No documents yet</div>'
                '<div style="font-size:0.85rem;">Upload files in the sidebar.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
    else:
        for msg in msgs:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">🧑 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-assistant">🤖 {msg["content"]}'
                    f'<div class="resp-time">'
                    f'⏱ {msg.get("elapsed", 0):.2f}s · '
                    f'{msg.get("chunks", 0)} chunks · '
                    f'{msg.get("method", "vector")} · '
                    f'temp={msg.get("temperature", 0.1)}'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

with src_col:
    last_sources = None
    for msg in reversed(active_messages()):
        if msg.get("sources"):
            last_sources = msg["sources"]
            break

    if last_sources:
        st.markdown(
            '<div style="font-size:0.75rem;color:#8b949e;font-weight:600;'
            'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.4rem;">'
            '📎 Sources</div>',
            unsafe_allow_html=True,
        )
        # Min-max normalize scores relative to this result set
        raw_scores = [src["relevance_score"] for src in last_sources]
        min_s = min(raw_scores)
        max_s = max(raw_scores)
        score_range = max_s - min_s

        for src in last_sources:
            raw = src["relevance_score"]
            pct = int((raw - min_s) / score_range * 100) if score_range > 0 else 50
            clr = score_color(pct)
            st.markdown(
                f'<div class="source-card">'
                f'<span class="fname">📄 {src["filename"]}</span>'
                f'<span class="sc" style="color:{clr};">{pct}%</span>'
                f'<div class="ex">{src["excerpt"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

if st.session_state.selected_docs:
    docs_str = ", ".join(list(st.session_state.selected_docs)[:2])
    if len(st.session_state.selected_docs) > 2:
        docs_str += f" +{len(st.session_state.selected_docs) - 2} more"
    st.markdown(
        f'<div style="font-size:0.78rem;color:#58a6ff;margin-bottom:0.3rem;">'
        f'🔍 Querying: {docs_str}</div>',
        unsafe_allow_html=True,
    )

with st.form(key="query_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        question = st.text_input(
            "Ask",
            placeholder="What are the key findings? Compare X and Y...",
            label_visibility="collapsed",
        )
    with col_btn:
        send = st.form_submit_button("Ask →", use_container_width=True)

# ── Handle query ──────────────────────────────────────────────────────────────
if send and question.strip():
    if not pipeline_ok or not is_ready:
        st.warning("⚠️ Please upload and ingest documents first.")
    else:
        s = st.session_state.settings
        filter_srcs = st.session_state.selected_docs if st.session_state.selected_docs else None

        active_messages().append({"role": "user", "content": question})

        with st.spinner("Thinking..."):
            result = pipeline.query(
                question,
                top_k=s["top_k"],
                filter_sources=filter_srcs,
                temperature=s["temperature"],
                max_tokens=s["max_tokens"],
                score_threshold=s["score_threshold"],
                use_hybrid=s.get("use_hybrid", True),
                use_reranking=s.get("use_reranking", True),
            )

        active_messages().append({
            "role": "assistant",
            "content": result.answer,
            "sources": result.sources,
            "elapsed": result.elapsed_seconds,
            "chunks": result.chunks_retrieved,
            "temperature": s["temperature"],
            "method": result.retrieval_method,
        })
        st.rerun()

# Footer
st.markdown(
    '<div style="text-align:center;color:#484f58;font-size:0.72rem;margin-top:2rem;">'
    'DocMind · HuggingFace Embeddings + Google Gemini · Built with Streamlit'
    '</div>',
    unsafe_allow_html=True,
)