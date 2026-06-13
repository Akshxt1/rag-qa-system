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
.main .block-container { padding-top: 1rem; padding-bottom: 0.5rem; max-width: 1200px; }
[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] * { font-size: 0.88rem; }

.hero-wrap { display:flex; align-items:center; gap:1rem; margin-bottom:0.6rem; }
.hero-title { font-size: 1.8rem; font-weight: 700; margin:0;
  background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 60%, #ff7b72 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.hero-tag { background:#1c2845; color:#58a6ff; font-size:0.7rem; font-weight:600;
  padding:0.18rem 0.6rem; border-radius:20px; border:1px solid #58a6ff;
  letter-spacing:0.05em; white-space:nowrap; }
.hero-sub { color:#8b949e; font-size:0.82rem; margin-bottom:0.8rem; }

.badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }
.badge-ready { background: #1a3a2a; color: #3fb950; border: 1px solid #3fb950; }
.badge-empty { background: #2d1a1a; color: #f85149; border: 1px solid #f85149; }

.chat-user { background: #1c2128; border: 1px solid #30363d;
  border-radius: 12px 12px 4px 12px; padding: 0.65rem 0.9rem; margin: 0.35rem 0; color: #e6edf3; font-size:0.9rem; }
.chat-assistant { background: #0f1923; border: 1px solid #21262d;
  border-left: 3px solid #58a6ff; border-radius: 4px 12px 12px 12px;
  padding: 0.65rem 1rem; margin: 0.35rem 0; color: #e6edf3; line-height: 1.65; font-size:0.9rem; }
.resp-time { font-size: 0.68rem; color: #8b949e; font-family: 'JetBrains Mono', monospace; margin-top: 0.25rem; }

.source-card { background: #161b22; border: 1px solid #30363d;
  border-radius: 8px; padding: 0.6rem 0.85rem; margin: 0.3rem 0; font-size: 0.8rem; }
.source-card .fname { color: #58a6ff; font-weight: 600; font-size: 0.76rem; }
.source-card .sc { float: right; font-weight: 600; font-size: 0.73rem; padding: 0.06rem 0.35rem; border-radius: 10px; }
.source-card .ex { color: #8b949e; margin-top: 0.3rem; font-size: 0.77rem; line-height: 1.45; }

.stat-box { background: #161b22; border: 1px solid #30363d;
  border-radius: 8px; padding: 0.5rem 0.4rem; text-align: center; }
.stat-num { font-size: 1.2rem; font-weight: 700; color: #58a6ff; }
.stat-label { font-size: 0.62rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.07em; }

.section-label { color: #8b949e; font-size: 0.7rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.08em; margin: 0.8rem 0 0.4rem 0; }

.stButton > button { background: #1f6feb; color: white; border: none;
  border-radius: 6px; font-weight: 600; font-size: 0.85rem; }
.stButton > button:hover { background: #388bfd; }
.stTextInput > div > div > input { background: #161b22 !important;
  border: 1px solid #30363d !important; border-radius: 8px !important; color: #e6edf3 !important; }
div[data-testid="stForm"] { border: none; padding: 0; }

.tech-card { background: #0d1117; border: 1px solid #21262d; border-radius: 8px;
  padding: 0.55rem 0.75rem; display:flex; justify-content:space-between; align-items:center; }
.tech-name { color: #e6edf3; font-size: 0.8rem; font-weight: 600; }
.tech-sub { color: #8b949e; font-size: 0.7rem; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_pipeline():
    from app.pipeline.rag import RAGPipeline
    return RAGPipeline()


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
            "model": "gemini-2.5-flash",
            "temperature": 0.1,
            "max_tokens": 1024,
            "top_k": 5,
            "score_threshold": 0.0,
            "use_hybrid": True,
            "use_reranking": True,
        }


init_state()


def active_messages():
    return st.session_state.chats[st.session_state.active_chat]["messages"]


def create_chat():
    cid = str(uuid.uuid4())[:8]
    n = len(st.session_state.chats) + 1
    st.session_state.chats[cid] = {"name": f"Chat {n}", "messages": [], "created_at": datetime.now()}
    st.session_state.active_chat = cid


def delete_chat(cid):
    if len(st.session_state.chats) == 1:
        create_chat()
    del st.session_state.chats[cid]
    if st.session_state.active_chat == cid:
        st.session_state.active_chat = list(st.session_state.chats.keys())[-1]


def score_color(pct):
    return "#3fb950" if pct >= 70 else "#d29922" if pct >= 45 else "#8b949e"


# ── Pipeline ──────────────────────────────────────────────────────────────────
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
        st.markdown('<span class="badge badge-ready">● READY</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-empty">○ NO DOCS</span>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">💬 Chats</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="section-label">📁 Documents</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload", type=["pdf", "docx", "txt", "md"],
                                accept_multiple_files=True, label_visibility="collapsed")
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

    if is_ready and status_data.get("sources"):
        st.markdown("**Indexed — select to filter:**")
        for src in status_data["sources"]:
            ext = Path(src).suffix.lstrip(".").lower()
            icon = {"pdf": "📕", "docx": "📘", "txt": "📄", "md": "📝"}.get(ext, "📄")
            col_chk, col_name, col_del = st.columns([1, 5, 1])
            with col_chk:
                checked = st.checkbox("", key=f"doc_{src}",
                                      value=src in st.session_state.selected_docs,
                                      label_visibility="collapsed")
                if checked:
                    st.session_state.selected_docs.add(src)
                else:
                    st.session_state.selected_docs.discard(src)
            with col_name:
                st.markdown(
                    f'<div style="font-size:0.78rem;color:#8b949e;padding-top:0.4rem;">'
                    f'{icon} {src[:22]}{"…" if len(src) > 22 else ""}</div>',
                    unsafe_allow_html=True)
            with col_del:
                if st.button("✕", key=f"del_doc_{src}"):
                    with st.spinner(f"Removing..."):
                        removed = pipeline.delete_document(src)
                    st.session_state.selected_docs.discard(src)
                    st.success(f"Removed {removed} chunks")
                    st.rerun()
        if st.session_state.selected_docs:
            st.markdown(
                f'<div style="font-size:0.75rem;color:#58a6ff;">🔍 Filtering: '
                f'{len(st.session_state.selected_docs)} doc(s)</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-label">⚙️ Settings</div>', unsafe_allow_html=True)
    with st.expander("Model & Generation", expanded=False):
        s = st.session_state.settings
        models = ["gemini-2.5-flash", "gemini-2.5-flash-lite-preview-06-17",
                  "gemini-2.5-pro", "gemini-3.1-flash-lite"]
        new_model = st.selectbox("Gemini Model", options=models,
                                 index=models.index(s.get("model", "gemini-2.5-flash"))
                                 if s.get("model", "gemini-2.5-flash") in models else 0)
        if new_model != s.get("model"):
            s["model"] = new_model
            if pipeline_ok:
                pipeline.set_model(new_model)
        s["temperature"] = st.slider("Temperature", 0.0, 1.0, s.get("temperature", 0.1), 0.05)
        s["max_tokens"] = st.slider("Max Response Length", 256, 2048, s.get("max_tokens", 1024), 128)

    with st.expander("Retrieval", expanded=False):
        s = st.session_state.settings
        s["top_k"] = st.slider("Top-K Chunks", 1, 15, s.get("top_k", 5))
        s["score_threshold"] = st.slider("Min Relevance Score", 0.0, 1.0, s.get("score_threshold", 0.0), 0.05)
        s["use_hybrid"] = st.toggle("🔀 Hybrid Search", value=s.get("use_hybrid", True))
        s["use_reranking"] = st.toggle("🎯 CrossEncoder Reranking", value=s.get("use_reranking", True))
        st.markdown(
            f'<div style="font-size:0.73rem;color:#8b949e;margin-top:0.3rem;">'
            f'Embed: <code>{status_data.get("embedding_model","—").split("/")[-1]}</code> · '
            f'Candidates: <code>{status_data.get("candidate_k",30)}</code>→<code>{s.get("top_k",5)}</code>'
            f'</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

# ── Compact hero ──────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero-wrap">'
    '<span class="hero-title">DocMind</span>'
    '<span class="hero-tag">RAG-POWERED</span>'
    '<span style="color:#8b949e;font-size:0.82rem;">Document Intelligence System</span>'
    '</div>'
    '<div class="hero-sub">'
    'Upload docs → Ask in plain English → Get cited answers · '
    '<span style="color:#58a6ff;">HuggingFace</span> + '
    '<span style="color:#bc8cff;">Gemini</span>'
    '</div>',
    unsafe_allow_html=True)

# ── Stats strip (only when ready) ─────────────────────────────────────────────
if pipeline_ok and is_ready:
    c1, c2, c3, c4 = st.columns(4)
    for col, num, label in [
        (c1, status_data["unique_documents"], "Documents"),
        (c2, status_data["total_chunks"], "Chunks"),
        (c3, len(active_messages()) // 2, "Queries"),
        (c4, status_data["embedding_dim"], "Dim"),
    ]:
        with col:
            st.markdown(
                f'<div class="stat-box"><div class="stat-num">{num}</div>'
                f'<div class="stat-label">{label}</div></div>',
                unsafe_allow_html=True)

st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

# ── Chat name ─────────────────────────────────────────────────────────────────
current_chat_name = st.session_state.chats[st.session_state.active_chat]["name"]
st.markdown(
    f'<div style="font-size:0.88rem;font-weight:600;color:#58a6ff;margin-bottom:0.5rem;">'
    f'💬 {current_chat_name}</div>',
    unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
chat_col, src_col = st.columns([3, 2])

with chat_col:
    msgs = active_messages()
    if not msgs:
        if is_ready:
            st.markdown(
                '<div style="color:#8b949e;font-size:0.88rem;padding:0.5rem 0;">'
                '💬 Documents ready — ask a question below.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown("""
<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:1.2rem 1.5rem;">
  <div style="font-size:1.05rem;font-weight:700;color:#e6edf3;margin-bottom:0.6rem;">👋 Welcome to DocMind!</div>
  <div style="color:#8b949e;font-size:0.83rem;line-height:1.65;margin-bottom:0.9rem;">
    An AI-powered Q&amp;A system — upload documents, ask questions, get cited answers.
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;margin-bottom:0.7rem;">
    <div style="background:#0d1117;border:1px solid #21262d;border-radius:7px;padding:0.75rem;">
      <div style="color:#58a6ff;font-weight:600;font-size:0.75rem;margin-bottom:0.4rem;">📥 SUPPORTED FILES</div>
      <div style="color:#8b949e;font-size:0.77rem;line-height:1.7;">
        📕 PDF — papers, reports, books<br>
        📘 DOCX — word documents<br>
        📄 TXT / MD — plain text
      </div>
    </div>
    <div style="background:#0d1117;border:1px solid #21262d;border-radius:7px;padding:0.75rem;">
      <div style="color:#bc8cff;font-weight:600;font-size:0.75rem;margin-bottom:0.4rem;">💬 EXAMPLE QUESTIONS</div>
      <div style="color:#8b949e;font-size:0.77rem;line-height:1.7;">
        → What is this document about?<br>
        → Summarize chapter 3<br>
        → Compare X and Y
      </div>
    </div>
  </div>
  <div style="background:#0d1117;border:1px solid #21262d;border-radius:7px;padding:0.7rem;margin-bottom:0.7rem;">
    <div style="color:#3fb950;font-weight:600;font-size:0.75rem;margin-bottom:0.4rem;">⚙️ HOW IT WORKS</div>
    <div style="display:flex;gap:0.4rem;flex-wrap:wrap;align-items:center;">
      <span style="background:#1a3a2a;color:#3fb950;padding:0.2rem 0.55rem;border-radius:20px;font-size:0.72rem;">1. Upload docs</span>
      <span style="color:#484f58;font-size:0.75rem;">→</span>
      <span style="background:#1a3a2a;color:#3fb950;padding:0.2rem 0.55rem;border-radius:20px;font-size:0.72rem;">2. Click Ingest</span>
      <span style="color:#484f58;font-size:0.75rem;">→</span>
      <span style="background:#1a3a2a;color:#3fb950;padding:0.2rem 0.55rem;border-radius:20px;font-size:0.72rem;">3. Ask anything</span>
      <span style="color:#484f58;font-size:0.75rem;">→</span>
      <span style="background:#1a3a2a;color:#3fb950;padding:0.2rem 0.55rem;border-radius:20px;font-size:0.72rem;">4. Get cited answers</span>
    </div>
  </div>
  <div style="background:#1c2845;border:1px solid #58a6ff;border-radius:7px;padding:0.6rem 0.9rem;">
    <span style="color:#58a6ff;font-size:0.79rem;">🚀 <b>Get started:</b> Upload a file in the <b>Documents</b> section on the left, then click <b>⚡ Ingest</b></span>
  </div>
</div>""", unsafe_allow_html=True)
    else:
        for msg in msgs:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">🧑 {msg["content"]}</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="chat-assistant">🤖 {msg["content"]}'
                    f'<div class="resp-time">⏱ {msg.get("elapsed",0):.2f}s · '
                    f'{msg.get("chunks",0)} chunks · {msg.get("method","vector")} · '
                    f'temp={msg.get("temperature",0.1)}</div></div>',
                    unsafe_allow_html=True)

with src_col:
    msgs = active_messages()
    if not msgs and not is_ready:
        # Tech stack panel — same height as welcome card
        st.markdown("""
<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:1.2rem 1.3rem;">
  <div style="color:#58a6ff;font-weight:600;font-size:0.75rem;margin-bottom:0.75rem;letter-spacing:0.06em;">🧠 POWERED BY</div>
  <div style="display:flex;flex-direction:column;gap:0.45rem;">
    <div class="tech-card">
      <div><div class="tech-name">HuggingFace Embeddings</div>
      <div class="tech-sub">all-MiniLM-L6-v2 · 384-dim</div></div>
      <span style="color:#58a6ff;font-size:0.7rem;">embed</span>
    </div>
    <div class="tech-card">
      <div><div class="tech-name">FAISS Vector Store</div>
      <div class="tech-sub">IndexFlatIP · cosine sim</div></div>
      <span style="color:#bc8cff;font-size:0.7rem;">store</span>
    </div>
    <div class="tech-card">
      <div><div class="tech-name">BM25 + CrossEncoder</div>
      <div class="tech-sub">Hybrid search · reranking</div></div>
      <span style="color:#3fb950;font-size:0.7rem;">rank</span>
    </div>
    <div class="tech-card">
      <div><div class="tech-name">Google Gemini</div>
      <div class="tech-sub">Context-grounded generation</div></div>
      <span style="color:#ff7b72;font-size:0.7rem;">generate</span>
    </div>
  </div>
  <div style="margin-top:0.75rem;padding-top:0.65rem;border-top:1px solid #21262d;
    display:flex;justify-content:space-around;">
    <span style="color:#8b949e;font-size:0.7rem;">⚡ &lt;5s</span>
    <span style="color:#8b949e;font-size:0.7rem;">🎯 Citations</span>
    <span style="color:#8b949e;font-size:0.7rem;">🔒 Grounded</span>
  </div>
</div>""", unsafe_allow_html=True)

    elif msgs:
        last_sources = None
        for msg in reversed(msgs):
            if msg.get("sources"):
                last_sources = msg["sources"]
                break
        if last_sources:
            st.markdown(
                '<div style="font-size:0.73rem;color:#8b949e;font-weight:600;'
                'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.35rem;">'
                '📎 Sources</div>',
                unsafe_allow_html=True)
            raw_scores = [s["relevance_score"] for s in last_sources]
            min_s, max_s = min(raw_scores), max(raw_scores)
            score_range = max_s - min_s
            for src in last_sources:
                raw = src["relevance_score"]
                pct = int((raw - min_s) / score_range * 100) if score_range > 0 else 50
                clr = score_color(pct)
                st.markdown(
                    f'<div class="source-card">'
                    f'<span class="fname">📄 {src["filename"]}</span>'
                    f'<span class="sc" style="color:{clr};">{pct}%</span>'
                    f'<div class="ex">{src["excerpt"]}</div></div>',
                    unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
if st.session_state.selected_docs:
    docs_str = ", ".join(list(st.session_state.selected_docs)[:2])
    if len(st.session_state.selected_docs) > 2:
        docs_str += f" +{len(st.session_state.selected_docs)-2} more"
    st.markdown(
        f'<div style="font-size:0.76rem;color:#58a6ff;margin-bottom:0.2rem;">'
        f'🔍 Querying: {docs_str}</div>',
        unsafe_allow_html=True)

with st.form(key="query_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        question = st.text_input("Ask", placeholder="Ask anything about your documents...",
                                 label_visibility="collapsed")
    with col_btn:
        send = st.form_submit_button("Ask →", use_container_width=True)

if send and question.strip():
    if not pipeline_ok or not is_ready:
        st.warning("⚠️ Please upload and ingest documents first.")
    else:
        s = st.session_state.settings
        filter_srcs = st.session_state.selected_docs if st.session_state.selected_docs else None
        active_messages().append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            result = pipeline.query(
                question, top_k=s["top_k"], filter_sources=filter_srcs,
                temperature=s["temperature"], max_tokens=s["max_tokens"],
                score_threshold=s["score_threshold"],
                use_hybrid=s.get("use_hybrid", True),
                use_reranking=s.get("use_reranking", True),
            )
        active_messages().append({
            "role": "assistant", "content": result.answer,
            "sources": result.sources, "elapsed": result.elapsed_seconds,
            "chunks": result.chunks_retrieved, "temperature": s["temperature"],
            "method": result.retrieval_method,
        })
        st.rerun()

st.markdown(
    '<div style="text-align:center;color:#484f58;font-size:0.68rem;margin-top:0.5rem;">'
    'DocMind · HuggingFace + FAISS + BM25 + CrossEncoder + Google Gemini · Streamlit'
    '</div>', unsafe_allow_html=True)