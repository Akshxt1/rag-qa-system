import sys
import os
import uuid
import math
import base64
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st

# ── Logo loader ───────────────────────────────────────────────────────────────
def get_logo_b64():
    logo_path = Path(__file__).parent / "assets" / "logo.png"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

LOGO_B64 = get_logo_b64()
LOGO_HTML = (
    f'<img src="data:image/png;base64,{LOGO_B64}" '
    f'style="width:36px;height:36px;object-fit:contain;border-radius:8px;" />'
    if LOGO_B64 else '<span style="font-size:1.8rem;">🧠</span>'
)

st.set_page_config(
    page_title="IntelliRAG — Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #060d1a;
    background-image:
        radial-gradient(ellipse at 20% 10%, rgba(99,60,255,0.12) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(0,180,255,0.08) 0%, transparent 50%),
        linear-gradient(180deg, #060d1a 0%, #080f1e 100%);
    color: #e2e8f0;
}

.main .block-container {
    padding-top: 0.8rem;
    padding-bottom: 0.5rem;
    max-width: 1280px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080f1f 0%, #060d1a 100%);
    border-right: 1px solid rgba(99,60,255,0.2);
}
[data-testid="stSidebar"] * { font-size: 0.87rem; }

/* ── Hero ── */
.hero-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.3rem;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(135deg, #7b5cf5 0%, #4fc3f7 50%, #00d4b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.hero-tag {
    background: rgba(99,60,255,0.15);
    color: #7b5cf5;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    border: 1px solid rgba(99,60,255,0.4);
    letter-spacing: 0.08em;
    white-space: nowrap;
}
.hero-sub {
    color: #64748b;
    font-size: 0.8rem;
    margin-bottom: 0.7rem;
}

/* ── Badges ── */
.badge { display:inline-block; padding:0.2rem 0.65rem; border-radius:20px; font-size:0.71rem; font-weight:600; }
.badge-ready { background:rgba(0,212,80,0.12); color:#00d450; border:1px solid rgba(0,212,80,0.35); }
.badge-empty { background:rgba(255,60,60,0.12); color:#ff4444; border:1px solid rgba(255,60,60,0.3); }

/* ── Stat boxes ── */
.stat-box {
    background: linear-gradient(135deg, rgba(13,25,50,0.9), rgba(8,18,38,0.9));
    border: 1px solid rgba(99,60,255,0.2);
    border-radius: 10px;
    padding: 0.55rem 0.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.stat-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #7b5cf5, #4fc3f7, #00d4b8);
}
.stat-num { font-size: 1.3rem; font-weight: 700; color: #4fc3f7; line-height: 1.2; }
.stat-icon { font-size: 1rem; margin-bottom: 0.1rem; }
.stat-label { font-size: 0.6rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; }

/* ── Cards ── */
.glass-card {
    background: rgba(13,25,50,0.7);
    border: 1px solid rgba(99,60,255,0.15);
    border-radius: 12px;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}

/* ── Chat messages ── */
.chat-user {
    background: rgba(20,35,70,0.8);
    border: 1px solid rgba(99,60,255,0.2);
    border-radius: 14px 14px 4px 14px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
    color: #e2e8f0;
    font-size: 0.88rem;
}
.chat-assistant {
    background: rgba(8,18,38,0.9);
    border: 1px solid rgba(79,195,247,0.15);
    border-left: 3px solid #4fc3f7;
    border-radius: 4px 14px 14px 14px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
    color: #e2e8f0;
    line-height: 1.65;
    font-size: 0.88rem;
}
.resp-time {
    font-size: 0.67rem;
    color: #475569;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.3rem;
}

/* ── Source cards ── */
.source-card {
    background: rgba(13,25,50,0.7);
    border: 1px solid rgba(99,60,255,0.15);
    border-radius: 10px;
    padding: 0.65rem 0.85rem;
    margin: 0.35rem 0;
    transition: border-color 0.2s;
}
.source-card:hover { border-color: rgba(79,195,247,0.35); }
.source-fname { color: #4fc3f7; font-weight: 600; font-size: 0.76rem; }
.source-page { color: #7b5cf5; font-size: 0.7rem; font-family: 'JetBrains Mono', monospace; }
.source-score { float:right; font-weight:700; font-size:0.73rem; }
.source-ex { color: #64748b; margin-top: 0.3rem; font-size: 0.76rem; line-height: 1.5; }

/* ── Doc info card ── */
.doc-info-card {
    background: rgba(13,25,50,0.7);
    border: 1px solid rgba(99,60,255,0.2);
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
    margin: 0.35rem 0;
}

/* ── Sidebar chat buttons ── */
.stButton > button {
    background: rgba(99,60,255,0.15);
    color: #a78bfa;
    border: 1px solid rgba(99,60,255,0.25);
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.83rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: rgba(99,60,255,0.3);
    border-color: rgba(99,60,255,0.5);
    color: #c4b5fd;
}

/* ── Input ── */
.stTextInput > div > div > input {
    background: rgba(13,25,50,0.8) !important;
    border: 1px solid rgba(99,60,255,0.25) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(79,195,247,0.5) !important;
    box-shadow: 0 0 0 2px rgba(79,195,247,0.1) !important;
}

div[data-testid="stForm"] { border: none; padding: 0; }

/* ── Form submit button ── */
[data-testid="stForm"] button[kind="primaryFormSubmit"],
[data-testid="stForm"] button {
    background: linear-gradient(135deg, #7b5cf5, #4fc3f7) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

.section-label {
    color: #475569;
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0.8rem 0 0.35rem 0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,60,255,0.3); border-radius: 4px; }
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
            "model": "gemini-2.5-flash",
            "temperature": 0.1,
            "max_tokens": 1024,
            "top_k": 5,
            "score_threshold": 0.0,
            "use_hybrid": True,
            "use_reranking": True,
        }
    if "renaming_chat" not in st.session_state:
        st.session_state.renaming_chat = None


init_state()


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


def score_color(pct):
    if pct >= 70: return "#00d450"
    if pct >= 45: return "#f59e0b"
    return "#64748b"


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

if pipeline_ok:
    s = st.session_state.settings
    if s["model"] != status_data.get("generative_model"):
        pipeline.set_model(s["model"])


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo + name
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.6rem;padding:0.3rem 0 0.2rem 0;">'
        f'{LOGO_HTML}'
        f'<div>'
        f'<div style="font-size:1.05rem;font-weight:700;background:linear-gradient(135deg,#7b5cf5,#4fc3f7);'
        f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">IntelliRAG</div>'
        f'<div style="font-size:0.6rem;color:#475569;letter-spacing:0.06em;">DOCUMENT INTELLIGENCE</div>'
        f'</div></div>',
        unsafe_allow_html=True)

    st.markdown('<div style="height:0.4rem;"></div>', unsafe_allow_html=True)

    if is_ready:
        st.markdown('<span class="badge badge-ready">● READY</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-empty">○ NO DOCS</span>', unsafe_allow_html=True)

    # ── CHATS ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">💬 Chats</div>', unsafe_allow_html=True)

    if st.button("＋ New Chat", use_container_width=True, key="new_chat_btn"):
        create_chat()
        st.rerun()

    for cid, chat in list(st.session_state.chats.items()):
        is_active = cid == st.session_state.active_chat
        msg_count = len(chat["messages"]) // 2

        if st.session_state.renaming_chat == cid:
            # Inline rename
            new_name = st.text_input(
                "Chat name", value=chat["name"],
                key=f"rename_{cid}", label_visibility="collapsed"
            )
            col_ok, col_cancel = st.columns(2)
            with col_ok:
                if st.button("✓ Save", key=f"save_{cid}", use_container_width=True):
                    if new_name.strip():
                        st.session_state.chats[cid]["name"] = new_name.strip()
                    st.session_state.renaming_chat = None
                    st.rerun()
            with col_cancel:
                if st.button("✕ Cancel", key=f"cancel_{cid}", use_container_width=True):
                    st.session_state.renaming_chat = None
                    st.rerun()
        else:
            col_btn, col_edit, col_del = st.columns([6, 1, 1])
            with col_btn:
                label = f"{'▶ ' if is_active else ''}{chat['name']} ({msg_count})"
                if st.button(label, key=f"chat_{cid}", use_container_width=True):
                    st.session_state.active_chat = cid
                    st.rerun()
            with col_edit:
                if st.button("✏", key=f"edit_{cid}", help="Rename"):
                    st.session_state.renaming_chat = cid
                    st.rerun()
            with col_del:
                if st.button("🗑", key=f"del_{cid}", help="Delete"):
                    delete_chat(cid)
                    st.rerun()

    # ── DOCUMENTS ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">📁 Documents</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        if st.button("⚡ Ingest Documents", use_container_width=True):
            with st.spinner("Processing..."):
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

    # Indexed documents
    if is_ready and status_data.get("sources"):
        st.markdown(
            '<div style="font-size:0.72rem;color:#475569;margin-bottom:0.3rem;">'
            'Select to filter query scope:</div>',
            unsafe_allow_html=True)

        for src in status_data["sources"]:
            ext = Path(src).suffix.lstrip(".").lower()
            icon = {"pdf": "📕", "docx": "📘", "txt": "📄", "md": "📝"}.get(ext, "📄")

            col_chk, col_info, col_del = st.columns([1, 6, 1])
            with col_chk:
                checked = st.checkbox(
                    f"Select {src}",
                    key=f"doc_{src}",
                    value=src in st.session_state.selected_docs,
                    label_visibility="collapsed",
                )
                if checked:
                    st.session_state.selected_docs.add(src)
                else:
                    st.session_state.selected_docs.discard(src)
            with col_info:
                short = src[:20] + "…" if len(src) > 20 else src
                st.markdown(
                    f'<div style="font-size:0.77rem;color:#94a3b8;padding-top:0.35rem;">'
                    f'{icon} {short}</div>',
                    unsafe_allow_html=True)
            with col_del:
                if st.button("✕", key=f"del_doc_{src}", help=f"Remove {src}"):
                    with st.spinner("Removing..."):
                        removed = pipeline.delete_document(src)
                    st.session_state.selected_docs.discard(src)
                    st.success(f"Removed {removed} chunks")
                    st.rerun()

        if st.session_state.selected_docs:
            st.markdown(
                f'<div style="font-size:0.73rem;color:#7b5cf5;margin-top:0.2rem;">'
                f'🔍 Filtering: {len(st.session_state.selected_docs)} doc(s)</div>',
                unsafe_allow_html=True)

    # ── SETTINGS ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">⚙️ Settings</div>', unsafe_allow_html=True)

    with st.expander("Model & Generation", expanded=False):
        s = st.session_state.settings
        models = ["gemini-2.5-flash", "gemini-2.5-flash-lite-preview-06-17",
                  "gemini-2.5-pro", "gemini-3.1-flash-lite"]
        cur = s.get("model", "gemini-2.5-flash")
        idx = models.index(cur) if cur in models else 0
        new_model = st.selectbox("Gemini Model", options=models, index=idx)
        if new_model != s.get("model"):
            s["model"] = new_model
            if pipeline_ok:
                pipeline.set_model(new_model)
        s["temperature"] = st.slider("Temperature", 0.0, 1.0, s.get("temperature", 0.1), 0.05,
                                     help="Lower = factual, Higher = creative")
        s["max_tokens"] = st.slider("Max Response Tokens", 256, 2048, s.get("max_tokens", 1024), 128)

    with st.expander("Retrieval", expanded=False):
        s = st.session_state.settings
        s["top_k"] = st.slider("Top-K Chunks", 1, 15, s.get("top_k", 5),
                                help="Chunks passed to Gemini after reranking")
        s["score_threshold"] = st.slider("Min Relevance", 0.0, 1.0, s.get("score_threshold", 0.0), 0.05)
        s["use_hybrid"] = st.toggle("🔀 Hybrid BM25 + Vector", value=s.get("use_hybrid", True))
        s["use_reranking"] = st.toggle("🎯 CrossEncoder Reranking", value=s.get("use_reranking", True))
        st.markdown(
            f'<div style="font-size:0.71rem;color:#475569;margin-top:0.3rem;">'
            f'Model: <code>{status_data.get("embedding_model","—").split("/")[-1]}</code> · '
            f'Candidates: <code>{status_data.get("candidate_k",30)}</code>→'
            f'<code>{s.get("top_k",5)}</code></div>',
            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="hero-row">'
    f'{LOGO_HTML}'
    f'<span class="hero-title">IntelliRAG</span>'
    f'<span class="hero-tag">RAG-POWERED</span>'
    f'<span style="color:#475569;font-size:0.8rem;">Intelligent RAG System · Document Intelligence</span>'
    f'</div>'
    f'<div class="hero-sub">'
    f'Upload docs → Ask in plain English → Get cited answers · '
    f'<span style="color:#7b5cf5;">HuggingFace</span> + '
    f'<span style="color:#4fc3f7;">Gemini</span>'
    f'</div>',
    unsafe_allow_html=True)

# ── Stats strip ───────────────────────────────────────────────────────────────
if pipeline_ok and is_ready:
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        (c1, status_data["unique_documents"], "📄", "Documents"),
        (c2, status_data["total_chunks"], "🧩", "Chunks"),
        (c3, len(active_messages()) // 2, "💬", "Queries"),
        (c4, status_data["embedding_dim"], "🔢", "Vector Dim"),
    ]
    for col, num, icon, label in stats:
        with col:
            st.markdown(
                f'<div class="stat-box">'
                f'<div class="stat-icon">{icon}</div>'
                f'<div class="stat-num">{num}</div>'
                f'<div class="stat-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True)
    st.markdown('<div style="height:0.4rem;"></div>', unsafe_allow_html=True)

# ── Chat title ────────────────────────────────────────────────────────────────
current_chat = st.session_state.chats[st.session_state.active_chat]
st.markdown(
    f'<div style="font-size:0.88rem;font-weight:600;'
    f'background:linear-gradient(90deg,#7b5cf5,#4fc3f7);'
    f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
    f'background-clip:text;margin-bottom:0.5rem;">'
    f'💬 {current_chat["name"]}</div>',
    unsafe_allow_html=True)

# ── Main columns ──────────────────────────────────────────────────────────────
chat_col, src_col = st.columns([3, 2])

with chat_col:
    msgs = active_messages()
    if not msgs:
        if is_ready:
            st.markdown(
                '<div style="color:#475569;font-size:0.87rem;padding:0.5rem 0;">'
                '✨ Documents ingested and ready — ask anything below.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="glass-card" style="padding:1.3rem 1.5rem;">
  <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:0.6rem;">👋 Welcome to IntelliRAG!</div>
  <div style="color:#64748b;font-size:0.83rem;line-height:1.65;margin-bottom:0.9rem;">
    An AI-powered document Q&amp;A system. Upload your files and ask questions in plain English —
    get accurate, cited answers grounded in your documents.
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;margin-bottom:0.7rem;">
    <div style="background:rgba(8,18,38,0.7);border:1px solid rgba(99,60,255,0.2);border-radius:8px;padding:0.8rem;">
      <div style="color:#7b5cf5;font-weight:700;font-size:0.72rem;margin-bottom:0.4rem;letter-spacing:0.06em;">📥 SUPPORTED FILES</div>
      <div style="color:#64748b;font-size:0.78rem;line-height:1.75;">
        📕 PDF — papers, reports, books<br>
        📘 DOCX — word documents<br>
        📄 TXT / MD — plain text
      </div>
    </div>
    <div style="background:rgba(8,18,38,0.7);border:1px solid rgba(79,195,247,0.15);border-radius:8px;padding:0.8rem;">
      <div style="color:#4fc3f7;font-weight:700;font-size:0.72rem;margin-bottom:0.4rem;letter-spacing:0.06em;">💬 EXAMPLE QUERIES</div>
      <div style="color:#64748b;font-size:0.78rem;line-height:1.75;">
        → What is this document about?<br>
        → Summarize chapter 3<br>
        → Compare X and Y
      </div>
    </div>
  </div>
  <div style="background:rgba(8,18,38,0.7);border:1px solid rgba(0,212,80,0.15);border-radius:8px;padding:0.7rem;margin-bottom:0.7rem;">
    <div style="color:#00d450;font-weight:700;font-size:0.72rem;margin-bottom:0.4rem;letter-spacing:0.06em;">⚙️ HOW IT WORKS</div>
    <div style="display:flex;gap:0.4rem;flex-wrap:wrap;align-items:center;">
      <span style="background:rgba(0,212,80,0.12);color:#00d450;padding:0.2rem 0.55rem;border-radius:20px;font-size:0.71rem;border:1px solid rgba(0,212,80,0.25);">1. Upload docs</span>
      <span style="color:#334155;font-size:0.8rem;">→</span>
      <span style="background:rgba(0,212,80,0.12);color:#00d450;padding:0.2rem 0.55rem;border-radius:20px;font-size:0.71rem;border:1px solid rgba(0,212,80,0.25);">2. Click Ingest</span>
      <span style="color:#334155;font-size:0.8rem;">→</span>
      <span style="background:rgba(0,212,80,0.12);color:#00d450;padding:0.2rem 0.55rem;border-radius:20px;font-size:0.71rem;border:1px solid rgba(0,212,80,0.25);">3. Ask anything</span>
      <span style="color:#334155;font-size:0.8rem;">→</span>
      <span style="background:rgba(0,212,80,0.12);color:#00d450;padding:0.2rem 0.55rem;border-radius:20px;font-size:0.71rem;border:1px solid rgba(0,212,80,0.25);">4. Cited answers</span>
    </div>
  </div>
  <div style="background:rgba(123,92,245,0.1);border:1px solid rgba(123,92,245,0.3);border-radius:8px;padding:0.65rem 0.9rem;">
    <span style="color:#a78bfa;font-size:0.8rem;">🚀 <b>Get started:</b> Upload a file in the <b>Documents</b> section on the left, then click <b>⚡ Ingest Documents</b></span>
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
        st.markdown("""
<div class="glass-card" style="padding:1.2rem 1.3rem;">
  <div style="color:#7b5cf5;font-weight:700;font-size:0.72rem;margin-bottom:0.8rem;letter-spacing:0.08em;">🧠 POWERED BY</div>
  <div style="display:flex;flex-direction:column;gap:0.45rem;">
    <div style="background:rgba(8,18,38,0.7);border:1px solid rgba(99,60,255,0.15);border-radius:8px;padding:0.65rem 0.8rem;display:flex;justify-content:space-between;align-items:center;">
      <div><div style="color:#e2e8f0;font-size:0.8rem;font-weight:600;">HuggingFace Embeddings</div>
      <div style="color:#475569;font-size:0.7rem;">all-MiniLM-L6-v2 · 384-dim vectors</div></div>
      <span style="color:#7b5cf5;font-size:0.68rem;font-weight:600;">EMBED</span>
    </div>
    <div style="background:rgba(8,18,38,0.7);border:1px solid rgba(99,60,255,0.15);border-radius:8px;padding:0.65rem 0.8rem;display:flex;justify-content:space-between;align-items:center;">
      <div><div style="color:#e2e8f0;font-size:0.8rem;font-weight:600;">FAISS Vector Store</div>
      <div style="color:#475569;font-size:0.7rem;">IndexFlatIP · cosine similarity</div></div>
      <span style="color:#4fc3f7;font-size:0.68rem;font-weight:600;">STORE</span>
    </div>
    <div style="background:rgba(8,18,38,0.7);border:1px solid rgba(99,60,255,0.15);border-radius:8px;padding:0.65rem 0.8rem;display:flex;justify-content:space-between;align-items:center;">
      <div><div style="color:#e2e8f0;font-size:0.8rem;font-weight:600;">BM25 + CrossEncoder</div>
      <div style="color:#475569;font-size:0.7rem;">Hybrid search · reranking</div></div>
      <span style="color:#00d450;font-size:0.68rem;font-weight:600;">RANK</span>
    </div>
    <div style="background:rgba(8,18,38,0.7);border:1px solid rgba(99,60,255,0.15);border-radius:8px;padding:0.65rem 0.8rem;display:flex;justify-content:space-between;align-items:center;">
      <div><div style="color:#e2e8f0;font-size:0.8rem;font-weight:600;">Google Gemini</div>
      <div style="color:#475569;font-size:0.7rem;">Context-grounded generation</div></div>
      <span style="color:#f59e0b;font-size:0.68rem;font-weight:600;">GENERATE</span>
    </div>
  </div>
  <div style="margin-top:0.8rem;padding-top:0.65rem;border-top:1px solid rgba(99,60,255,0.12);display:flex;justify-content:space-around;">
    <span style="color:#475569;font-size:0.7rem;">⚡ &lt;5s</span>
    <span style="color:#475569;font-size:0.7rem;">🎯 Page citations</span>
    <span style="color:#475569;font-size:0.7rem;">🔒 Grounded</span>
  </div>
</div>""", unsafe_allow_html=True)

    elif msgs and is_ready:
        # Document info cards
        if status_data.get("sources"):
            for src in status_data["sources"]:
                ext = Path(src).suffix.lstrip(".").lower()
                icon = {"pdf": "📕", "docx": "📘", "txt": "📄", "md": "📝"}.get(ext, "📄")
                # Count chunks for this doc
                doc_chunks = sum(
                    1 for c in pipeline.vector_store._chunks
                    if c.get("filename") == src
                ) if pipeline_ok else "—"
                doc_pages = next(
                    (c.get("doc_pages") for c in pipeline.vector_store._chunks
                     if c.get("filename") == src and c.get("doc_pages")),
                    None
                ) if pipeline_ok else None
                pages_info = f"· {doc_pages} pages" if doc_pages else ""
                st.markdown(
                    f'<div class="doc-info-card">'
                    f'<div style="color:#4fc3f7;font-size:0.77rem;font-weight:600;">{icon} {src[:28]}{"…" if len(src)>28 else ""}</div>'
                    f'<div style="color:#475569;font-size:0.7rem;margin-top:0.15rem;">'
                    f'{doc_chunks} chunks indexed{pages_info} · {ext.upper()}</div>'
                    f'</div>',
                    unsafe_allow_html=True)

        # Sources from last answer
        last_sources = None
        for msg in reversed(msgs):
            if msg.get("sources"):
                last_sources = msg["sources"]
                break

        if last_sources:
            st.markdown(
                '<div style="font-size:0.71rem;color:#475569;font-weight:700;'
                'text-transform:uppercase;letter-spacing:0.08em;margin:0.6rem 0 0.35rem 0;">'
                '📎 Source Citations</div>',
                unsafe_allow_html=True)

            raw_scores = [s["relevance_score"] for s in last_sources]
            min_s, max_s = min(raw_scores), max(raw_scores)
            score_range = max_s - min_s

            for src in last_sources:
                raw = src["relevance_score"]
                pct = int((raw - min_s) / score_range * 100) if score_range > 0 else 50
                clr = score_color(pct)
                page_ref = src.get("page_ref", "")
                page_html = (f'<span class="source-page"> · {page_ref}</span>'
                             if page_ref else "")
                st.markdown(
                    f'<div class="source-card">'
                    f'<span class="source-fname">📄 {src["filename"]}</span>'
                    f'{page_html}'
                    f'<span class="source-score" style="color:{clr};">{pct}%</span>'
                    f'<div class="source-ex">{src["excerpt"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
if st.session_state.selected_docs:
    docs_str = ", ".join(list(st.session_state.selected_docs)[:2])
    if len(st.session_state.selected_docs) > 2:
        docs_str += f" +{len(st.session_state.selected_docs)-2} more"
    st.markdown(
        f'<div style="font-size:0.74rem;color:#7b5cf5;margin-bottom:0.2rem;">'
        f'🔍 Querying: {docs_str}</div>',
        unsafe_allow_html=True)

with st.form(key="query_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        question = st.text_input(
            "Ask a question",
            placeholder="Ask anything about your documents...",
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

        # Attach page_ref to sources
        sources_with_pages = []
        for src in result.sources:
            chunk_match = next(
                (c for c in pipeline.vector_store._chunks
                 if c.get("filename") == src["filename"]
                 and c.get("chunk_idx") == src["chunk_idx"]),
                None
            )
            src["page_ref"] = chunk_match.get("page_ref") if chunk_match else None
            sources_with_pages.append(src)

        active_messages().append({
            "role": "assistant",
            "content": result.answer,
            "sources": sources_with_pages,
            "elapsed": result.elapsed_seconds,
            "chunks": result.chunks_retrieved,
            "temperature": s["temperature"],
            "method": result.retrieval_method,
        })
        st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center;color:#1e293b;font-size:0.67rem;margin-top:0.4rem;">'
    'IntelliRAG · HuggingFace + FAISS + BM25 + CrossEncoder + Google Gemini · Streamlit'
    '</div>',
    unsafe_allow_html=True)