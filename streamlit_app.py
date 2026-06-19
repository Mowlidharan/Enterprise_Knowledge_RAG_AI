import streamlit as st
import asyncio
import os
import zipfile
from pathlib import Path
from datetime import datetime
import concurrent.futures

# Import existing RAG logic
from rag_engine import RagEngine, DocumentAlreadyExistsError
from llm_infrastructure_service import rag_config, cLog

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Enterprise Knowledge AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS — Fixed Theming
# ============================================================

st.markdown("""
<style>
/* ── 1. Root background ─────────────────────────────────── */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {
    background-color: #F5F7FA !important;
}

/* ── 2. Main content block ──────────────────────────────── */
.block-container {
    background-color: #F5F7FA !important;
    padding-top: 1.5rem !important;
    max-width: 1100px;
}

/* ── 3. Sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
}

/* ── 4. Typography (scoped — NOT wildcard *) ────────────── */
.stMarkdown p, .stMarkdown li, .stMarkdown span,
.stText, label[data-testid], div[data-testid="stText"] {
    color: #1E293B !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #0F172A !important;
    font-weight: 700 !important;
}

/* ── 5. Buttons — light style (default) ─────────────────── */
.stButton > button {
    background-color: #F1F5F9 !important;
    color: #1E293B !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 0.45rem 1rem !important;
}
.stButton > button:hover {
    background-color: #E2E8F0 !important;
    border-color: #94A3B8 !important;
    color: #0F172A !important;
}

/* Primary button (Process Documents) */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background-color: #3B82F6 !important;
    color: #FFFFFF !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
}

/* ── 6. Chat input — fix dark bar at bottom ─────────────── */
[data-testid="stBottom"] {
    background-color: #F5F7FA !important;
    border-top: 1px solid #E5E7EB !important;
}

[data-testid="stChatInputContainer"],
.stChatFloatingInputContainer,
[data-testid="stChatInput"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}

textarea[data-testid="stChatInputTextArea"] {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
}

/* ── 7. Chat messages ────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    margin-bottom: 0.75rem !important;
}

/* ── 8. Metrics ─────────────────────────────────────────── */
[data-testid="metric-container"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
}

[data-testid="stMetricValue"] {
    color: #0F172A !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

[data-testid="stMetricLabel"] {
    color: #64748B !important;
    font-size: 0.8rem !important;
}

/* ── 9. File uploader — fix doubled text ────────────────── */
[data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;
    border: 2px dashed #CBD5E1 !important;
    border-radius: 10px !important;
}

/* Hide the default label that duplicates the uploader text */
[data-testid="stFileUploaderDropzone"] label {
    display: none !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #64748B !important;
}

/* ── 10. Status widget ──────────────────────────────────── */
[data-testid="stStatusWidget"],
[data-testid="stStatus"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 10px !important;
}

/* ── 11. Expander ────────────────────────────────────────── */
[data-testid="stExpander"] {
    background-color: #F8FAFC !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
}

/* ── 12. Alert / info boxes ─────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
}

/* ── 13. General input fields ───────────────────────────── */
input {
    color: #1E293B !important;
    background-color: #FFFFFF !important;
}

/* ── 14. Divider ─────────────────────────────────────────── */
hr {
    border-color: #E5E7EB !important;
    margin: 0.75rem 0 !important;
}

/* ── 15. Scrollbar ───────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #F1F5F9; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }

/* ── 16. Hide Streamlit chrome ───────────────────────────── */
/* NOTE: stHeader must stay (not display:none) — it's where Streamlit
   renders the control to re-expand a collapsed sidebar. Hiding it
   entirely traps users with no way to bring the sidebar back. */
#MainMenu { visibility: hidden !important; }
footer    { display: none !important; }
[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
}
[data-testid="stToolbar"]    { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Async Helper — avoids "loop already running" crash
# ============================================================

def run_async(coro):
    """Run an async coroutine safely regardless of the current event-loop state."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Streamlit runs in a thread; delegate to a fresh thread with its own loop
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ============================================================
# Session-State Initialisation
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_registry" not in st.session_state:
    st.session_state.uploaded_files_registry = []

if "metrics" not in st.session_state:
    st.session_state.metrics = {"queries": 0}


# ============================================================
# RAG Engine
# ============================================================

@st.cache_resource
def get_rag_engine():
    return RagEngine()

rag_engine = get_rag_engine()


# ============================================================
# File Processing Helpers
# ============================================================

def process_single_file(file_obj, rag_engine, status) -> bool:
    """Save and ingest one file; return True on success."""
    upload_dir = Path(rag_config["UploadedDirFile"])
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file_obj.name
    with open(file_path, "wb") as f:
        f.write(file_obj.getbuffer())

    status.update(label=f"Indexing: {file_obj.name}…", state="running")

    try:
        result = run_async(rag_engine.ingest_document(
            collection_name="enterprise_knowledge",
            file_path=str(file_path),
            user_id="executive_user",
            spaceid="default_space"
        ))
        if result["status"]:
            st.session_state.uploaded_files_registry.append({
                "name": file_obj.name,
                "size": f"{file_obj.size / 1024:.1f} KB",
                "status": "✅ Success",
                "time": datetime.now().strftime("%H:%M:%S"),
            })
            return True
    except DocumentAlreadyExistsError:
        st.session_state.uploaded_files_registry.append({
            "name": file_obj.name,
            "size": f"{file_obj.size / 1024:.1f} KB",
            "status": "ℹ️ Already indexed",
            "time": datetime.now().strftime("%H:%M:%S"),
        })
        return True
    except Exception as e:
        st.error(f"Failed to ingest {file_obj.name}: {e}")
    return False


def handle_zip(zip_file, rag_engine, status):
    """Extract and process all valid documents inside a ZIP."""
    extract_dir = (
        Path(rag_config["ExtractedDirFile"])
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    extract_dir.mkdir(parents=True, exist_ok=True)

    status.update(label="Extracting ZIP archive…", state="running")
    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(extract_dir)

    extracted_files = [
        f for f in extract_dir.rglob("*")
        if f.is_file() and f.suffix.lstrip(".") in rag_config["document_validate"]
    ]
    status.update(label=f"Found {len(extracted_files)} valid documents in ZIP.", state="running")

    for idx, file_path in enumerate(extracted_files):
        status.update(
            label=f"Indexing ({idx + 1}/{len(extracted_files)}): {file_path.name}…",
            state="running",
        )
        try:
            run_async(rag_engine.ingest_document(
                collection_name="enterprise_knowledge",
                file_path=str(file_path),
                user_id="executive_user",
                spaceid="default_space"
            ))
            st.session_state.uploaded_files_registry.append({
                "name": file_path.name,
                "size": f"{os.path.getsize(file_path) / 1024:.1f} KB",
                "status": "✅ Success",
                "time": datetime.now().strftime("%H:%M:%S"),
            })
        except DocumentAlreadyExistsError:
            st.session_state.uploaded_files_registry.append({
                "name": file_path.name,
                "size": f"{os.path.getsize(file_path) / 1024:.1f} KB",
                "status": "ℹ️ Already indexed",
                "time": datetime.now().strftime("%H:%M:%S"),
            })
        except Exception as e:
            st.warning(f"Failed to process {file_path.name}: {e}")


def render_source(msg_idx: int, src_idx: int, source) -> None:
    """Render one retrieved source as a clickable reference instead of a raw text dump."""
    if not isinstance(source, dict):
        st.code(str(source), language="markdown")
        return

    file_name = source.get("file_name", "Unknown document")
    link = source.get("source_link")
    file_path = source.get("file_path")
    score = source.get("score")
    badge = f"  ·  relevance {score}" if score is not None else ""

    if link:
        st.markdown(f"**{src_idx}.** [{file_name}]({link}){badge}")
    elif file_path and os.path.exists(file_path):
        st.markdown(f"**{src_idx}.** {file_name}{badge}")
        with open(file_path, "rb") as fh:
            st.download_button(
                label="⬇️ Open source document",
                data=fh.read(),
                file_name=file_name,
                key=f"src_{msg_idx}_{src_idx}_{file_name}",
            )
    else:
        st.markdown(f"**{src_idx}.** {file_name}{badge}")


def dedupe_sources(sources: list) -> list:
    """Collapse multiple chunks from the same file into one reference, keeping the best-scoring chunk."""
    best_by_file: dict = {}
    order = []
    for source in sources:
        key = source.get("file_name", "Unknown document") if isinstance(source, dict) else str(source)
        if key not in best_by_file:
            best_by_file[key] = source
            order.append(key)
        elif isinstance(source, dict) and isinstance(best_by_file[key], dict):
            if (source.get("score") or 0) > (best_by_file[key].get("score") or 0):
                best_by_file[key] = source
    return [best_by_file[key] for key in order]


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## 📂 Upload Knowledge")

    uploaded_files = st.file_uploader(
        "Drop documents here",          # shown as accessible label
        type=rag_config["document_validate"],
        accept_multiple_files=True,
        help="Supports PDF, DOCX, PPTX, images and ZIP",
        label_visibility="collapsed",   # hide the duplicate on-screen label
    )

    if st.button("🚀 Process Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one file first.")
        else:
            with st.status("Initialising pipeline…", expanded=True) as status:
                for f in uploaded_files:
                    if f.name.lower().endswith(".zip"):
                        handle_zip(f, rag_engine, status)
                    else:
                        process_single_file(f, rag_engine, status)
                status.update(
                    label="✅ Knowledge base updated",
                    state="complete",
                    expanded=False,
                )
            st.success("All documents processed successfully!")

    st.markdown("---")
    st.markdown("### 📊 System Metrics")
    doc_count = run_async(rag_engine.get_document_count("enterprise_knowledge"))
    m1, m2 = st.columns(2)
    m1.metric("Documents", doc_count)
    m2.metric("Queries",   st.session_state.metrics["queries"])

    if st.session_state.uploaded_files_registry:
        with st.expander(
            f"📁 Indexed Files ({len(st.session_state.uploaded_files_registry)})",
            expanded=False,
        ):
            for f in st.session_state.uploaded_files_registry:
                st.markdown(f"**{f['name']}** — {f['size']} — {f['status']}")

    st.markdown("---")
    st.markdown("### ⚙️ Session Controls")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with c2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.clear()
            st.rerun()


# ============================================================
# Main Header
# ============================================================

hc1, hc2 = st.columns([8, 2])
with hc1:
    st.markdown("# Enterprise Knowledge AI")
    st.markdown("*Secure AI-Powered Document Intelligence*")
with hc2:
    st.success("● System Ready")

st.markdown("---")


# ============================================================
# Chat Display
# ============================================================

if not st.session_state.messages:
    st.markdown("""
    <div style="
        background:#EFF6FF;
        border:1px solid #BFDBFE;
        border-radius:12px;
        padding:20px 24px;
        margin:8px 0 16px 0;
    ">
        <h4 style="color:#1D4ED8;margin:0 0 10px 0;">👋 Welcome to Enterprise Knowledge AI</h4>
        <p style="color:#374151;margin:0;line-height:1.7;">
            <strong>Get started in 3 steps:</strong><br>
            1️⃣&nbsp; Upload your business documents in the sidebar.<br>
            2️⃣&nbsp; Click <strong>Process Documents</strong> to index the content.<br>
            3️⃣&nbsp; Ask questions to extract insights, summarise reports, or compare data.
        </p>
    </div>
    """, unsafe_allow_html=True)

for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        sources = dedupe_sources(message.get("sources") or [])
        if sources:
            with st.expander(f"📚 View Sources & Citations ({len(sources)})"):
                for src_idx, source in enumerate(sources, start=1):
                    render_source(msg_idx, src_idx, source)


# ============================================================
# Chat Input — FIX: st.rerun() moved OUTSIDE the chat_message
# context so the widget tree is not disrupted mid-render
# ============================================================

if query := st.chat_input("Ask about your enterprise knowledge…"):
    # Capture prior turns BEFORE appending the current question, so the
    # reframing step doesn't see the question it's about to answer as its own history
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-6:]
    ]

    # Show user bubble immediately
    st.session_state.messages.append({"role": "user", "content": query})

    # Generate answer
    answer = ""
    sources = []
    error_msg = ""

    with st.spinner("Analysing knowledge base…"):
        try:
            response_data = run_async(rag_engine.generate_response(
                question=query,
                collection_name="enterprise_knowledge",
                chat_history=history,
                spaceid="default_space",
                return_sources=True,
            ))
            
            # Robust response extraction
            fallback_answer = "I couldn't generate a response for that — please try rephrasing your question."
            if isinstance(response_data, dict):
                answer  = response_data.get("answer") or fallback_answer
                sources = response_data.get("sources", [])
            else:
                answer  = str(response_data).strip() or fallback_answer
                sources = []
        except Exception as e:
            error_msg = f"⚠️ I encountered an error while searching: {e}"

    # Persist assistant reply
    if error_msg:
        st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
    else:
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        st.session_state.metrics["queries"] += 1

    # Single rerun renders everything cleanly
    st.rerun()


# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.markdown(
    f'<p style="text-align:center;color:#94A3B8;font-size:13px;">'
    f"Enterprise Knowledge AI • v1.0.0 • Secure Document Intelligence • {datetime.now().year}"
    f"</p>",
    unsafe_allow_html=True,
)