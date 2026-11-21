"""QuickQuery AI - Hybrid Text-to-SQL & Document Intelligence Engine (Streamlit UI)."""
import io
import os
import re

import pandas as pd
import streamlit as st
from fpdf import FPDF
from dotenv import load_dotenv

load_dotenv()
try:  # Streamlit Community Cloud: expose secrets as env vars so LiteLLM finds the keys
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

from backend.db_manager import DB_TYPES, DEFAULT_PORTS, DBManager, build_url  # noqa: E402
from backend.demo_seeder import DB_PATH, PDF_PATH, ensure_demo_data, reset_demo_data  # noqa: E402
from backend.llm_router import DEFAULT_MODEL, KEY_ENV, MODELS, AgentResult, QueryAgent  # noqa: E402
from backend.rag_manager import RAGManager  # noqa: E402

st.set_page_config(page_title="QuickQuery AI", page_icon="⚡", layout="wide")

STARTERS = [
    ("Hybrid · SQL + PDF", "Which employees qualify for full-time remote work per our handbook policy, "
                           "and how many PTO days do they have left?"),
    ("SQL only", "What is the average salary and headcount per department?"),
    ("PDF only", "What is the maximum number of PTO days that can carry over to next year?"),
]

st.markdown("""
<style>
[data-testid="stSidebar"] {background:#111827; border-right:1px solid #1F2937;}
.block-container {max-width:none; padding-top:24px; padding-left:3rem; padding-right:3rem;}
[data-testid="stSidebarHeader"] {height:60px; min-height:60px; padding-top:0; padding-bottom:0; position:relative; z-index:2;}
[data-testid="stSidebarUserContent"] {padding-top:0; margin-top:-60px;}
[data-testid="stHeader"] {background:transparent;}
[data-testid="InputInstructions"] {display:none;}
.qq-brand {font-size:1.35rem; font-weight:700; letter-spacing:-0.02em; line-height:40px; margin:0; padding-top:24px;}
.qq-brand span {color:#3B82F6;}
.qq-label {text-transform:uppercase; font-size:0.68rem; letter-spacing:0.08em; color:#64748B;
           font-weight:600; margin:1.2rem 0 0.35rem;}
.qq-badge {display:block; box-sizing:border-box; width:100%; padding:7px 14px; border-radius:8px; font-size:0.74rem; font-weight:500;
           margin:0.2rem 0 0.5rem;}
.qq-badge.ok {background:rgba(16,185,129,.12); color:#34D399; border:1px solid rgba(16,185,129,.35);}
.qq-badge.err {background:rgba(239,68,68,.12); color:#F87171; border:1px solid rgba(239,68,68,.35);}
.qq-badge.idle {background:rgba(148,163,184,.1); color:#94A3B8; border:1px solid #334155;}
.qq-doc {display:flex; justify-content:space-between; font-size:0.8rem; padding:6px 9px; margin-bottom:5px;
         border:1px solid #1F2937; border-radius:8px; background:#0B0F19;}
.qq-doc span {color:#64748B;}
.qq-col {font-size:0.8rem; color:#CBD5E1; line-height:1.55;}
.qq-col code {color:#93C5FD; background:transparent; padding:0;}
.qq-col i {color:#64748B; font-style:normal; font-size:0.72rem;}
.qq-hero h1 {font-size:1.9rem; font-weight:700; letter-spacing:-0.03em; line-height:40px; margin:0; padding:0;}
.qq-hero p {color:#94A3B8; margin:0.15rem 0 0.6rem;}
.qq-pill {display:inline-block; background:#111827; border:1px solid #1F2937; color:#CBD5E1;
          border-radius:999px; padding:2px 10px; font-size:0.72rem; margin-right:6px;}
.qq-tag {font-size:0.66rem; color:#3B82F6; text-transform:uppercase; letter-spacing:.07em; font-weight:600;}
.qq-empty {border:1px dashed #1F2937; border-radius:12px; padding:1.4rem; color:#94A3B8; text-align:center;
           margin:1rem 0 1.4rem;}
[data-testid="stMain"] .stButton > button {text-align:left; justify-content:flex-start; border-radius:12px;
    border:1px solid #1F2937; background:#111827; color:#CBD5E1; min-height:88px; font-size:0.85rem;}
[data-testid="stMain"] .stButton > button:hover {border-color:#3B82F6; color:#FFFFFF;}
</style>
""", unsafe_allow_html=True)


# ---------------- cached resources ----------------
@st.cache_resource(show_spinner=False)
def get_rag() -> RAGManager:
    return RAGManager()


@st.cache_resource(show_spinner=False)
def get_db(url: str) -> DBManager:
    return DBManager(url)


def index_handbook(rag: RAGManager) -> None:
    if PDF_PATH.exists():
        rag.index_pdf(PDF_PATH.read_bytes(), PDF_PATH.name)


@st.cache_resource(show_spinner="Preparing demo data (first run downloads a small embedding model)…")
def bootstrap() -> bool:
    ensure_demo_data()
    rag = get_rag()
    if not rag.list_documents():
        index_handbook(rag)
    return True


bootstrap()
rag = get_rag()
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("indexed_uploads", set())
ss.setdefault("uploader_key", 0)
ss.setdefault("api_keys", {})


def friendly_error(e: Exception, model_label: str) -> str:
    msg = str(e).strip().splitlines()[0][:300] if str(e).strip() else type(e).__name__
    if "auth" in type(e).__name__.lower() or "api key" in msg.lower() or "api_key" in msg.lower():
        return (f"**Missing or invalid API key for {model_label}.** Add `{KEY_ENV[model_label]}` to your `.env` "
                f"(or Streamlit secrets), or paste a key in the API KEY box.")
    return f"**Something went wrong:** {msg}"


# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown('<p class="qq-brand">⚡ Quick<span>Query</span> AI</p>', unsafe_allow_html=True)

    st.markdown('<div class="qq-label">Model & Database</div>', unsafe_allow_html=True)
    model_label = st.selectbox("LLM", list(MODELS), index=list(MODELS).index(DEFAULT_MODEL))
    # One key box per model; the agent always uses the key that matches the chosen LLM (empty = .env key).
    key_for = st.selectbox("API KEY", list(MODELS), index=list(MODELS).index(model_label),
                           format_func=lambda m: f"{m} key")
    ss.api_keys[key_for] = st.text_input(f"{key_for} key", value=ss.api_keys.get(key_for, ""),
                                         type="password", label_visibility="collapsed")
    api_key = ss.api_keys.get(model_label, "")
    db_type = st.selectbox("DATABASE", list(DB_TYPES))

    if db_type == "SQLite":
        db_url = build_url("SQLite", sqlite_path=DB_PATH)
    else:
        prefix = "MYSQL" if db_type == "MySQL" else "POSTGRES"
        with st.form(f"conn_{db_type}"):
            c1, c2 = st.columns([3, 1.3])
            host = c1.text_input("Host", os.getenv(f"{prefix}_HOST", "localhost"))
            port = c2.text_input("Port", os.getenv(f"{prefix}_PORT", str(DEFAULT_PORTS[db_type])))
            user = st.text_input("User", os.getenv(f"{prefix}_USER", ""))
            password = st.text_input("Password", os.getenv(f"{prefix}_PASSWORD", ""), type="password")
            database = st.text_input("Database name", os.getenv(f"{prefix}_DATABASE", ""))
            if st.form_submit_button("Connect", width="stretch"):
                ss[f"url_{db_type}"] = build_url(db_type, host, port, user, password, database)
        db_url = ss.get(f"url_{db_type}")

    db, db_error = None, None
    if db_url is not None:
        try:
            db = get_db(db_url.render_as_string(hide_password=False))
        except Exception as e:  # noqa: BLE001
            db_error = str(getattr(e, "orig", None) or e).splitlines()[0][:200]

    st.markdown('<div class="qq-label">Schema & Knowledge</div>', unsafe_allow_html=True)
    if db:
        st.markdown(f'<span class="qq-badge ok">● Connected · {db_type} · {len(db.schema)} table(s)</span>',
                    unsafe_allow_html=True)
    elif db_error:
        st.markdown('<span class="qq-badge err">● Connection failed</span>', unsafe_allow_html=True)
        st.caption(db_error)
    else:
        st.markdown('<span class="qq-badge idle">● Not connected</span>', unsafe_allow_html=True)

    with st.expander("View Database Schema"):
        if not db:
            st.caption("Connect a database to see its tables.")
        for table, cols in (db.schema.items() if db else []):
            lines = "<br>".join(f"<code>{c['name']}</code> <i>{c['type']}{' · PK' if c['pk'] else ''}</i>"
                                for c in cols)
            st.markdown(f"**{table}**<div class='qq-col'>{lines}</div>", unsafe_allow_html=True)

    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True,
                               key=f"uploader_{ss.uploader_key}")
    for f in uploads or []:
        file_key = (f.name, f.size)
        if file_key not in ss.indexed_uploads:
            with st.spinner(f"Indexing {f.name}…"):
                n = rag.index_pdf(f.getvalue(), f.name)
            ss.indexed_uploads.add(file_key)
            if n:
                st.toast(f"Indexed {f.name} · {n} chunks")
            else:
                st.warning(f"No text found in {f.name} (scanned PDF?).")

    docs = rag.list_documents()
    for d in docs:
        st.markdown(f'<div class="qq-doc">📄 {d["name"]}<span>{d["pages"]}p · {d["chunks"]} chunks</span></div>',
                    unsafe_allow_html=True)
    if not docs:
        st.caption("No PDFs indexed yet.")

    if st.button("Reset Demo Data", width="stretch"):
        if db is not None and db.dialect == "sqlite":
            db.dispose()
        get_db.clear()
        rag.reset()
        reset_demo_data()
        index_handbook(rag)
        ss.messages, ss.indexed_uploads = [], set()
        ss.uploader_key += 1
        st.rerun()


# ---------------- downloads ----------------
def _plain(text: str) -> str:
    """Markdown answer -> plain text, limited to characters the built-in PDF font supports."""
    text = re.sub(r"[*#`]", "", text or "")
    for a, b in {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
                 "\u2022": "-", "\u2026": "...", "\u2192": "->", "\u00a0": " "}.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


def to_text(question: str, result: AgentResult, df: pd.DataFrame | None) -> bytes:
    parts = [f"Question: {question}", "", f"Answer ({result.model}):", _plain(result.answer)]
    if df is not None:
        parts += ["", "Data:", df.to_string(index=False)]
    return "\n".join(parts).encode("utf-8")


def to_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False, sheet_name="Results")
    return buf.getvalue()


def to_pdf(question: str, result: AgentResult, df: pd.DataFrame | None) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, "QuickQuery AI - Answer", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 6, _plain(f"Question: {question}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, _plain(result.answer), new_x="LMARGIN", new_y="NEXT")
    if df is not None and not df.empty:
        pdf.ln(4)
        pdf.set_font("Helvetica", "", 8)
        with pdf.table(text_align="LEFT", line_height=5) as table:
            for values in [list(df.columns)] + df.head(500).values.tolist():
                row = table.row()
                for v in values:
                    row.cell(_plain(str(v)))
    return bytes(pdf.output())


def render_downloads(question: str, result: AgentResult, df: pd.DataFrame | None, key: str) -> None:
    cols = st.columns(4)
    if df is not None:
        cols[0].download_button("CSV", df.to_csv(index=False).encode("utf-8"), "quickquery_results.csv",
                                "text/csv", key=f"csv_{key}", icon=":material/download:", width="stretch")
        cols[1].download_button("Excel", to_excel(df), "quickquery_results.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"xlsx_{key}", icon=":material/download:", width="stretch")
    cols[2].download_button("PDF", to_pdf(question, result, df), "quickquery_answer.pdf", "application/pdf",
                            key=f"pdf_{key}", icon=":material/download:", width="stretch")
    cols[3].download_button("Text", to_text(question, result, df), "quickquery_answer.txt", "text/plain",
                            key=f"txt_{key}", icon=":material/download:", width="stretch")


# ---------------- main workspace ----------------
def render_result(result: AgentResult, question: str = "", key: str = "0") -> None:
    st.markdown(result.answer)
    shown, main_df = set(), None
    for run in result.sql_runs:
        if run.ok and run.df is not None and not run.df.empty and run.sql not in shown:
            shown.add(run.sql)
            main_df = run.df
            st.dataframe(run.df, width="stretch", hide_index=True)
    try:
        render_downloads(question, result, main_df, key)
    except Exception as e:  # noqa: BLE001 - a download problem must never hide the answer
        st.caption(f"Downloads unavailable: {e}")
    with st.expander("View SQL Query & PDF Sources"):
        st.markdown(f"**Model:** {result.model} &nbsp;·&nbsp; **Steps:** {result.steps}")
        st.markdown("**SQL executed**")
        if not result.sql_runs:
            st.caption("No SQL was needed for this answer.")
        for run in result.sql_runs:
            if run.corrected_from:
                st.caption(f"⚠️ First attempt failed: `{run.first_error}` → auto-corrected")
                st.code(run.corrected_from, language="sql")
            st.code(run.sql, language="sql")
            st.caption(f"✅ {len(run.df)} row(s) returned" if run.ok else f"❌ {run.error}")
        st.markdown("**PDF sources**")
        if not result.pdf_hits:
            st.caption("No documents were searched for this answer.")
        for h in result.pdf_hits:
            st.markdown(f"📄 **{h['source']}** · page {h['page']} · relevance {h['score']:.2f}")
            st.caption(h["text"][:700] + ("…" if len(h["text"]) > 700 else ""))


st.markdown('<div class="qq-hero"><h1>QuickQuery AI</h1>'
            '<p>Hybrid PDF-to-SQL & Document Intelligence Engine</p></div>'
            f'<span class="qq-pill">🧠 {model_label}</span><span class="qq-pill">🗄️ {db_type}</span>'
            f'<span class="qq-pill">📄 {len(docs)} PDF(s)</span>', unsafe_allow_html=True)

chat = st.container()
with chat:
    if not ss.messages:
        st.markdown('<div class="qq-empty">Ask anything about your database or documents in plain English.<br>'
                    'Try one of the prompts below.</div>', unsafe_allow_html=True)
    for i, m in enumerate(ss.messages):
        with st.chat_message(m["role"], avatar="⚡" if m["role"] == "assistant" else None):
            if m.get("result"):
                render_result(m["result"], m.get("question", ""), key=str(i))
            else:
                st.markdown(m["content"])

chip_prompt = None
for col, (tag, text) in zip(st.columns(3), STARTERS):
    with col:
        st.markdown(f'<div class="qq-tag">{tag}</div>', unsafe_allow_html=True)
        if st.button(text, key=f"chip_{tag}", width="stretch"):
            chip_prompt = text

prompt = st.chat_input("Ask about your data or documents…") or chip_prompt
if prompt:
    history = [{"role": m["role"], "content": m["content"]} for m in ss.messages]
    ss.messages.append({"role": "user", "content": prompt})
    with chat:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="⚡"):
            with st.status("Thinking…", expanded=False) as status:
                def on_step(s: str) -> None:
                    status.update(label=s + "…")
                    status.write(s)
                try:
                    agent = QueryAgent(model_label, db, rag, api_key=api_key.strip() or None)
                    result = agent.run(prompt, history=history, on_step=on_step)
                    status.update(label=f"Done · {result.steps} step(s)", state="complete")
                except Exception as e:  # noqa: BLE001
                    result = None
                    error_text = friendly_error(e, model_label)
                    status.update(label="Failed", state="error")
            if result:
                render_result(result, prompt, key=str(len(ss.messages)))
                ss.messages.append({"role": "assistant", "content": result.answer, "result": result,
                                    "question": prompt})
            else:
                st.markdown(error_text)
                ss.messages.append({"role": "assistant", "content": error_text})
