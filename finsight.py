import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import re
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
import plotly.graph_objects as go
import tempfile
import time
import tiktoken
import datetime

# ── API KEY ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY is not set. Add it to Streamlit secrets or your environment.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── SESSION STATE INIT ────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "analytics": {"queries": 0, "total_tokens": 0, "total_cost": 0.0, "avg_response_time": 0.0},
        "uploaded_files": [],
        "chunk_count": 0,
        "confirm_reset": False,
        "messages": [],
        "chat_history": [],
        "charts": {},
        "vector_store": None,
        "files_signature": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_app_state():
    for key in ["analytics", "uploaded_files", "chunk_count", "confirm_reset",
                "messages", "chat_history", "charts", "vector_store", "files_signature"]:
        if key in st.session_state:
            del st.session_state[key]
    init_session_state()

init_session_state()

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = '''You are FinSight, an expert financial analyst specializing in Big Tech competitive intelligence. 
You have deep expertise in analyzing 10-K filings for Alphabet (Google), Amazon, and Microsoft.

Your behavior:
- Always specify WHICH company and WHICH document section your answer comes from
- When citing numbers, always include the fiscal year (e.g. "Amazon's 2024 10-K states...")
- If a question asks you to compare companies, structure your answer company by company
- If the answer is not found in the provided documents, say clearly: 
  "This information is not available in the uploaded 10-K filings." 
  Never guess or use outside knowledge to fill gaps.
- If the question is subjective or ambiguous (e.g. "which company is best?", "who is winning?", "who is doing better?"), do NOT refuse. Instead, interpret it as a financial performance comparison and answer using the most relevant available metrics such as revenue, operating income, net income, or growth rate. Structure your answer company by company and let the data speak for itself.
- For financial figures, always include units (billions, millions, %) and the time period
- Flag any notable risks or caveats when discussing financial health

Your tone:
- Professional but conversational — like a senior analyst briefing an executive
- Concise: lead with the direct answer, then support with detail
- Never use filler phrases like "Great question!" or "Certainly!"'''

# ── VISUAL REPORT SYSTEM PROMPT ───────────────────────────────────────────────
VISUAL_PROMPT = '''You are FinSight, an expert financial analyst. The user wants a VISUAL REPORT.

Your job:
1. Extract the relevant numbers from the context provided
2. Return a JSON object (and nothing else) in this exact format:

{
  "chart_type": "bar",
  "title": "Cloud Revenue Comparison 2024",
  "explanation": "AWS leads with $90.8B, followed by Azure at $60.2B and Google Cloud at $33.1B. AWS maintains market dominance but Azure shows fastest growth.",
  "data": {
    "labels": ["Amazon AWS", "Microsoft Azure", "Google Cloud"],
    "datasets": [
      {
        "name": "2024 Revenue",
        "values": [90.8, 60.2, 33.1]
      }
    ]
  },
  "unit": "$ Billions"
}

Chart type selection rules:
- Use "bar" for comparing values across companies or categories
- Use "line" for showing trends over multiple years (e.g., 2022, 2023, 2024)
- Use "pie" for showing composition/breakdown of a single entity (e.g., Amazon's revenue by segment)

CRITICAL RULES:
- Return ONLY the JSON. No markdown, no explanation outside the JSON, no backticks.
- If the user asks about multiple companies, you MUST include ALL of them.
- If a value is missing for one company, use 0 and note it in the explanation.
- If the data is not available in the context at all, return: {"error": "Data not available in the uploaded 10-K filings."}
- For pie charts, use "labels" for categories and ONE dataset with "values"
- Always specify the fiscal year in the title when applicable'''

# ── VISUAL INTENT DETECTION ───────────────────────────────────────────────────
VISUAL_KEYWORDS = [
    "chart", "graph", "plot", "visualize", "visualise", "visual",
    "show me a", "draw", "diagram", "compare visually", "visual report",
    "bar chart", "pie chart", "line chart", "line graph", "bar graph",
    "growth", "over time"
]

def is_visual_request(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in VISUAL_KEYWORDS)

# ── CHART RENDERER ────────────────────────────────────────────────────────────
def render_chart(chart_data: dict):
    if "error" in chart_data:
        return None

    chart_type = chart_data.get("chart_type", "bar")
    title      = chart_data.get("title", "")
    datasets   = chart_data["data"]["datasets"]
    labels     = chart_data["data"]["labels"]
    unit       = chart_data.get("unit", "")

    layout = dict(
        title=dict(text=title, font=dict(color="white", size=16), x=0.5),
        paper_bgcolor="rgba(17,45,31,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="rgba(255,255,255,0.85)", family="DM Sans"),
        legend=dict(bgcolor="rgba(255,255,255,0.05)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        margin=dict(t=60, b=60, l=60, r=30),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.1)",
                   title=unit if unit else ""),
    )

    COLORS = ["#f0c040", "#4ade80", "#60a5fa", "#f87171", "#a78bfa", "#fb923c"]

    if chart_type == "pie":
        values = datasets[0]["values"] if datasets else []
        fig = go.Figure(go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=COLORS[:len(labels)],
                        line=dict(color="rgba(17,45,31,1)", width=2)),
            textfont=dict(color="white"),
            hovertemplate="%{label}: %{value} " + unit + "<extra></extra>",
        ))
        fig.update_layout(**{k: v for k, v in layout.items() if k not in ("xaxis", "yaxis")})

    elif chart_type == "line":
        fig = go.Figure()
        for i, ds in enumerate(datasets):
            fig.add_trace(go.Scatter(
                x=labels, y=ds["values"], name=ds["name"],
                mode="lines+markers",
                line=dict(color=COLORS[i % len(COLORS)], width=2.5),
                marker=dict(size=7),
                hovertemplate="%{x}: %{y} " + unit + "<extra>" + ds["name"] + "</extra>",
            ))
        fig.update_layout(**layout)

    else:  # bar (default)
        fig = go.Figure()
        for i, ds in enumerate(datasets):
            fig.add_trace(go.Bar(
                x=labels, y=ds["values"], name=ds["name"],
                marker=dict(color=COLORS[i % len(COLORS)],
                            line=dict(color="rgba(255,255,255,0.1)", width=1)),
                hovertemplate="%{x}: %{y} " + unit + "<extra>" + ds["name"] + "</extra>",
            ))
        if len(datasets) > 1:
            fig.update_layout(barmode="group")
        fig.update_layout(**layout)

    return fig

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #112d1f; min-height: 100vh; }

    .demo-header {
        background: rgba(255,255,255,0.04);
        padding: 2.5rem 3rem;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 2rem;
    }
    .demo-header h1 {
        font-family: 'DM Serif Display', serif;
        color: white;
        text-align: center;
        margin: 0 0 0.5rem 0;
        font-size: 2.4rem;
        letter-spacing: -0.5px;
    }
    .demo-header p {
        color: rgba(255,255,255,0.75);
        text-align: center;
        font-size: 1.1rem;
        margin: 0;
        font-weight: 300;
    }

    .metric-container {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        transition: background 0.2s;
        text-align: center;
    }
    .metric-container:hover { background: rgba(255,255,255,0.09); }
    .metric-label {
        color: rgba(255,255,255,0.75);
        font-size: 0.88rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        letter-spacing: 0.2px;
    }
    .metric-value {
        color: #f0c040;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.1;
        letter-spacing: -0.5px;
    }

    .section-title {
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: 0.2px;
    }

    .chat-scroll-area {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 20px;
        background: rgba(255,255,255,0.03);
        padding: 1rem;
        min-height: 600px;
        max-height: 600px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .message-wrapper {
        display: flex;
        flex-direction: column;
        max-width: 75%;
        margin: 0.2rem 0;
    }
    .message-wrapper.user {
        margin-left: auto;
        margin-right: 0;
        align-items: flex-end;
    }
    .message-wrapper.assistant {
        margin-right: auto;
        margin-left: 0;
        align-items: flex-start;
    }

    .message-bubble {
        padding: 0.7rem 1rem;
        border-radius: 18px;
        position: relative;
        word-wrap: break-word;
    }
    .message-bubble.user {
        background: linear-gradient(135deg, #f0c040 0%, #d4a834 100%);
        color: #1a1a1a;
        border-bottom-right-radius: 4px;
        margin-right: 0.5rem;
    }
    .message-bubble.assistant {
        background: rgba(255,255,255,0.08);
        color: rgba(255,255,255,0.95);
        border: 1px solid rgba(255,255,255,0.12);
        border-bottom-left-radius: 4px;
        margin-left: 0.5rem;
    }

    .sender-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.35rem;
        opacity: 0.7;
    }
    .message-bubble.user .sender-label { color: rgba(26,26,26,0.6); text-align: right; }
    .message-bubble.assistant .sender-label { color: rgba(255,255,255,0.6); text-align: left; }

    .message-content {
        font-size: 0.93rem;
        line-height: 1.55;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .message-bubble.user .message-content { color: #1a1a1a; }
    .message-bubble.assistant .message-content { color: rgba(255,255,255,0.95); }

    .message-time { font-size: 0.68rem; margin-top: 0.25rem; opacity: 0.5; }
    .message-bubble.user .message-time { color: rgba(26,26,26,0.5); text-align: right; }
    .message-bubble.assistant .message-time { color: rgba(255,255,255,0.5); text-align: left; }

    .chart-container {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 0.75rem 1.5rem 0.25rem 1.5rem;
        margin-top: 1rem;
    }
    .chart-label {
        color: rgba(255,255,255,0.5);
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin: 0 0 0.25rem 0;
    }
    .chart-explanation {
        color: rgba(255,255,255,0.75);
        font-size: 0.9rem;
        line-height: 1.6;
        margin-top: 1rem;
        padding: 0.75rem 1rem;
        background: rgba(255,255,255,0.04);
        border-left: 3px solid #f0c040;
        border-radius: 0 8px 8px 0;
    }

    .stChatInputContainer {
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 16px !important;
        background: rgba(255,255,255,0.05) !important;
        margin-top: 0.8rem !important;
        padding: 0.4rem 0.6rem !important;
    }
    .stChatInputContainer textarea {
        color: white !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 0.3rem 0 !important;
    }

    .upload-placeholder {
        background: rgba(255,255,255,0.05);
        border: 2px dashed rgba(150,255,150,0.2);
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        color: rgba(255,255,255,0.5);
        font-size: 1rem;
        margin-top: 1rem;
    }
    .upload-placeholder .icon { font-size: 3rem; display: block; margin-bottom: 1rem; }
    .upload-placeholder strong {
        display: block;
        color: rgba(255,255,255,0.75);
        font-size: 1.1rem;
        margin-bottom: 0.4rem;
    }

    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 14px !important;
    }
    [data-testid="stExpander"] summary {
        color: rgba(255,255,255,0.85) !important;
        font-weight: 500 !important;
    }

    hr { border-color: rgba(255,255,255,0.1) !important; margin: 1.5rem 0 !important; }

    .footer {
        text-align: center;
        color: rgba(255,255,255,0.4);
        font-size: 0.85rem;
        padding: 1rem 0;
        font-weight: 300;
    }

    .stButton > button {
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        background: rgba(255,255,255,0.08) !important;
        color: white !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        background: rgba(255,255,255,0.16) !important;
        border-color: rgba(255,255,255,0.35) !important;
    }

    .stDownloadButton > button {
        border-radius: 10px !important;
        border: 1px solid rgba(240,192,64,0.35) !important;
        background: rgba(240,192,64,0.07) !important;
        color: #f0c040 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        transition: all 0.2s !important;
    }
    .stDownloadButton > button:hover {
        background: rgba(240,192,64,0.15) !important;
        border-color: rgba(240,192,64,0.65) !important;
    }

    [data-testid="stChatMessage"] {
        padding: 0 !important;
        margin: 0 !important;
        background: transparent !important;
        border: none !important;
    }
    [data-testid="stChatMessageAvatar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="demo-header">
    <h1>📈 FinSight</h1>
    <p>Real-time financial insights — ask anything, or say "show me a chart of..." for visual reports!</p>
</div>
""", unsafe_allow_html=True)

# ── METRICS ROW ───────────────────────────────────────────────────────────────
doc_count   = len(st.session_state.get("uploaded_files", []))
chunk_count = st.session_state.get("chunk_count", 0)
queries     = st.session_state.analytics["queries"]
cost        = f"${st.session_state.analytics['total_cost']:.4f}"

st.markdown(f"""
<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.25rem; margin-bottom: 0.5rem;">
    <div class="metric-container">
        <div class="metric-label">📁 Documents</div>
        <div class="metric-value">{doc_count}</div>
    </div>
    <div class="metric-container">
        <div class="metric-label">🔍 Chunks</div>
        <div class="metric-value">{chunk_count}</div>
    </div>
    <div class="metric-container">
        <div class="metric-label">⚡ Queries</div>
        <div class="metric-value">{queries}</div>
    </div>
    <div class="metric-container">
        <div class="metric-label">💰 Cost</div>
        <div class="metric-value">{cost}</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

# ── TWO-COLUMN LAYOUT ─────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.4, 3.6], gap="large")

with left_col:
    st.markdown('<p class="section-title">💼 Financial Documents</p>', unsafe_allow_html=True)
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "📊 Upload 10-K Reports",
        accept_multiple_files=True,
        type="pdf",
        help="Upload 10-Ks or other financial files for analysis and insights."
    )
    st.session_state.uploaded_files = uploaded_files or []
    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    if not st.session_state.confirm_reset:
        if st.button("🔄 Reset Analytics", use_container_width=True):
            st.session_state.confirm_reset = True
            st.rerun()
    else:
        st.warning("⚠️ This will clear all documents, chat history, and analytics. Are you sure?")
        confirm_col, cancel_col = st.columns(2)
        with confirm_col:
            if st.button("✅ Yes, reset", use_container_width=True):
                reset_app_state()
                st.rerun()
        with cancel_col:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.confirm_reset = False
                st.rerun()

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(240,192,64,0.07); border:1px solid rgba(240,192,64,0.2);
                border-radius:12px; padding:0.9rem 1rem;">
        <p style="color:#f0c040; font-size:0.78rem; font-weight:700; margin:0 0 0.4rem 0;
                  text-transform:uppercase; letter-spacing:0.5px;">📊 Visual Reports</p>
        <p style="color:rgba(255,255,255,0.65); font-size:0.82rem; margin:0; line-height:1.5;">
            Add <strong style="color:rgba(255,255,255,0.9);">"chart"</strong>, 
            <strong style="color:rgba(255,255,255,0.9);">"graph"</strong>, or 
            <strong style="color:rgba(255,255,255,0.9);">"visualize"</strong> to any 
            question to get an interactive chart.<br><br>
            e.g. <em style="color:rgba(255,255,255,0.55);">"Bar chart of cloud revenue across all three companies"</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

with right_col:

    if st.session_state.get("messages"):
        chat_export = "\n\n".join([
            f"{'You' if m['role'] == 'user' else 'FinSight'}: {m['content']}"
            for m in st.session_state.messages
        ])
        _, btn_col = st.columns([4, 1])
        with btn_col:
            st.download_button(
                "💾 Export Chat",
                chat_export,
                "finsight_chat.txt",
                use_container_width=True,
            )

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # ── DOCUMENT PROCESSING ───────────────────────────────────────────────────
    if uploaded_files and st.session_state.get("vector_store") is None:
        status = st.empty()
        progress_bar = st.progress(0)

        documents = []
        with tempfile.TemporaryDirectory() as temp_dir:
            total = len(uploaded_files)
            for i, file in enumerate(uploaded_files):
                status.markdown(
                    f"<p style='color:rgba(255,255,255,0.7); font-size:0.9rem;'>"
                    f"📄 Processing <strong>{file.name}</strong> ({i+1}/{total})…</p>",
                    unsafe_allow_html=True
                )
                progress_bar.progress(int((i / total) * 70))
                temp_file_path = os.path.join(temp_dir, file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(file.getbuffer())
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                company = file.name.lower()
                if "amazon" in company:
                    company_name = "Amazon"
                elif "alphabet" in company or "google" in company:
                    company_name = "Alphabet (Google)"
                elif "microsoft" in company:
                    company_name = "Microsoft"
                else:
                    company_name = file.name
                for d in docs:
                    d.metadata["company"] = company_name
                    d.metadata["source"] = file.name
                documents.extend(docs)

        status.markdown(
            "<p style='color:rgba(255,255,255,0.7); font-size:0.9rem;'>"
            "🔗 Building vector index…</p>",
            unsafe_allow_html=True
        )
        progress_bar.progress(80)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        st.session_state.chunk_count = len(docs)

        progress_bar.progress(95)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        st.session_state.vector_store = FAISS.from_documents(docs, embeddings)
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.charts = {}

        progress_bar.progress(100)
        status.empty()
        progress_bar.empty()

        st.success(f"✅ {total} document{'s' if total > 1 else ''} loaded — FinSight is ready!")
        time.sleep(1.5)
        st.rerun()

    # ── CHAT AREA ─────────────────────────────────────────────────────────────
    if st.session_state.get("vector_store") is None:
        st.markdown("""
        <div class="upload-placeholder">
            <span class="icon">📂</span>
            <strong>No documents loaded yet</strong>
            Upload one or more 10-K PDFs on the left to activate FinSight.
        </div>
        """, unsafe_allow_html=True)

    else:
        messages_html = ""
        if not st.session_state.messages:
            messages_html = (
                "<p style='color:rgba(255,255,255,0.35); text-align:center; "
                "margin-top:160px; font-size:0.92rem; line-height:1.6;'>"
                "✅ <strong>FinSight is ready!</strong><br/>"
                "Ask about revenue trends, risk factors, or segment performance.<br/>"
                'Add <strong>"chart"</strong> or <strong>"graph"</strong> to get a visual report!</p>'
            )
        else:
            for message in st.session_state.messages[-20:]:
                is_user = message["role"] == "user"
                sender = "You" if is_user else "FinSight"
                msg_class = "user" if is_user else "assistant"
                timestamp = message.get("timestamp", "")
                content = (
                    message["content"]
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                time_html = f'<div class="message-time">{timestamp}</div>' if timestamp else ""
                messages_html += (
                    f'<div class="message-wrapper {msg_class}">'
                    f'<div class="message-bubble {msg_class}">'
                    f'<div class="sender-label">{sender}</div>'
                    f'<div class="message-content">{content}</div>'
                    f'{time_html}'
                    f'</div></div>'
                )

        st.markdown(f"""
        <div class="chat-scroll-area" id="chat-scroll-area">
            {messages_html}
        </div>
        """, unsafe_allow_html=True)

        user_input = st.chat_input("Ask anything — or say 'show me a chart of...' for visuals!")

        if user_input:
            if st.session_state.get("vector_store") is None:
                st.error("⚠️ Vector store not initialized. Please upload documents first.")
                st.stop()

            current_time = datetime.datetime.now().strftime("%I:%M %p")
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": current_time
            })
            st.session_state.pending_query = user_input
            st.session_state.query_start_time = time.time()
            st.rerun()

        # ── QUERY PROCESSING ──────────────────────────────────────────────────
        if st.session_state.get("pending_query"):
            pending    = st.session_state.pending_query
            start_time = st.session_state.query_start_time
            visual     = is_visual_request(pending)

            if visual:
                visual_qa_prompt = PromptTemplate(
                    input_variables=["context", "question"],
                    template=f"""{VISUAL_PROMPT}

Context from 10-K filings:
{{context}}

User request: {{question}}

JSON:"""
                )

                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 8})
                docs = retriever.invoke(pending)
                context = "\n\n".join([d.page_content for d in docs])
                filled_prompt = visual_qa_prompt.format(context=context, question=pending)
                
                with st.spinner("FinSight is building your visual report…"):
                    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
                    raw = llm.invoke(filled_prompt).content.strip()
                raw = re.sub(r"^```json\s*", "", raw, flags=re.MULTILINE)
                raw = re.sub(r"^```\s*",     "", raw, flags=re.MULTILINE)
                raw = re.sub(r"\s*```$",     "", raw, flags=re.MULTILINE)

                try:
                    chart_data    = json.loads(raw)
                    response_text = "Visual report generated — see chart below!"
                    msg_idx = len(st.session_state.messages)
                    st.session_state.charts[msg_idx] = chart_data
                except json.JSONDecodeError:
                    response_text = "I wasn't able to extract structured data for a chart. Try rephrasing, e.g. 'bar chart of total revenue for Amazon, Google, and Microsoft in 2024'."

            else:
                history_context = ""
                if st.session_state.chat_history:
                    last_human, last_ai = st.session_state.chat_history[-1]
                    history_context = f"Previous question: {last_human}\nPrevious answer summary: {last_ai[:300]}\n\nFollow-up: "
                enriched_query = history_context + pending

                qa_prompt = PromptTemplate(
                    input_variables=["context", "question"],
                    template=f"""{SYSTEM_PROMPT}

Use the following excerpts from the 10-K filings to answer the question.
If this is a follow-up question, use the conversation history below to understand what metric or company is being referenced.

Context:
{{context}}

Question: {{question}}

Answer:"""
                )

                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 8})
                docs = retriever.invoke(enriched_query)
                context = "\n\n".join([d.page_content for d in docs])
                filled_prompt = qa_prompt.format(context=context, question=enriched_query)
                
                with st.spinner("FinSight is thinking…"):
                    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
                    response_text = llm.invoke(filled_prompt).content.strip()

                st.session_state.chat_history.append((pending, response_text))
                if len(st.session_state.chat_history) > 6:
                    st.session_state.chat_history = st.session_state.chat_history[-6:]

            response_time = datetime.datetime.now().strftime("%I:%M %p")
            elapsed_time  = time.time() - start_time

            del st.session_state.pending_query
            del st.session_state.query_start_time

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "timestamp": response_time
            })

            encoding      = tiktoken.encoding_for_model("gpt-4o")
            input_tokens  = len(encoding.encode(pending))
            output_tokens = len(encoding.encode(response_text))
            tokens        = input_tokens + output_tokens
            query_cost    = (input_tokens / 1_000_000 * 2.50) + (output_tokens / 1_000_000 * 10.00)

            current_queries = st.session_state.analytics["queries"]
            new_queries     = current_queries + 1
            new_avg_time    = (
                elapsed_time if current_queries == 0
                else (st.session_state.analytics["avg_response_time"] * current_queries + elapsed_time) / new_queries
            )

            st.session_state.analytics.update({
                "queries":           new_queries,
                "total_tokens":      st.session_state.analytics["total_tokens"] + tokens,
                "total_cost":        st.session_state.analytics["total_cost"] + query_cost,
                "avg_response_time": new_avg_time,
            })

            st.rerun()

# ── VISUAL REPORT CHARTS ──────────────────────────────────────────────────────
if st.session_state.get("charts"):
    for msg_idx in sorted(st.session_state.charts.keys()):
        chart_info = st.session_state.charts[msg_idx]
        fig = render_chart(chart_info)
        if fig:
            chart_title = chart_info.get("title", "Visual Report")
            explanation = chart_info.get("explanation", "")
            st.markdown(f'<div class="chart-container"><p class="chart-label">📊 {chart_title}</p></div>', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, config={
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "displaylogo": False,
            })
            if explanation:
                st.markdown(
                    f'<div class="chart-explanation">💡 {explanation}</div>',
                    unsafe_allow_html=True
                )
            st.markdown("<hr style='border-color:rgba(255,255,255,0.06); margin: 0.5rem 0 1.5rem 0;'>", unsafe_allow_html=True)
        elif "error" in chart_info:
            st.warning(chart_info["error"])

# ── DETAILED ANALYTICS EXPANDER ───────────────────────────────────────────────
st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
with st.expander("📊 Detailed Analytics", expanded=False):
    avg_time  = f"{st.session_state.analytics['avg_response_time']:.2f}s"
    total_tok = f"{st.session_state.analytics['total_tokens']:,}"
    queries   = st.session_state.analytics["queries"]
    avg_tok_per_query = f"{int(st.session_state.analytics['total_tokens'] / queries):,}" if queries > 0 else "—"
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.25rem; padding: 0.5rem 0;">
        <div class="metric-container">
            <div class="metric-label">⏱️ Avg Response Time</div>
            <div class="metric-value">{avg_time}</div>
        </div>
        <div class="metric-container">
            <div class="metric-label">🔤 Total Tokens</div>
            <div class="metric-value">{total_tok}</div>
        </div>
        <div class="metric-container">
            <div class="metric-label">📊 Avg Tokens / Query</div>
            <div class="metric-value">{avg_tok_per_query}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p class='footer'>FinSight &nbsp;•&nbsp; Vritika Narra &nbsp;•&nbsp; Megan Huber &nbsp;•&nbsp; Fahda Alajmi &nbsp;•&nbsp; Yuying Ding &nbsp;•&nbsp; Jialu Li &nbsp;•&nbsp; Zhiruo Zhao</p>",
    unsafe_allow_html=True
)
