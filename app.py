import os
import sys
import streamlit as st
from textblob import TextBlob

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocessor import Preprocessor
from feature_extractor import FeatureExtractor
from summarizer import ExtractiveSummarizer, AbstractiveSummarizer
from action_item_extractor import ActionItemExtractor
from utils import parse_docx, parse_pdf, create_pdf_download_link, transcribe_audio

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MeetingAI — Smart Meeting Summarizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    /* ── Base ─────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px;
    }

    /* ── Hide Streamlit default chrome ────────────── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── App background ────────────────────────────── */
    .stApp {
        background: #0f1117;
    }

    /* ── Sidebar ────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #161b27 !important;
        border-right: 1px solid #2a2f3e;
    }
    [data-testid="stSidebar"] * { color: #c9d1d9 !important; }
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSlider label { color: #8b949e !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #e6edf3 !important; }

    /* ── Hero Header ────────────────────────────────── */
    .hero {
        text-align: center;
        padding: 2.5rem 1rem 2rem;
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 50%, #ff7b72 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    .hero-sub {
        font-size: 1.1rem;
        color: #8b949e;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
    }

    /* ── Tab styling ─────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: #161b27;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid #2a2f3e;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #8b949e;
        border-radius: 8px;
        font-weight: 500;
        padding: 0.5rem 1.5rem;
        font-size: 0.95rem;
    }
    .stTabs [aria-selected="true"] {
        background: #1f6feb !important;
        color: white !important;
    }

    /* ── Input Card ──────────────────────────────────── */
    .input-card {
        background: #161b27;
        border: 1px solid #2a2f3e;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    }
    .stTextArea textarea {
        background: #0d1117 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextArea textarea:focus {
        border-color: #1f6feb !important;
        box-shadow: 0 0 0 3px rgba(31, 111, 235, 0.15) !important;
    }

    /* ── Buttons ─────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.65rem 1.5rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(31, 111, 235, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(31, 111, 235, 0.45) !important;
    }

    /* ── Result Cards ────────────────────────────────── */
    .result-card {
        background: #161b27;
        border: 1px solid #2a2f3e;
        border-radius: 16px;
        padding: 1.75rem;
        margin-bottom: 1.25rem;
    }
    .result-card-title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8b949e;
        margin-bottom: 1rem;
    }
    .result-card p {
        color: #c9d1d9;
        line-height: 1.75;
        font-size: 0.95rem;
        margin: 0;
    }

    /* ── Stat Pills ──────────────────────────────────── */
    .stat-row {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
    }
    .stat-pill {
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 50px;
        padding: 0.45rem 1rem;
        font-size: 0.85rem;
        font-weight: 500;
        color: #c9d1d9;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .stat-pill .dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #1f6feb;
    }

    /* ── Action Items ────────────────────────────────── */
    .task-item {
        background: #0d1117;
        border: 1px solid #2a2f3e;
        border-left: 4px solid #1f6feb;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        color: #c9d1d9;
        font-size: 0.9rem;
        line-height: 1.6;
        transition: border-color 0.2s, background 0.2s;
    }
    .task-item:hover { background: #161b27; border-left-color: #388bfd; }
    .task-item.decision { border-left-color: #bc8cff; }
    .task-item.deadline  { border-left-color: #ff7b72; }

    /* ── Keyword Chips ───────────────────────────────── */
    .chip {
        display: inline-block;
        background: rgba(31, 111, 235, 0.12);
        color: #58a6ff;
        border: 1px solid rgba(31, 111, 235, 0.3);
        border-radius: 50px;
        padding: 4px 14px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 3px;
    }

    /* ── Sentiment Display ───────────────────────────── */
    .vibe-card {
        background: #161b27;
        border: 1px solid #2a2f3e;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1.25rem;
    }
    .vibe-label {
        font-size: 1.2rem;
        font-weight: 700;
        color: #e6edf3;
        margin-bottom: 0.75rem;
    }
    .vibe-bar-bg {
        background: #21262d;
        border-radius: 10px;
        height: 10px;
        overflow: hidden;
    }
    .vibe-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease;
    }

    /* ── Entity Rows ─────────────────────────────────── */
    .entity-row {
        display: flex;
        justify-content: space-between;
        padding: 0.6rem 0;
        border-bottom: 1px solid #21262d;
        font-size: 0.9rem;
        color: #c9d1d9;
    }
    .entity-row:last-child { border-bottom: none; }
    .entity-label { color: #8b949e; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }

    /* ── QA Section ──────────────────────────────────── */
    .qa-answer {
        background: #0d1117;
        border: 1px solid #30363d;
        border-left: 4px solid #bc8cff;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-top: 0.5rem;
        color: #c9d1d9;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* ── Divider ─────────────────────────────────────── */
    hr { border-color: #21262d !important; }

    /* ── Streamlit component text overrides ──────────── */
    .stMarkdown, .stText, p, li, label { color: #c9d1d9 !important; }
    h1, h2, h3, h4 { color: #e6edf3 !important; }
    .stInfo, .stSuccess, .stWarning, .stError { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    preprocessor     = Preprocessor()
    feature_extractor = FeatureExtractor()
    ext_summarizer   = ExtractiveSummarizer()
    abs_summarizer   = AbstractiveSummarizer("facebook/bart-large-cnn")
    action_extractor = ActionItemExtractor()
    return preprocessor, feature_extractor, ext_summarizer, abs_summarizer, action_extractor

try:
    with st.spinner("Loading AI models..."):
        prep, feat_ext, ext_summ, abs_summ, action_ext = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ── Session State ─────────────────────────────────────────────────────────────
for _k, _v in [
    ("transcript", ""),
    ("run_analysis", False),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 MeetingAI")
    st.markdown("<p style='color:#8b949e; font-size:0.85rem;'>Smart Meeting Summarizer</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("**Summary Method**")
    summary_type = st.radio(
        "method", ["Abstractive (Smart)", "Extractive (Key Sentences)"],
        index=0, label_visibility="collapsed"
    )
    st.divider()

    st.markdown("**Summarization Depth**")
    max_len = st.slider("Max Length (words)", 50, 300, 150)
    min_len = st.slider("Min Length (words)", 20, 100, 40)
    st.divider()

    st.markdown("""
    <div style='background:#21262d; border:1px solid #30363d; border-radius:10px; padding:1rem; font-size:0.85rem; color:#8b949e;'>
    💡 <b style='color:#c9d1d9;'>Tip</b><br>
    <b>Abstractive</b> rewrites the content in a natural summary.<br><br>
    <b>Extractive</b> picks the most important original sentences.
    </div>
    """, unsafe_allow_html=True)

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🧠 MeetingAI</div>
    <div class="hero-sub">Transform your meeting transcripts into smart summaries, action items, and insights — instantly.</div>
</div>
""", unsafe_allow_html=True)

# ── Input Tabs ────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📄  Text & Document", "🎙️  Audio Recording"])

with tab1:
    col_in, col_act = st.columns([3, 1], gap="large")
    with col_in:
        st.markdown("**How would you like to provide the transcript?**")
        input_method = st.radio("Input Method", ["✏️  Paste Text", "📎  Upload File"],
                                horizontal=True, label_visibility="collapsed")

        if input_method == "✏️  Paste Text":
            st.session_state.transcript = st.text_area(
                "Transcript", height=320,
                placeholder="Paste your meeting transcript here...\n\nExample:\nAlice: Let's review Q3 goals.\nBob: I will finish the report by Friday.",
                label_visibility="collapsed"
            )
        else:
            up = st.file_uploader("Drop your file here", type=["pdf", "docx", "txt"],
                                  label_visibility="collapsed")
            if up:
                try:
                    if up.type == "application/pdf":
                        st.session_state.transcript = parse_pdf(up)
                    elif "wordprocessingml" in up.type:
                        st.session_state.transcript = parse_docx(up)
                    else:
                        st.session_state.transcript = up.read().decode("utf-8")
                    st.success(f"✅ **{up.name}** parsed successfully.")
                    with st.expander("Preview extracted text"):
                        st.code(st.session_state.transcript[:600] + "...", language=None)
                except Exception as e:
                    st.error(f"Parse error: {e}")

    with col_act:
        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='result-card'>
            <div class='result-card-title'>Ready to analyze</div>
            <p style='font-size:0.85rem; margin-bottom:1rem;'>Click below to run the full AI pipeline on your transcript.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀  Analyze Transcript", use_container_width=True):
            st.session_state.run_analysis = True

with tab2:
    st.markdown("""
    <div class='input-card'>
        <div class='result-card-title'>🎙️ Audio Transcription</div>
        <p style='font-size:0.9rem; margin-bottom:1.25rem;'>
            Upload a meeting recording. The AI will convert speech to text, then run the full analysis automatically.
        </p>
    </div>
    """, unsafe_allow_html=True)

    audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a", "flac"],
                                  label_visibility="collapsed")
    if audio_file:
        st.audio(audio_file)
        if st.button("🎙️  Transcribe & Analyze", use_container_width=True):
            ext = os.path.splitext(audio_file.name)[-1] or ".mp3"
            tmp = f"temp_meeting_audio{ext}"
            with open(tmp, "wb") as f:
                f.write(audio_file.getbuffer().tobytes())

            # ── Synchronous transcription on main thread (no threading needed) ──
            try:
                with st.spinner(f"🎙️ Transcribing **{audio_file.name}**… (first run downloads the model — please wait)"):
                    transcript = transcribe_audio(tmp)
                st.session_state.transcript   = transcript
                st.session_state.run_analysis = True
                st.success("✅ Transcription complete — running full analysis…")
            except Exception as exc:
                st.error(f"❌ Transcription failed: {exc}")
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

st.divider()

# (threading poller removed — transcription now runs synchronously in main thread)

# ── Analysis ──────────────────────────────────────────────────────────────────
if st.session_state.run_analysis and st.session_state.transcript.strip():
    st.session_state.run_analysis = False
    tx = st.session_state.transcript

    with st.spinner("Running AI analysis…"):
        # Summarize
        try:
            if summary_type == "Extractive (Key Sentences)":
                sents, _ = prep.process_for_extractive(tx)
                summary  = ext_summ.summarize(sents, top_n=max(3, int(max_len/30)))
            else:
                summary = abs_summ.summarize(tx, max_length=max_len, min_length=min_len)
        except Exception as e:
            st.error(f"Summarization failed: {e}")
            summary = ""

        # Keywords & Entities
        try:
            keywords = feat_ext.extract_keywords(tx, top_n=10)
            entities = feat_ext.extract_entities(tx)
        except Exception as e:
            st.warning(f"Feature extraction issue: {e}")
            keywords, entities = [], {}

        # Action items
        try:
            ai = action_ext.extract_action_items(tx)
        except Exception as e:
            st.warning(f"Action extraction issue: {e}")
            ai = {"tasks": [], "decisions": [], "deadlines": []}

        # Sentiment
        score = TextBlob(tx).sentiment.polarity
        if   score >  0.1: vibe_label, vibe_color, vibe_icon = "Positive & Productive", "#2ea043", "😊"
        elif score < -0.1: vibe_label, vibe_color, vibe_icon = "Tense / Negative",       "#f85149", "😟"
        else:               vibe_label, vibe_color, vibe_icon = "Neutral & Balanced",     "#d29922", "😐"

    # ── Dashboard ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='display:flex; align-items:center; justify-content:space-between; margin-bottom:1.5rem;'>
        <h2 style='margin:0; font-size:1.6rem; font-weight:700;'>📊 Analysis Results</h2>
    </div>
    """, unsafe_allow_html=True)

    # Stat pills row
    n_tasks     = len(ai["tasks"])
    n_decisions = len(ai["decisions"])
    n_deadlines = len(ai["deadlines"])
    st.markdown(f"""
    <div class='stat-row'>
        <div class='stat-pill'><span class='dot' style='background:#1f6feb'></span> {n_tasks} Tasks</div>
        <div class='stat-pill'><span class='dot' style='background:#bc8cff'></span> {n_decisions} Decisions</div>
        <div class='stat-pill'><span class='dot' style='background:#ff7b72'></span> {n_deadlines} Deadlines</div>
        <div class='stat-pill'><span class='dot' style='background:{vibe_color}'></span> {vibe_icon} {vibe_label}</div>
    </div>
    """, unsafe_allow_html=True)

    # PDF export
    try:
        pdf_link = create_pdf_download_link(summary, ai)
        st.markdown(pdf_link, unsafe_allow_html=True)
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"PDF export unavailable: {e}")

    left, right = st.columns([3, 2], gap="large")

    with left:
        # Summary
        st.markdown(f"""
        <div class='result-card'>
            <div class='result-card-title'>📖 Executive Summary</div>
            <p>{summary if summary else "Could not generate a summary."}</p>
        </div>
        """, unsafe_allow_html=True)

        # Action Items
        st.markdown("<div class='result-card-title' style='margin-top:1rem;'>🎯 Action Items</div>", unsafe_allow_html=True)
        act_t1, act_t2, act_t3 = st.tabs(["✅  Tasks", "💡  Decisions", "⏰  Deadlines"])

        with act_t1:
            if ai["tasks"]:
                st.markdown("".join(f"<div class='task-item'>{t}</div>" for t in ai["tasks"]), unsafe_allow_html=True)
            else:
                st.caption("No tasks detected.")

        with act_t2:
            if ai["decisions"]:
                st.markdown("".join(f"<div class='task-item decision'>{d}</div>" for d in ai["decisions"]), unsafe_allow_html=True)
            else:
                st.caption("No decisions detected.")

        with act_t3:
            if ai["deadlines"]:
                st.markdown("".join(
                    f"<div class='task-item deadline'><b>{dl['deadline']}</b> — {dl['context']}</div>"
                    for dl in ai["deadlines"]
                ), unsafe_allow_html=True)
            else:
                st.caption("No deadlines detected.")

        # Q&A
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='result-card-title'>❓ Ask about the Meeting</div>", unsafe_allow_html=True)
        q = st.text_input("Your question", placeholder="e.g. What did Alice say about the deadline?",
                          label_visibility="collapsed")
        if q:
            words = [w for w in q.split() if len(w) > 3]
            hits  = [ln for ln in tx.split("\n") if ln.strip() and any(w.lower() in ln.lower() for w in words)]
            if hits:
                st.markdown("".join(f"<div class='qa-answer'>{ln.strip()}</div>" for ln in hits[:4]), unsafe_allow_html=True)
            else:
                st.caption("No matching lines found. Try different keywords.")

    with right:
        # Sentiment
        st.markdown(f"""
        <div class='vibe-card'>
            <div class='result-card-title'>📊 Meeting Vibe</div>
            <div class='vibe-label'>{vibe_icon} {vibe_label}</div>
            <div class='vibe-bar-bg'>
                <div class='vibe-bar-fill' style='width:100%; background:{vibe_color};'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Entities
        persons  = list(entities.get("PERSON", []))[:6]
        orgs     = list(entities.get("ORG",    []))[:4]
        locations= list(entities.get("GPE",    []))[:4]
        entity_html = "<div class='result-card'><div class='result-card-title'>👥 Key People & Orgs</div>"
        if persons:
            entity_html += f"<div class='entity-row'><span class='entity-label'>People</span><span>{', '.join(persons)}</span></div>"
        if orgs:
            entity_html += f"<div class='entity-row'><span class='entity-label'>Orgs</span><span>{', '.join(orgs)}</span></div>"
        if locations:
            entity_html += f"<div class='entity-row'><span class='entity-label'>Places</span><span>{', '.join(locations)}</span></div>"
        if not persons and not orgs and not locations:
            entity_html += "<p style='color:#8b949e; font-size:0.85rem;'>None detected.</p>"
        entity_html += "</div>"
        st.markdown(entity_html, unsafe_allow_html=True)

        # Keywords
        if keywords:
            chips = "".join(f"<span class='chip'>{kw}</span>" for kw in keywords)
            st.markdown(f"""
            <div class='result-card'>
                <div class='result-card-title'>🏷️ Top Keywords</div>
                <div style='line-height:2.2;'>{chips}</div>
            </div>
            """, unsafe_allow_html=True)

elif st.session_state.run_analysis and not st.session_state.transcript.strip():
    st.warning("⚠️ Please provide a transcript before running analysis.")
    st.session_state.run_analysis = False
