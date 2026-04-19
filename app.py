import streamlit as st
import os
import sys

# Ensure src modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessor import Preprocessor
from feature_extractor import FeatureExtractor
from summarizer import ExtractiveSummarizer, AbstractiveSummarizer
from action_item_extractor import ActionItemExtractor

st.set_page_config(page_title="AI Meeting Summarizer", page_icon="📝", layout="wide")

# Custom CSS for a clean, modern UI
st.markdown("""
<style>
    .stTextArea textarea {
        font-size: 16px !important;
    }
    .main-header {
        text-align: center;
        color: #2e6c80;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .action-item {
        margin-bottom: 10px;
        padding: 10px;
        border-left: 5px solid #2e6c80;
        background-color: #ffffff;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load and cache the models so they don't reload on every interaction."""
    preprocessor = Preprocessor()
    feature_extractor = FeatureExtractor()
    ext_summarizer = ExtractiveSummarizer()
    abs_summarizer = AbstractiveSummarizer("facebook/bart-large-cnn")
    action_extractor = ActionItemExtractor()
    return preprocessor, feature_extractor, ext_summarizer, abs_summarizer, action_extractor

# Initialize Models
try:
    with st.spinner("Loading NLP Models (This might take a minute initially)..."):
        prep, feat_ext, ext_summ, abs_summ, action_ext = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

st.markdown('<h1 class="main-header">📝 AI Meeting Summarization System</h1>', unsafe_allow_html=True)
st.write("Automatically process your meeting transcripts to extract key points, action items, and concise summaries.")

# Input Section
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Meeting Transcript")
    sample_text = """Alice: Hi everyone, thanks for joining the product sync. Let's start with updates. Bob, how is the new feature coming along?
Bob: I've finished the backend API, but I need to integrate the frontend. I will do this by Friday.
Charlie: Great. Once that's done, we decided that we need to test it thoroughly before the launch next week.
Alice: Good point. Charlie, can you write the test cases by Wednesday?
Charlie: Yes, no problem.
Alice: Awesome. I need to prepare the presentation for the stakeholders. Are there any other topics?
Bob: We have an issue with the database latency. It's slowing down queries.
Alice: Let's discuss that in a separate meeting. I will schedule a call with the DevOps team tomorrow to address the latency."""
    transcript = st.text_area("Paste your meeting transcript here:", value=sample_text, height=350)

with col2:
    st.subheader("Settings")
    summary_type = st.radio("Summary Type", ["Abstractive (BART)", "Extractive (TF-IDF)"], index=0)
    generate_btn = st.button("Generate Summary & Insights", use_container_width=True, type="primary")

st.divider()

if generate_btn and transcript:
    with st.spinner("Processing transcript..."):
        # 1. Preprocessing (mostly internal for extractive/feature steps)
        sentences, cleaned_sentences = prep.process_for_extractive(transcript)
        
        # 2. Summarization
        if summary_type == "Extractive (TF-IDF)":
            summary = ext_summ.summarize(sentences, top_n=3)
        else:
            summary = abs_summ.summarize(transcript)

        # 3. Feature Extraction
        keywords = feat_ext.extract_keywords(transcript, top_n=5)
        entities = feat_ext.extract_entities(transcript)

        # 4. Action Item Extraction
        action_items = action_ext.extract_action_items(transcript)

        # UI Layout for Results
        res_col1, res_col2 = st.columns([2, 1])
        
        with res_col1:
            st.subheader("Executive Summary")
            st.info(summary)
            
            st.subheader("Action Items")
            
            # Tasks
            if action_items["tasks"]:
                st.markdown("✅ **Tasks**")
                for t in action_items["tasks"]:
                    st.markdown(f'<div class="action-item">{t}</div>', unsafe_allow_html=True)
            
            # Decisions
            if action_items["decisions"]:
                st.markdown("🎯 **Key Decisions**")
                for d in action_items["decisions"]:
                    st.markdown(f'<div class="action-item">{d}</div>', unsafe_allow_html=True)
                    
            # Deadlines
            if action_items["deadlines"]:
                st.markdown("⏰ **Deadlines**")
                for dl in action_items["deadlines"]:
                    st.markdown(f'<div class="action-item"><b>{dl["deadline"]}</b>: {dl["context"]}</div>', unsafe_allow_html=True)
                    
            if not any(action_items.values()):
                st.write("No action items detected.")

        with res_col2:
            st.subheader("Key Entities")
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            for label, items in entities.items():
                if label in ['PERSON', 'ORG', 'DATE', 'TIME']:
                    st.write(f"**{label}**: {', '.join(items)}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.subheader("Top Keywords")
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            # Render keywords as badges
            badges = " ".join([f"<span style='background-color:#2e6c80;color:white;padding:5px 10px;border-radius:15px;margin:2px;display:inline-block;'>{kw}</span>" for kw in keywords])
            st.markdown(badges, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
