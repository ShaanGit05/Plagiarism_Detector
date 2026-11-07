"""
PLAGIARISM DETECTOR: AN NLP-BASED TEXT SIMILARITY DETECTION SYSTEM
Based on the methodology from the project report.

Modules:
- Text Preprocessing & Cleaning (lowercase, punctuation removal, whitespace trimming)
- Sentence Tokenization using NLTK sent_tokenize
- N-gram Feature Extraction (Unigrams + Bigrams) with Scikit-learn TfidfVectorizer
- Document-level & Sentence-level Cosine Similarity calculation
- Threshold-based candidate match flagging & Matched-Text Share estimation
- Streamlit interactive web interface with visual highlighting
"""

import re
import html
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="Plagiarism Detector | NLP-Based Similarity System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassy Pink Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Global Typography */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Ambient Glassmorphic Pink Mesh Background */
    [data-testid="stAppViewContainer"], .stApp {
        background: 
            radial-gradient(at 8% 12%, rgba(251, 113, 133, 0.20) 0px, transparent 40%),
            radial-gradient(at 92% 8%, rgba(244, 114, 182, 0.24) 0px, transparent 45%),
            radial-gradient(at 48% 52%, rgba(249, 168, 212, 0.22) 0px, transparent 50%),
            radial-gradient(at 12% 88%, rgba(253, 164, 175, 0.22) 0px, transparent 45%),
            radial-gradient(at 88% 85%, rgba(244, 63, 94, 0.16) 0px, transparent 42%),
            linear-gradient(135deg, #FFF0F5 0%, #FDF2F8 30%, #FCE7F3 65%, #FFF5F7 100%) !important;
        background-attachment: fixed !important;
        color: #831843 !important;
    }

    /* Frosted Glass Top Streamlit Header */
    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.35) !important;
        backdrop-filter: blur(16px) saturate(160%) !important;
        -webkit-backdrop-filter: blur(16px) saturate(160%) !important;
        border-bottom: 1px solid rgba(244, 114, 182, 0.22) !important;
    }

    /* Frosted Glass Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.42) !important;
        backdrop-filter: blur(24px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
        border-right: 1px solid rgba(244, 114, 182, 0.28) !important;
        box-shadow: 4px 0 24px 0 rgba(244, 114, 182, 0.08) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #9D174D !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] span {
        color: #831843 !important;
    }

    /* Glass Header Card */
    .glass-header-card {
        background: rgba(255, 255, 255, 0.60);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.85);
        border-bottom: 2px solid rgba(244, 114, 182, 0.40);
        border-radius: 20px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 12px 36px 0 rgba(244, 114, 182, 0.14), inset 0 1px 2px 0 rgba(255, 255, 255, 0.95);
        position: relative;
        overflow: hidden;
    }
    .glass-header-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #F472B6, #EC4899, #DB2777, #F43F5E);
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #9D174D 0%, #BE185D 35%, #EC4899 75%, #F43F5E 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1.02rem;
        color: #9D174D;
        font-weight: 500;
        margin-bottom: 0;
        line-height: 1.55;
    }
    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(244, 114, 182, 0.16);
        border: 1px solid rgba(244, 114, 182, 0.38);
        color: #9D174D;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 4px 12px;
        border-radius: 999px;
        margin-bottom: 8px;
    }

    /* Document Panels & Titles */
    .glass-panel-title {
        font-size: 1.12rem;
        font-weight: 700;
        color: #9D174D;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .glass-meta-info {
        display: inline-block;
        background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(244, 114, 182, 0.32);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.8rem;
        color: #9D174D;
        font-weight: 600;
        margin-top: 6px;
        box-shadow: 0 2px 8px 0 rgba(244, 114, 182, 0.08);
    }

    /* Glass Text Areas */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(16px) saturate(160%) !important;
        -webkit-backdrop-filter: blur(16px) saturate(160%) !important;
        border: 1.5px solid rgba(244, 114, 182, 0.38) !important;
        border-radius: 16px !important;
        color: #4C0519 !important;
        font-size: 0.96rem !important;
        line-height: 1.65 !important;
        padding: 14px 16px !important;
        box-shadow: 0 4px 20px 0 rgba(244, 114, 182, 0.08), inset 0 1px 2px 0 rgba(255, 255, 255, 0.9) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stTextArea textarea:focus {
        border-color: #EC4899 !important;
        background: rgba(255, 255, 255, 0.82) !important;
        box-shadow: 0 0 0 3px rgba(236, 72, 153, 0.25), 0 8px 25px 0 rgba(244, 114, 182, 0.16) !important;
    }
    .stTextArea textarea::placeholder {
        color: #BE185D !important;
        opacity: 0.50 !important;
    }

    /* Primary Glass Buttons */
    .stButton > button[kind="primary"], .stButton > button:first-child {
        background: linear-gradient(135deg, #F472B6 0%, #EC4899 45%, #BE185D 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.50) !important;
        border-radius: 14px !important;
        font-size: 1.02rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em !important;
        padding: 10px 24px !important;
        box-shadow: 0 8px 24px -2px rgba(236, 72, 153, 0.48), inset 0 1px 1px 0 rgba(255, 255, 255, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button[kind="primary"]:hover, .stButton > button:first-child:hover {
        background: linear-gradient(135deg, #FB7185 0%, #F43F5E 45%, #9F1239 100%) !important;
        box-shadow: 0 12px 28px -2px rgba(244, 63, 94, 0.58), inset 0 1px 2px 0 rgba(255, 255, 255, 0.75) !important;
        transform: translateY(-2px) !important;
    }
    .stButton > button[kind="primary"]:active, .stButton > button:first-child:active {
        transform: translateY(1px) !important;
    }

    /* Sidebar Glass Secondary Buttons */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.55) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        color: #9D174D !important;
        border: 1px solid rgba(244, 114, 182, 0.45) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px 0 rgba(244, 114, 182, 0.10) !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(253, 242, 248, 0.85) !important;
        border-color: #EC4899 !important;
        color: #831843 !important;
        box-shadow: 0 6px 18px 0 rgba(244, 114, 182, 0.20) !important;
        transform: translateY(-1px) !important;
    }

    /* Glass Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.62);
        backdrop-filter: blur(18px) saturate(170%);
        -webkit-backdrop-filter: blur(18px) saturate(170%);
        border: 1px solid rgba(255, 255, 255, 0.85);
        border-bottom: 2.5px solid rgba(244, 114, 182, 0.45);
        border-radius: 18px;
        padding: 18px 14px;
        text-align: center;
        box-shadow: 0 10px 32px 0 rgba(244, 114, 182, 0.14), inset 0 1px 2px 0 rgba(255, 255, 255, 0.85);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 38px 0 rgba(244, 114, 182, 0.22), inset 0 1px 2px 0 rgba(255, 255, 255, 0.95);
        border-bottom-color: #EC4899;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 800;
        margin-top: 4px;
        letter-spacing: -0.02em;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #9D174D;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 700;
    }
    .metric-sub {
        color: #9D174D;
        font-size: 0.76rem;
        opacity: 0.85;
        font-weight: 500;
    }

    /* Glass Highlight Card */
    .highlight-card {
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(18px) saturate(170%);
        -webkit-backdrop-filter: blur(18px) saturate(170%);
        border: 1px solid rgba(255, 255, 255, 0.85);
        border-left: 5px solid #EC4899;
        border-radius: 18px;
        padding: 24px;
        line-height: 1.95;
        font-size: 1.03rem;
        box-shadow: 0 10px 32px 0 rgba(244, 114, 182, 0.12), inset 0 1px 2px 0 rgba(255, 255, 255, 0.90);
    }
    .match-sentence {
        background: rgba(254, 205, 211, 0.65);
        border-bottom: 2.5px solid #E11D48;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        color: #881337;
        box-shadow: 0 2px 6px rgba(225, 29, 72, 0.12);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
    }
    .clean-sentence {
        padding: 2px 4px;
        color: #374151;
    }

    /* Glass Match Boxes */
    .match-box {
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(16px) saturate(160%);
        -webkit-backdrop-filter: blur(16px) saturate(160%);
        border: 1px solid rgba(244, 114, 182, 0.35);
        border-left: 5px solid #EC4899;
        padding: 18px 22px;
        margin-bottom: 14px;
        border-radius: 16px;
        box-shadow: 0 6px 22px 0 rgba(244, 114, 182, 0.10), inset 0 1px 1px 0 rgba(255, 255, 255, 0.85);
        transition: transform 0.2s ease;
    }
    .match-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 28px 0 rgba(244, 114, 182, 0.16);
    }

    /* Glass Disclaimer Box */
    .disclaimer-box {
        background: rgba(255, 255, 255, 0.58);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(244, 114, 182, 0.32);
        border-left: 4px solid #DB2777;
        border-radius: 14px;
        padding: 16px 20px;
        color: #831843;
        font-size: 0.9rem;
        margin-top: 24px;
        box-shadow: 0 6px 20px 0 rgba(244, 114, 182, 0.08), inset 0 1px 1px 0 rgba(255, 255, 255, 0.8);
    }

    /* Glass Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.45) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-radius: 14px !important;
        padding: 6px !important;
        border: 1px solid rgba(244, 114, 182, 0.28) !important;
        gap: 6px !important;
        box-shadow: 0 4px 18px 0 rgba(244, 114, 182, 0.08) !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        color: #9D174D !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.92) !important;
        color: #BE185D !important;
        box-shadow: 0 4px 14px rgba(236, 72, 153, 0.25) !important;
    }

    /* Glass Expander */
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.48) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(244, 114, 182, 0.28) !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 16px 0 rgba(244, 114, 182, 0.06) !important;
        overflow: hidden !important;
    }

    /* Glass Selectbox & Slider */
    div[data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1.5px solid rgba(244, 114, 182, 0.35) !important;
        border-radius: 12px !important;
        color: #831843 !important;
    }
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #EC4899 !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 2px 10px rgba(236, 72, 153, 0.5) !important;
    }

    /* Glass Dataframe */
    [data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.68) !important;
        backdrop-filter: blur(18px) !important;
        -webkit-backdrop-filter: blur(18px) !important;
        border: 1px solid rgba(244, 114, 182, 0.28) !important;
        border-radius: 16px !important;
        padding: 6px !important;
        box-shadow: 0 8px 24px 0 rgba(244, 114, 182, 0.08) !important;
    }

    /* Smooth Pink Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255, 240, 245, 0.5);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(244, 114, 182, 0.5);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(236, 72, 153, 0.8);
    }
</style>
""", unsafe_allow_html=True)


# --- Core NLP and Vectorization Functions (from Report Section 4.1 & 4.2) ---

@st.cache_resource
def init_nltk():
    """Ensure required NLTK tokenizers and stop words are loaded."""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception:
        pass


init_nltk()


def clean_text(text: str) -> str:
    """Normalize text: convert to lowercase, remove punctuation, collapse extra spaces."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def split_sentences(text: str) -> list[str]:
    """Tokenize text into sentences using NLTK sent_tokenize."""
    try:
        sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
        if sentences:
            return sentences
    except Exception:
        pass
    # Fallback punctuation split if tokenizer resource is missing
    raw_sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw_sents if s.strip()]


def analyze(reference: str, submitted: str, threshold: float = 0.40):
    """
    Performs document-level and sentence-level similarity analysis.
    Implements TF-IDF vectorization with unigrams and bigrams, stop-words filtering,
    and cosine similarity matrix calculation.
    """
    ref_sents = split_sentences(reference)
    sub_sents = split_sentences(submitted)

    if not ref_sents or not sub_sents:
        return 0.0, 0.0, [], []

    # Document-Level Vectorization & Cosine Similarity
    doc_vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        stop_words='english',
        ngram_range=(1, 2)
    )

    try:
        doc_matrix = doc_vectorizer.fit_transform([reference, submitted])
        doc_score = float(cosine_similarity(doc_matrix[0:1], doc_matrix[1:2])[0, 0])
    except ValueError:
        # Handles empty vocabularies (e.g. only stop words or symbols entered)
        return 0.0, 0.0, [], []

    # Sentence-Level Vectorization & Cosine Similarity
    sent_vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        stop_words='english',
        ngram_range=(1, 2)
    )

    all_sentences = ref_sents + sub_sents
    try:
        sent_matrix = sent_vectorizer.fit_transform(all_sentences)
    except ValueError:
        return doc_score, 0.0, [], []

    ref_matrix = sent_matrix[:len(ref_sents)]
    sub_matrix = sent_matrix[len(ref_sents):]

    # Calculate similarity between each submitted sentence and all reference sentences
    sim_matrix = cosine_similarity(sub_matrix, ref_matrix)

    best_match_indices = sim_matrix.argmax(axis=1)
    best_scores = sim_matrix.max(axis=1)

    sentence_results = []
    flagged_matches = []

    for idx, (sub_sent, score_val) in enumerate(zip(sub_sents, best_scores)):
        score = float(score_val)
        ref_idx = int(best_match_indices[idx])
        matching_ref = ref_sents[ref_idx]
        is_flagged = score >= threshold

        item = {
            "sentence_index": idx + 1,
            "submitted_sentence": sub_sent,
            "similarity": score,
            "is_match": is_flagged,
            "matched_ref_sentence": matching_ref,
            "ref_index": ref_idx + 1
        }
        sentence_results.append(item)
        if is_flagged:
            flagged_matches.append(item)

    # Calculate estimated matched-text share (% of words in flagged submitted sentences)
    total_words = sum(len(s.split()) for s in sub_sents)
    matched_words = sum(len(m["submitted_sentence"].split()) for m in flagged_matches)
    matched_pct = (100.0 * matched_words / total_words) if total_words > 0 else 0.0

    return doc_score, matched_pct, sentence_results, flagged_matches


# Backward-compatible alias
analyze_plagiarism = analyze


# --- Sample Data for Quick Evaluation ---
SAMPLES = {
    "Sample 1: Plagiarized Passage (Direct & Near-Direct Copy)": {
        "ref": (
            "Natural Language Processing provides techniques to process and analyze textual information "
            "and identify similarities between documents. By applying text preprocessing, tokenization, "
            "n-gram extraction, and statistical text representation, a system can compare documents systematically. "
            "Term Frequency-Inverse Document Frequency is used to convert the text into numerical feature vectors. "
            "Cosine similarity measures the angle between vectors to produce a similarity score."
        ),
        "sub": (
            "Natural Language Processing provides techniques to process and analyze textual information "
            "and identify similarities between documents. Term Frequency-Inverse Document Frequency converts "
            "the text into numerical feature vectors to measure overlap. Cosine similarity calculates the angle "
            "between these vectors to produce a similarity score. Overall, this method makes document analysis fast."
        ),
        "desc": "Contains copied sentences alongside lightly rephrased statements."
    },
    "Sample 2: Paraphrased / Moderate Overlap": {
        "ref": (
            "Machine learning algorithms build mathematical models based on sample training data. "
            "Supervised learning requires labeled datasets to map inputs to corresponding target outputs. "
            "Common supervised applications include image classification and financial price forecasting."
        ),
        "sub": (
            "Supervised machine learning algorithms learn patterns from labeled training sets to predict outcomes. "
            "Deep neural networks have achieved state of the art performance on computer vision tasks. "
            "Image classification assigns meaningful category labels to unseen images."
        ),
        "desc": "Shares core vocabulary and concepts but uses altered grammatical structures."
    },
    "Sample 3: Original Content (Different Topics)": {
        "ref": (
            "Relational database management systems use structured query language to create and query schemas. "
            "Indexes such as B-trees improve query lookup speeds at the expense of additional storage and write overhead. "
            "Transaction ACID properties ensure reliability and data consistency during concurrent updates."
        ),
        "sub": (
            "Solar photovoltaic panels convert sunlight directly into clean electricity using silicon semiconductors. "
            "Modern solar installations are often paired with lithium battery storage systems to store excess generation. "
            "Renewable energy reduces greenhouse gas emissions and mitigates climate change."
        ),
        "desc": "Distinct domains with negligible lexical or phrase overlap."
    }
}


def load_sample_callback():
    choice = st.session_state.get("sample_selector", "-- Select a sample --")
    if choice in SAMPLES:
        st.session_state["ref_text"] = SAMPLES[choice]["ref"]
        st.session_state["sub_text"] = SAMPLES[choice]["sub"]
        st.session_state["auto_analyze"] = True


def clear_inputs_callback():
    st.session_state["ref_text"] = ""
    st.session_state["sub_text"] = ""
    st.session_state["sample_selector"] = "-- Select a sample --"
    st.session_state["analysis_done"] = False


# --- Main Application UI ---

def main():
    # Session state initialization
    if "ref_text" not in st.session_state:
        st.session_state["ref_text"] = ""
    if "sub_text" not in st.session_state:
        st.session_state["sub_text"] = ""
    if "auto_analyze" not in st.session_state:
        st.session_state["auto_analyze"] = False
    if "analysis_done" not in st.session_state:
        st.session_state["analysis_done"] = False

    # Sidebar Controls
    with st.sidebar:
        st.title("⚙️ Settings")
        st.markdown("Plagiarism detection configuration:")

        threshold = st.slider(
            "Match Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.40,
            step=0.05,
            help="Flag submitted sentences whose cosine similarity against any reference sentence is greater than or equal to this threshold. Default is 0.40."
        )

        st.divider()

        st.subheader("📚 Test Samples")
        st.selectbox(
            "Load a ready-to-test pair:",
            ["-- Select a sample --"] + list(SAMPLES.keys()),
            key="sample_selector",
            on_change=load_sample_callback
        )

        st.button("Clear All Texts", on_click=clear_inputs_callback, use_container_width=True)

        st.divider()

        with st.expander("📖 Algorithm Workflow", expanded=False):
            st.markdown("""
            **Processing Pipeline (from Project Report):**
            1. **Input:** Accept reference & submitted texts
            2. **Validation:** Ensure non-empty inputs
            3. **Sentence Split:** NLTK `sent_tokenize`
            4. **Preprocessing:** Lowercase, punctuation & whitespace cleaning
            5. **Feature Extraction:** Unigram & Bigram TF-IDF vectors
            6. **Cosine Similarity:**
               - Full document-to-document similarity
               - Sentence-level cross-comparison
            7. **Thresholding:** Flag candidate matching passages
            8. **Reporting:** Estimated matched-text share & highlight report
            """)

    # Main Header
    st.markdown("""
    <div class="glass-header-card">
        <div class="header-badge">✨ NLP Similarity Engine</div>
        <div class="main-header">📄 Text Plagiarism Detector</div>
        <div class="sub-header">An NLP-Based Text Similarity Detection System using N-Grams, TF-IDF Vectorization, and Cosine Similarity.</div>
    </div>
    """, unsafe_allow_html=True)

    # Document Input Section
    col_ref, col_sub = st.columns(2)

    with col_ref:
        st.markdown('<div class="glass-panel-title">📘 Reference Document (Source)</div>', unsafe_allow_html=True)
        ref_input = st.text_area(
            "Reference Document",
            key="ref_text",
            height=240,
            placeholder="Paste the original source or reference text here...",
            label_visibility="collapsed"
        )
        ref_words = len(ref_input.split()) if ref_input.strip() else 0
        st.markdown(
            f'<div class="glass-meta-info">Word count: <strong>{ref_words}</strong> &nbsp;|&nbsp; Characters: <strong>{len(ref_input)}</strong></div>',
            unsafe_allow_html=True
        )

    with col_sub:
        st.markdown('<div class="glass-panel-title">📝 Submitted Document (Candidate)</div>', unsafe_allow_html=True)
        sub_input = st.text_area(
            "Submitted Document",
            key="sub_text",
            height=240,
            placeholder="Paste the submitted document or passage to check for plagiarism here...",
            label_visibility="collapsed"
        )
        sub_words = len(sub_input.split()) if sub_input.strip() else 0
        st.markdown(
            f'<div class="glass-meta-info">Word count: <strong>{sub_words}</strong> &nbsp;|&nbsp; Characters: <strong>{len(sub_input)}</strong></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Action Bar
    btn_col, _ = st.columns([1, 4])
    with btn_col:
        analyze_clicked = st.button("🚀 Analyze Similarity", type="primary", use_container_width=True)

    should_run = analyze_clicked or st.session_state.get("auto_analyze", False)

    if should_run:
        # Reset auto_analyze trigger
        st.session_state["auto_analyze"] = False

        if not ref_input.strip() or not sub_input.strip():
            st.warning("⚠️ Please provide content for both the Reference Document and Submitted Document before analysis.")
            return

        with st.spinner("Analyzing text similarity, calculating TF-IDF vectors, and comparing sentences..."):
            doc_score, matched_pct, sentence_results, flagged_matches = analyze(
                ref_input, sub_input, threshold=threshold
            )
            st.session_state["analysis_done"] = True
            st.session_state["last_doc_score"] = doc_score
            st.session_state["last_matched_pct"] = matched_pct
            st.session_state["last_sentence_results"] = sentence_results
            st.session_state["last_flagged_matches"] = flagged_matches

    # Render results if analysis has been performed
    if st.session_state.get("analysis_done", False):
        doc_score = st.session_state["last_doc_score"]
        matched_pct = st.session_state["last_matched_pct"]
        sentence_results = st.session_state["last_sentence_results"]
        flagged_matches = st.session_state["last_flagged_matches"]

        st.markdown('<div style="margin: 28px 0 18px 0; border-top: 1px solid rgba(244, 114, 182, 0.35);"></div>', unsafe_allow_html=True)
        st.markdown('<h2 style="color: #9D174D; font-weight: 800; letter-spacing: -0.02em;">📊 Similarity Report</h2>', unsafe_allow_html=True)

        # Color-coded risk status
        if doc_score >= 0.70 or matched_pct >= 60.0:
            status_color = "#E11D48"
            risk_label = "HIGH SIMILARITY"
            risk_desc = "Significant textual overlap detected."
        elif doc_score >= 0.35 or matched_pct >= 25.0:
            status_color = "#D97706"
            risk_label = "MODERATE SIMILARITY"
            risk_desc = "Notable shared phrases detected."
        else:
            status_color = "#059669"
            risk_label = "LOW SIMILARITY"
            risk_desc = "Minimal shared n-grams or overlap."

        # Key Metrics Row
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Document Similarity</div>
                <div class="metric-value" style="color: {status_color};">{doc_score:.1%}</div>
                <div class="metric-sub">Cosine TF-IDF Vector Score</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Matched-Text Share</div>
                <div class="metric-value" style="color: {status_color};">{matched_pct:.1f}%</div>
                <div class="metric-sub">Words in Flagged Sentences</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            total_sub_sents = len(sentence_results)
            flagged_cnt = len(flagged_matches)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Sentences Flagged</div>
                <div class="metric-value" style="color: {status_color};">{flagged_cnt} / {total_sub_sents}</div>
                <div class="metric-sub">Threshold &ge; {threshold:.0%}</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Overall Status</div>
                <div class="metric-value" style="color: {status_color}; font-size: 1.25rem; line-height: 2.2rem;">{risk_label}</div>
                <div class="metric-sub">{risk_desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabbed Views for Analysis
        tab1, tab2, tab3 = st.tabs([
            "🔍 Highlighted Submitted Document",
            "📋 Candidate Matching Passages",
            "📊 Full Sentence Audit Table"
        ])

        with tab1:
            st.markdown("#### Highlighted Submitted Document")
            st.caption("Submitted sentences with similarity meeting or exceeding the threshold are highlighted in red.")

            highlighted_html = '<div class="highlight-card">'
            for item in sentence_results:
                escaped_text = html.escape(item["submitted_sentence"])
                if item["is_match"]:
                    score_badge = f"{item['similarity']:.1%}"
                    highlighted_html += (
                        f'<span class="match-sentence" title="Sentence #{item["sentence_index"]}: {score_badge} similarity to Reference #{item["ref_index"]}">'
                        f'{escaped_text} <sup style="font-size:0.75rem; font-weight:700; color: #BE185D;">[{score_badge}]</sup></span> '
                    )
                else:
                    highlighted_html += f'<span class="clean-sentence">{escaped_text}</span> '
            highlighted_html += '</div>'

            st.markdown(highlighted_html, unsafe_allow_html=True)

        with tab2:
            st.markdown(f"#### Candidate Matching Passages (Similarity &ge; {threshold:.0%})")
            if not flagged_matches:
                st.success(f"✅ No submitted sentences met or exceeded the {threshold:.0%} similarity threshold.")
            else:
                for idx, match in enumerate(flagged_matches, 1):
                    st.markdown(f"""
                    <div class="match-box">
                        <div style="font-weight: 700; color: #BE185D; margin-bottom: 8px; font-size: 0.95rem;">
                            🚩 Flagged Match #{idx} &mdash; Similarity: <span style="color: #E11D48;">{match['similarity']:.1%}</span>
                        </div>
                        <div style="margin-bottom: 10px;">
                            <strong style="color: #831843;">Submitted Sentence #{match['sentence_index']}:</strong><br>
                            <span style="color: #4C0519;">&ldquo;{html.escape(match['submitted_sentence'])}&rdquo;</span>
                        </div>
                        <div>
                            <strong style="color: #9D174D;">Most Similar Reference Sentence #{match['ref_index']}:</strong><br>
                            <span style="color: #6B7280;">&ldquo;{html.escape(match['matched_ref_sentence'])}&rdquo;</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with tab3:
            st.markdown("#### Complete Sentence Audit Table")
            st.caption("Detailed score breakdown for all sentences in the submitted document.")

            table_rows = []
            for item in sentence_results:
                table_rows.append({
                    "Sentence #": item["sentence_index"],
                    "Submitted Sentence": item["submitted_sentence"],
                    "Similarity Score": f"{item['similarity']:.1%}",
                    "Status": "⚠️ Flagged" if item["is_match"] else "✓ Passed",
                    "Closest Reference Match": item["matched_ref_sentence"]
                })

            st.dataframe(table_rows, use_container_width=True)

        # Methodology Disclaimer Note (From Project Report Section 1 & Section 5)
        st.markdown("""
        <div class="disclaimer-box">
            <strong>⚠️ Screening Tool Notice:</strong>
            Text similarity scores reflect lexical and n-gram overlap between the provided documents.
            A high similarity score does not automatically prove intentional plagiarism, as properly cited quotations,
            common technical terms, and standard phrases may also produce matches. This application is intended as a
            preliminary screening tool to assist human review.
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
