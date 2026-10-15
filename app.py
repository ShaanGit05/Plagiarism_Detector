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

# Custom Styling matching report mockups and modern clean aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .highlight-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 18px;
        line-height: 1.85;
        font-size: 1.02rem;
    }
    .match-sentence {
        background-color: #FEE2E2;
        border-bottom: 2px solid #EF4444;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 500;
        color: #991B1B;
    }
    .clean-sentence {
        padding: 2px 4px;
        color: #1F2937;
    }
    .match-box {
        background-color: #FFF7ED;
        border-left: 4px solid #F97316;
        padding: 14px 16px;
        margin-bottom: 12px;
        border-radius: 0 6px 6px 0;
    }
    .disclaimer-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        color: #475569;
        font-size: 0.88rem;
        margin-top: 20px;
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
    st.markdown('<div class="main-header">📄 Text Plagiarism Detector</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">An NLP-Based Text Similarity Detection System using N-Grams, TF-IDF Vectorization, and Cosine Similarity.</div>',
        unsafe_allow_html=True
    )

    # Document Input Section
    col_ref, col_sub = st.columns(2)

    with col_ref:
        st.markdown("### 📘 Reference Document (Source)")
        ref_input = st.text_area(
            "Reference Document",
            key="ref_text",
            height=240,
            placeholder="Paste the original source or reference text here...",
            label_visibility="collapsed"
        )
        ref_words = len(ref_input.split()) if ref_input.strip() else 0
        st.caption(f"Word count: **{ref_words}** | Characters: **{len(ref_input)}**")

    with col_sub:
        st.markdown("### 📝 Submitted Document (Candidate)")
        sub_input = st.text_area(
            "Submitted Document",
            key="sub_text",
            height=240,
            placeholder="Paste the submitted document or passage to check for plagiarism here...",
            label_visibility="collapsed"
        )
        sub_words = len(sub_input.split()) if sub_input.strip() else 0
        st.caption(f"Word count: **{sub_words}** | Characters: **{len(sub_input)}**")

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

        st.markdown("---")
        st.markdown("## 📊 Similarity Report")

        # Color-coded risk status
        if doc_score >= 0.70 or matched_pct >= 60.0:
            status_color = "#DC2626"
            risk_label = "HIGH SIMILARITY"
            risk_desc = "Significant textual overlap detected."
        elif doc_score >= 0.35 or matched_pct >= 25.0:
            status_color = "#D97706"
            risk_label = "MODERATE SIMILARITY"
            risk_desc = "Notable shared phrases detected."
        else:
            status_color = "#16A34A"
            risk_label = "LOW SIMILARITY"
            risk_desc = "Minimal shared n-grams or overlap."

        # Key Metrics Row
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Document Similarity</div>
                <div class="metric-value" style="color: {status_color};">{doc_score:.1%}</div>
                <small style="color: #64748B;">Cosine TF-IDF Vector Score</small>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Matched-Text Share</div>
                <div class="metric-value" style="color: {status_color};">{matched_pct:.1f}%</div>
                <small style="color: #64748B;">Words in Flagged Sentences</small>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            total_sub_sents = len(sentence_results)
            flagged_cnt = len(flagged_matches)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Sentences Flagged</div>
                <div class="metric-value" style="color: {status_color};">{flagged_cnt} / {total_sub_sents}</div>
                <small style="color: #64748B;">Threshold &ge; {threshold:.0%}</small>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Overall Status</div>
                <div class="metric-value" style="color: {status_color}; font-size: 1.2rem; line-height: 2.2rem;">{risk_label}</div>
                <small style="color: #64748B;">{risk_desc}</small>
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
                        f'{escaped_text} <sup style="font-size:0.75rem; font-weight:bold;">[{score_badge}]</sup></span> '
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
                        <div style="font-weight: 700; color: #C2410C; margin-bottom: 6px;">
                            Flagged Match #{idx} &mdash; Similarity: {match['similarity']:.1%}
                        </div>
                        <div style="margin-bottom: 8px;">
                            <strong>Submitted Sentence #{match['sentence_index']}:</strong><br>
                            <span style="color: #1F2937;">&ldquo;{html.escape(match['submitted_sentence'])}&rdquo;</span>
                        </div>
                        <div>
                            <strong>Most Similar Reference Sentence #{match['ref_index']}:</strong><br>
                            <span style="color: #4B5563;">&ldquo;{html.escape(match['matched_ref_sentence'])}&rdquo;</span>
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
