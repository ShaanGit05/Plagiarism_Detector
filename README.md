# Plagiarism Detector: An NLP-Based Text Similarity Detection System

This repository contains the text plagiarism detector website built according to the specifications in the project report (*"Plagiarism Detector: An NLP-Based Text Similarity Detection System"*).

---

## 📌 Overview

The system provides an automated text comparison tool to identify potential plagiarism between a **Reference Document** (the original source) and a **Submitted Document** (the document under review).

Instead of relying on simple exact word matching, the system leverages:
1. **Text Preprocessing & Normalization:** Lowercasing, punctuation stripping, and whitespace collapse.
2. **Sentence Segmentation:** Tokenization using NLTK `sent_tokenize`.
3. **N-Gram Feature Extraction:** Captures single words (unigrams) and adjacent word pairs (bigrams) using `ngram_range=(1, 2)`.
4. **TF-IDF Vectorization:** Represents texts as numerical vectors with Scikit-learn's smoothed Inverse Document Frequency.
5. **Cosine Similarity Measurement:**
   - **Document-Level Similarity:** Overall degree of overlap between complete texts.
   - **Sentence-Level Cross-Comparison:** Compares each submitted sentence against all reference sentences.
6. **Configurable Thresholding:** User-adjustable similarity slider (default `0.40`) to flag candidate matching passages.
7. **Highlighting & Metric Reporting:**
   - Overall Document Similarity (%)
   - Estimated Matched-Text Share (%)
   - Visual passage highlighting
   - Side-by-side comparison with the closest matching source sentence

---

## 🚀 Quick Start

### 1. Run the Web Application
Simply double-click `run_app.bat` or run the following command in PowerShell / Command Prompt:

```powershell
.\.venv\Scripts\streamlit run app.py
```

The application will open automatically in your browser at:
`http://localhost:8501`

---

## 🛠️ Project Structure

```
D:\NLP\
├── app.py                     # Streamlit web application & NLP pipeline
├── test_detector.py           # Verification and unit tests
├── requirements.txt           # Python dependencies
├── run_app.bat                # 1-click Windows launch script
└── Docs Plagiarism_Detector_Report_261015105605.pdf   # Academic project report
```

---

## 💡 How to Use the App

1. **Enter Reference Text:** Paste the original document into the left text box.
2. **Enter Submitted Text:** Paste the candidate document into the right text box.
3. *(Optional)* Or select a preloaded sample from the sidebar ("Sample 1: Plagiarized Passage", "Sample 2: Moderate Similarity", "Sample 3: Original Content").
4. **Adjust Threshold:** Set the match threshold slider in the sidebar (default is `0.40`).
5. **Click "Analyze Similarity":**
   - View the **Document Similarity** and **Estimated Matched-Text Share** metrics.
   - Inspect the **Highlighted Submitted Document** tab to see matching sentences marked in red.
   - Check the **Candidate Matching Passages** tab to compare flagged sentences directly against their closest reference matches.
   - Review the **Complete Sentence Audit Table** for a full breakdown.

---

## ⚠️ Screening Tool Notice

As noted in the project report, text similarity indicates lexical and n-gram overlap. A high score suggests passages warranting closer inspection, but does not replace human editorial judgment.
