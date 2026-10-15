"""
Unit tests for Plagiarism Detector NLP logic
"""

from app import clean_text, split_sentences, analyze_plagiarism


def test_clean_text():
    raw = "Hello, WORLD! This is a test... with 123 numbers & symbols #$%"
    cleaned = clean_text(raw)
    assert "world" in cleaned
    assert "hello" in cleaned
    assert "123" in cleaned
    assert "!" not in cleaned
    assert "," not in cleaned
    assert "#" not in cleaned


def test_split_sentences():
    text = "This is the first sentence. Here is the second sentence! And a third?"
    sents = split_sentences(text)
    assert len(sents) == 3
    assert sents[0] == "This is the first sentence."
    assert sents[1] == "Here is the second sentence!"
    assert sents[2] == "And a third?"


def test_exact_plagiarism():
    ref = "Natural language processing enables computers to understand human language."
    sub = "Natural language processing enables computers to understand human language."
    doc_score, matched_pct, sentence_results, flagged_matches = analyze_plagiarism(ref, sub, threshold=0.40)
    
    assert doc_score >= 0.99
    assert matched_pct == 100.0
    assert len(flagged_matches) == 1
    assert flagged_matches[0]["similarity"] >= 0.99


def test_unrelated_text():
    ref = "Quantum computing uses qubits and quantum superposition to perform calculations."
    sub = "Baking chocolate chip cookies requires flour, butter, sugar, and eggs."
    doc_score, matched_pct, sentence_results, flagged_matches = analyze_plagiarism(ref, sub, threshold=0.40)
    
    assert doc_score == 0.0 or doc_score < 0.1
    assert matched_pct == 0.0
    assert len(flagged_matches) == 0


def test_partial_plagiarism():
    ref = (
        "Supervised learning algorithms require labeled datasets. "
        "They map inputs to outputs based on historical patterns."
    )
    sub = (
        "Supervised learning algorithms require labeled datasets. "
        "Cooking pizza involves making dough and preheating an oven."
    )
    doc_score, matched_pct, sentence_results, flagged_matches = analyze_plagiarism(ref, sub, threshold=0.40)
    
    assert len(sentence_results) == 2
    assert len(flagged_matches) == 1
    assert flagged_matches[0]["is_match"] is True
    assert 0 < matched_pct < 100


def test_empty_input():
    doc_score, matched_pct, sentence_results, flagged_matches = analyze_plagiarism("", "", threshold=0.40)
    assert doc_score == 0.0
    assert matched_pct == 0.0
    assert len(sentence_results) == 0
    assert len(flagged_matches) == 0


if __name__ == "__main__":
    test_clean_text()
    test_split_sentences()
    test_exact_plagiarism()
    test_unrelated_text()
    test_partial_plagiarism()
    test_empty_input()
    print("All tests passed successfully!")
