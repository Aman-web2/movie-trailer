"""
sentiment.py
Sentiment scoring, classification, data processing, and summary statistics
using VADER (Valence Aware Dictionary and sEntiment Reasoner).
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Single shared analyzer instance (thread-safe for reads)
_analyzer = SentimentIntensityAnalyzer()


def score_comment(comment: str) -> dict:
    """
    Score a single comment and return label + compound score.

    Returns a dict with:
        - text:      original comment text
        - label:     'Positive' | 'Negative' | 'Neutral'
        - score:     compound float in [-1.0, 1.0]
    """
    text = str(comment).strip()
    if not text:
        return {"text": text, "label": "Neutral", "score": 0.0}

    compound = _analyzer.polarity_scores(text)["compound"]

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {"text": text, "label": label, "score": round(compound, 4)}


def analyze_comments(comments: list) -> list:
    """
    Analyze a list of comment strings.
    Returns a list of dicts (text, label, score).
    Skips None / empty entries safely.
    """
    results = []
    for c in comments:
        if c and str(c).strip():
            results.append(score_comment(c))
    return results


def build_summary(scored: list) -> dict:
    """
    Calculate summary statistics from a list of scored comment dicts.

    Returns:
        total, positive, negative, neutral counts,
        percentages, and average compound score.
    """
    if not scored:
        return {
            "total": 0,
            "positive": 0, "positive_pct": 0.0,
            "negative": 0, "negative_pct": 0.0,
            "neutral":  0, "neutral_pct":  0.0,
            "avg_score": 0.0,
            "dominant": "Neutral",
        }

    total    = len(scored)
    positive = sum(1 for s in scored if s["label"] == "Positive")
    negative = sum(1 for s in scored if s["label"] == "Negative")
    neutral  = sum(1 for s in scored if s["label"] == "Neutral")
    avg      = round(sum(s["score"] for s in scored) / total, 4)

    counts = {"Positive": positive, "Negative": negative, "Neutral": neutral}
    dominant = max(counts, key=counts.get)

    return {
        "total":        total,
        "positive":     positive,
        "positive_pct": round(positive / total * 100, 1),
        "negative":     negative,
        "negative_pct": round(negative / total * 100, 1),
        "neutral":      neutral,
        "neutral_pct":  round(neutral  / total * 100, 1),
        "avg_score":    avg,
        "dominant":     dominant,
    }
