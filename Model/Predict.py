"""
Member B — ML Engineer
=======================
Reusable prediction function. Loads the saved TF-IDF vectorizer and
classifier once, then exposes predict_news() for the backend
(Member C) to import and call.
"""

import math
import os
import re
import string

import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved")

_model = None
_vectorizer = None

LABELS = {0: "REAL", 1: "FAKE"}


def _load() -> None:
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        model_path = os.path.join(MODEL_DIR, "baseline_model.pkl")
        vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
        if not (os.path.exists(model_path) and os.path.exists(vec_path)):
            raise FileNotFoundError(
                "Trained model not found. Run model/train_baseline.py first."
            )
        _model = joblib.load(model_path)
        _vectorizer = joblib.load(vec_path)


def _clean_text(text: str) -> str:
    """Mirrors the cleaning logic in data_processing/preprocess.py so
    inference-time text is processed the same way as training data."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def predict_news(text: str) -> dict:
    """
    Takes raw article text and returns:
        {"label": "REAL" | "FAKE", "confidence": float between 0 and 1}
    """
    _load()

    if not text or not text.strip():
        raise ValueError("Input text is empty.")

    cleaned = _clean_text(text)
    features = _vectorizer.transform([cleaned])
    pred = int(_model.predict(features)[0])

    if hasattr(_model, "predict_proba"):
        confidence = float(_model.predict_proba(features)[0][pred])
    elif hasattr(_model, "decision_function"):
        # LinearSVC has no predict_proba — turn its margin into a
        # 0-1 pseudo-confidence with a sigmoid.
        score = _model.decision_function(features)[0]
        confidence = 1 / (1 + math.exp(-abs(score)))
    else:
        confidence = 0.5

    return {"label": LABELS[pred], "confidence": round(confidence, 4)}


if __name__ == "__main__":
    sample = "Scientists confirm the moon is actually made of cheese, NASA admits."
    print(predict_news(sample))
