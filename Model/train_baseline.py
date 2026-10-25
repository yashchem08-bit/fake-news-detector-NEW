"""
Member B — ML Engineer
=======================
Trains and evaluates baseline text classifiers (Logistic Regression,
Naive Bayes, Linear SVM) on the cleaned data produced by Member A,
picks the best performer by F1-score, and saves the model + TF-IDF
vectorizer for the backend (Member C) to load.

Run (after data_processing/preprocess.py has produced the CSVs):
    python model/train_baseline.py
"""

import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR = os.path.join("model", "saved")

CANDIDATES = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "naive_bayes": MultinomialNB(),
    "linear_svm": LinearSVC(),
}


def load_split(name: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run data_processing/preprocess.py first."
        )
    return pd.read_csv(path).dropna(subset=["clean_content"])


def evaluate(model, X_test, y_test, name: str) -> float:
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary"
    )
    cm = confusion_matrix(y_test, preds)

    print(f"\n--- {name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Confusion matrix:\n{cm}")
    return f1


def main() -> None:
    train_df = load_split("train")
    test_df = load_split("test")

    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df["clean_content"])
    X_test = vectorizer.transform(test_df["clean_content"])
    y_train = train_df["label"]
    y_test = test_df["label"]

    best_name, best_model, best_f1 = None, None, -1.0

    for name, model in CANDIDATES.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        f1 = evaluate(model, X_test, y_test, name)
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, model, f1

    print(f"\nBest model: {best_name} (F1 = {best_f1:.4f})")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, "baseline_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    with open(os.path.join(MODEL_DIR, "model_info.txt"), "w") as f:
        f.write(f"model_type={best_name}\nf1_score={best_f1:.4f}\n")

    print(f"Saved model + vectorizer to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
