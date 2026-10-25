"""
Member B — ML Engineer
OPTION 2: Cross-validation + hyperparameter tuning
=====================================================
Upgrades train_baseline.py by no longer guessing one setting per
model. Instead, for each algorithm this searches over several TF-IDF
and model settings using 5-fold cross-validation, and keeps whichever
combination scores best — squeezing out accuracy from the SAME data
you already have, no new downloads needed.

This will take noticeably longer to run than train_baseline.py
(it's training many more model variants internally), but needs no
new data and no GPU.

Run (after data_processing/preprocess.py or preprocess_multi_source.py
has produced the CSVs):
    python model/train_baseline_tuned.py
"""

import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR = os.path.join("model", "saved")

# Each entry pairs a classifier with a small hyperparameter grid to
# search. Keep grids small at first — they multiply fast (this one
# is already ~3-6x more training runs than train_baseline.py).
CANDIDATES = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=1000),
        "param_grid": {
            "clf__C": [0.1, 1, 10],
            "tfidf__max_features": [10000, 20000],
            "tfidf__ngram_range": [(1, 1), (1, 2)],
        },
    },
    "naive_bayes": {
        "estimator": MultinomialNB(),
        "param_grid": {
            "clf__alpha": [0.1, 0.5, 1.0],
            "tfidf__max_features": [10000, 20000],
            "tfidf__ngram_range": [(1, 1), (1, 2)],
        },
    },
    "linear_svm": {
        "estimator": LinearSVC(),
        "param_grid": {
            "clf__C": [0.1, 1, 10],
            "tfidf__max_features": [10000, 20000],
            "tfidf__ngram_range": [(1, 1), (1, 2)],
        },
    },
}


def load_split(name: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run a preprocessing script first."
        )
    return pd.read_csv(path).dropna(subset=["clean_content"])


def evaluate(model, X_test, y_test, name: str) -> float:
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary"
    )
    cm = confusion_matrix(y_test, preds)

    print(f"\n--- {name} (tuned) ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Confusion matrix:\n{cm}")
    return f1


def main() -> None:
    train_df = load_split("train")
    test_df = load_split("test")

    X_train_text = train_df["clean_content"]
    y_train = train_df["label"]
    X_test_text = test_df["clean_content"]
    y_test = test_df["label"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_name, best_search, best_f1 = None, None, -1.0

    for name, config in CANDIDATES.items():
        print(f"\nSearching hyperparameters for {name}...")

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", config["estimator"]),
        ])

        search = GridSearchCV(
            pipeline,
            param_grid=config["param_grid"],
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train_text, y_train)

        print(f"Best params for {name}: {search.best_params_}")
        print(f"Best cross-val F1: {search.best_score_:.4f}")

        f1 = evaluate(search.best_estimator_, X_test_text, y_test, name)
        if f1 > best_f1:
            best_name, best_search, best_f1 = name, search, f1

    print(f"\nBest overall model: {best_name} (test F1 = {best_f1:.4f})")
    print(f"Best params: {best_search.best_params_}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_pipeline = best_search.best_estimator_

    # Save the fitted TF-IDF vectorizer and classifier separately so
    # they stay compatible with model/predict.py, which loads them
    # as two files (same format train_baseline.py produces).
    joblib.dump(best_pipeline.named_steps["clf"], os.path.join(MODEL_DIR, "baseline_model.pkl"))
    joblib.dump(best_pipeline.named_steps["tfidf"], os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    with open(os.path.join(MODEL_DIR, "model_info.txt"), "w") as f:
        f.write(f"model_type={best_name}_tuned\n")
        f.write(f"f1_score={best_f1:.4f}\n")
        f.write(f"best_params={best_search.best_params_}\n")

    print(f"Saved tuned model + vectorizer to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
