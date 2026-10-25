"""
Member A — Data Engineer
OPTION 1: Multi-source preprocessing
======================================
Combines several Kaggle fake-news datasets (instead of just one) so
the model sees more variety in writing style, topic, and source —
this reduces the risk of it learning "which outlet this came from"
instead of "is this actually fake."

Supported sources (use as many as you've downloaded — the script
skips any that aren't present):

    1. ISOT (Fake.csv + True.csv)        — data/raw/Fake.csv, data/raw/True.csv
    2. WELFake (WELFake_Dataset.csv)     — data/raw/WELFake_Dataset.csv
    3. LIAR (train.tsv/test.tsv/valid.tsv) — data/raw/liar/*.tsv

Download each with the Kaggle CLI, e.g.:
    kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p data/raw --unzip
    kaggle datasets download -d saurabhshahane/fake-news-classification -p data/raw --unzip
    kaggle datasets download -d doanquanvietnamca/liar-dataset -p data/raw/liar --unzip

Column names on Kaggle can drift between dataset versions — if a
loader below throws a KeyError, open the CSV/TSV once and adjust the
column names in that loader to match what you actually see.

Output (same shape as the single-source preprocess.py, just with
more rows in it):
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv

Run:
    python data_processing/preprocess_multi_source.py
"""

import os
import re
import string

import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

# label convention used throughout this whole project: 1 = fake, 0 = real


def ensure_nltk_data() -> None:
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")


# ── Per-dataset loaders ────────────────────────────────────────────
# Each one returns a DataFrame with exactly two columns: content, label.
# A loader returns None (and prints why) if its files aren't there,
# so you can add datasets to your raw/ folder one at a time.

def load_isot() -> pd.DataFrame | None:
    fake_path = os.path.join(RAW_DIR, "Fake.csv")
    true_path = os.path.join(RAW_DIR, "True.csv")
    if not (os.path.exists(fake_path) and os.path.exists(true_path)):
        print("  [skip] ISOT: Fake.csv / True.csv not found in data/raw/")
        return None

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    fake_df["label"] = 1
    true_df["label"] = 0
    df = pd.concat([fake_df, true_df], ignore_index=True)

    df["content"] = df.get("title", "").fillna("") + ". " + df.get("text", "").fillna("")
    print(f"  [ok] ISOT: {len(df)} rows")
    return df[["content", "label"]]


def load_welfake() -> pd.DataFrame | None:
    path = os.path.join(RAW_DIR, "WELFake_Dataset.csv")
    if not os.path.exists(path):
        print("  [skip] WELFake: WELFake_Dataset.csv not found in data/raw/")
        return None

    df = pd.read_csv(path)
    # WELFake's public release uses label 0 = fake, 1 = real — the
    # opposite of this project's convention — so we flip it here.
    df["label"] = 1 - df["label"].astype(int)
    df["content"] = df.get("title", "").fillna("") + ". " + df.get("text", "").fillna("")
    print(f"  [ok] WELFake: {len(df)} rows")
    return df[["content", "label"]]


def load_liar() -> pd.DataFrame | None:
    liar_dir = os.path.join(RAW_DIR, "liar")
    files = ["train.tsv", "test.tsv", "valid.tsv"]
    paths = [os.path.join(liar_dir, f) for f in files]
    if not any(os.path.exists(p) for p in paths):
        print("  [skip] LIAR: no .tsv files found in data/raw/liar/")
        return None

    columns = [
        "id", "label", "statement", "subject", "speaker", "job", "state",
        "party", "barely_true_c", "false_c", "half_true_c", "mostly_true_c",
        "pants_fire_c", "context",
    ]

    frames = []
    for p in paths:
        if os.path.exists(p):
            frames.append(pd.read_csv(p, sep="\t", header=None, names=columns))
    df = pd.concat(frames, ignore_index=True)

    # LIAR has 6 truthfulness levels — collapse to binary. "pants-fire",
    # "false", and "barely-true" become fake; the rest become real.
    fake_labels = {"pants-fire", "false", "barely-true"}
    df["label"] = df["label"].apply(lambda x: 1 if x in fake_labels else 0)
    df["content"] = df["statement"].fillna("")
    print(f"  [ok] LIAR: {len(df)} rows")
    return df[["content", "label"]]


LOADERS = [load_isot, load_welfake, load_liar]


def clean_text(text: str, stop_words: set) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t not in stop_words and len(t) > 2]
    return " ".join(tokens)


def main() -> None:
    ensure_nltk_data()
    stop_words = set(stopwords.words("english"))

    print("Loading available sources...")
    frames = [loader() for loader in LOADERS]
    frames = [f for f in frames if f is not None]

    if not frames:
        raise FileNotFoundError(
            "No datasets found in data/raw/. Add at least one of: "
            "Fake.csv+True.csv, WELFake_Dataset.csv, or liar/*.tsv."
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["content"])
    df = df[df["content"].str.strip().str.len() > 0]
    df = df.drop_duplicates(subset="content")
    print(f"\nCombined total: {len(df)} rows across {len(frames)} source(s).")
    print(f"Class balance -> fake: {(df['label'] == 1).sum()}, real: {(df['label'] == 0).sum()}")

    print("\nCleaning text (can take a few minutes on a large combined set)...")
    df["clean_content"] = df["content"].apply(lambda x: clean_text(x, stop_words))
    df = df[df["clean_content"].str.len() > 0]

    print("Splitting into train/val/test (70/15/15, stratified)...")
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Saved cleaned splits to {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()
