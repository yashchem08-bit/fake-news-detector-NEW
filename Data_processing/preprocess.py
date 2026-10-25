"""
Member A — Data Engineer
=========================
Loads and MERGES multiple fake/real news datasets, cleans the text,
and splits the combined data into train/validation/test sets for
Member B (ML Engineer). Using more than one source (from different
publishers/writing styles) helps the model learn real vs fake
patterns instead of accidentally learning one dataset's quirks.

Supported sources — the script auto-detects whichever of these are
present under data/raw/ and merges them. You don't need all three.

    1. ISOT dataset     → data/raw/Fake.csv
                           data/raw/True.csv
    2. WELFake dataset  → data/raw/welfake/WELFake_Dataset.csv
    3. LIAR dataset     → data/raw/liar/train.tsv
                           data/raw/liar/test.tsv
                           data/raw/liar/valid.tsv

Download commands (run once — needs the Kaggle CLI + API token,
see the README for setup):

    kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p data/raw --unzip
    kaggle datasets download -d saurabhshahane/fake-news-classification -p data/raw/welfake --unzip
    kaggle datasets download -d doanquanvietnamca/liar-dataset -p data/raw/liar --unzip

After each download, run `ls data/raw` (Mac) or `dir data\raw`
(Windows) to confirm the actual filenames — some Kaggle mirrors
rename files slightly, in which case adjust the paths below to match.

Output:
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv

Run:
    python data_processing/preprocess.py
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

# LIAR's original 6-way truthfulness labels, collapsed to binary.
# "half-true" is a judgment call — treated as real here since it was
# graded as more true than false by the original human annotators.
LIAR_FAKE_LABELS = {"false", "barely-true", "pants-fire"}
LIAR_REAL_LABELS = {"true", "mostly-true", "half-true"}

LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job_title",
    "state", "party", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts", "pants_on_fire_counts",
    "context",
]


def ensure_nltk_data() -> None:
    """Download the NLTK stopwords list the first time this runs."""
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")


def load_isot() -> pd.DataFrame:
    """ISOT dataset: data/raw/Fake.csv + data/raw/True.csv"""
    fake_path = os.path.join(RAW_DIR, "Fake.csv")
    true_path = os.path.join(RAW_DIR, "True.csv")

    if not (os.path.exists(fake_path) and os.path.exists(true_path)):
        return pd.DataFrame(columns=["content", "label"])

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    fake_df["label"] = 1  # 1 = fake
    true_df["label"] = 0  # 0 = real
    df = pd.concat([fake_df, true_df], ignore_index=True)

    if "title" in df.columns and "text" in df.columns:
        df["content"] = df["title"].fillna("") + ". " + df["text"].fillna("")
    else:
        df["content"] = df.get("text", pd.Series(dtype=str)).fillna("")

    df = df[["content", "label"]].dropna()
    print(f"  ISOT dataset:    {len(df):>6} rows")
    return df


def load_welfake() -> pd.DataFrame:
    """WELFake dataset: data/raw/welfake/WELFake_Dataset.csv"""
    path = os.path.join(RAW_DIR, "welfake", "WELFake_Dataset.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["content", "label"])

    df = pd.read_csv(path)
    # WELFake convention is 0 = fake, 1 = real — flip so 1 = fake everywhere
    df["label"] = 1 - df["label"].astype(int)
    df["content"] = df["title"].fillna("") + ". " + df["text"].fillna("")

    df = df[["content", "label"]].dropna()
    print(f"  WELFake dataset: {len(df):>6} rows")
    return df


def load_liar() -> pd.DataFrame:
    """LIAR dataset: data/raw/liar/{train,test,valid}.tsv"""
    liar_dir = os.path.join(RAW_DIR, "liar")
    frames = []
    for filename in ("train.tsv", "test.tsv", "valid.tsv"):
        path = os.path.join(liar_dir, filename)
        if os.path.exists(path):
            frames.append(
                pd.read_csv(path, sep="\t", header=None, names=LIAR_COLUMNS)
            )

    if not frames:
        return pd.DataFrame(columns=["content", "label"])

    df = pd.concat(frames, ignore_index=True)
    df = df[df["label"].isin(LIAR_FAKE_LABELS | LIAR_REAL_LABELS)].copy()
    df["label"] = df["label"].apply(lambda x: 1 if x in LIAR_FAKE_LABELS else 0)
    df["content"] = df["statement"].fillna("")

    df = df[["content", "label"]].dropna()
    print(f"  LIAR dataset:    {len(df):>6} rows")
    return df


def load_raw_data() -> pd.DataFrame:
    print("Loading available datasets from data/raw/ ...")
    parts = [load_isot(), load_welfake(), load_liar()]
    parts = [p for p in parts if len(p) > 0]

    if not parts:
        raise FileNotFoundError(
            "No datasets found under data/raw/. See the download commands "
            "in this file's docstring for ISOT, WELFake, and LIAR."
        )

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset="content")
    print(f"\nCombined total (after de-duplication): {len(df)} rows")
    print(f"Label balance —\n{df['label'].value_counts().rename({0: 'real', 1: 'fake'})}")
    return df


def clean_text(text: str, stop_words: set) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)       # URLs
    text = re.sub(r"<.*?>", " ", text)                   # HTML tags
    text = re.sub(r"\d+", " ", text)                     # standalone numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t not in stop_words and len(t) > 2]
    return " ".join(tokens)


def main() -> None:
    ensure_nltk_data()
    stop_words = set(stopwords.words("english"))

    df = load_raw_data()

    print("\nCleaning text (this can take a few minutes on the full dataset)...")
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

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Saved cleaned splits to {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()
