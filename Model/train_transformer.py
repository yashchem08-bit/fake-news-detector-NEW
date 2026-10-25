"""
Member B — ML Engineer (Phase 2 / OPTIONAL upgrade)
======================================================
Fine-tunes a lightweight transformer (DistilBERT) on the same dataset
for higher accuracy than the TF-IDF baseline. The app works fully
without this file — only run it if you want to try to beat the
baseline's F1-score.

Requires the optional deps in requirements.txt:
    transformers
    torch

This is slow on a CPU. Recommended: run on a machine or notebook with
a GPU (e.g. Google Colab).

Run (after data_processing/preprocess.py has produced the CSVs):
    python model/train_transformer.py
"""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR = os.path.join("model", "saved", "transformer")
BASE_MODEL = "distilbert-base-uncased"


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int = 256):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True, max_length=max_len
        )
        self.labels = list(labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def main() -> None:
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv")).dropna(
        subset=["clean_content"]
    )
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv")).dropna(
        subset=["clean_content"]
    )

    # Transformers handle raw (uncleaned) text fine — prefer it if available
    text_col = "content" if "content" in train_df.columns else "clean_content"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2
    )

    train_dataset = NewsDataset(train_df[text_col], train_df["label"], tokenizer)
    val_dataset = NewsDataset(val_df[text_col], val_df["label"], tokenizer)

    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "checkpoints"),
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Saved fine-tuned transformer to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
