import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from nlp_pipeline import clean_text

MODEL_DIR = "models/bert_fake_news"
PRETRAINED_MODEL = "distilbert-base-multilingual-cased"
LABEL2ID = {"Fake": 0, "Real": 1, "Suspicious": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).astype(np.float32).mean().item()
    return {"accuracy": accuracy}


def train_model():
    df = pd.read_csv("data/dataset.csv")
    df = df.dropna(subset=["text", "label"])

    # Keep raw text for BERT and cleaned text for heuristic features if needed.
    texts = df["text"].astype(str).tolist()
    labels = df["label"].map(LABEL2ID).astype(int).tolist()

    X_train, X_eval, y_train, y_eval = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    train_dataset = FakeNewsDataset(X_train, y_train, tokenizer)
    eval_dataset = FakeNewsDataset(X_eval, y_eval, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        no_cuda=not torch.cuda.is_available(),
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    eval_results = trainer.evaluate()
    print(f"BERT model trained and saved to {MODEL_DIR}")
    print(f"Evaluation accuracy: {eval_results['eval_accuracy']:.4f}")


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_model()
