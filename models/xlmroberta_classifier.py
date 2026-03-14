"""
XLM-RoBERTa multi-label classifier.
Primary transformer model specified in CRIMSON-India architecture.
Handles both English and Indian-language news articles (multilingual).
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
import numpy as np
import os
from utils.helpers import setup_logging, get_device, NewsDataset

logger = setup_logging()


class XLMRoBertaClassifier:
    """
    Fine-tuned XLM-RoBERTa-base for multi-label crime/accident classification.
    Preferred over mBERT for Indian-language articles due to superior multilingual
    pretraining on 100 languages including Hindi, Bengali, Tamil, etc.
    """

    MODEL_NAME = "xlm-roberta-base"

    def __init__(self, categories, model_name: str = None):
        self.categories = categories
        self.device = get_device()
        self.num_labels = len(categories)
        self.model_name = model_name or self.MODEL_NAME

        logger.info(f"Initialising XLM-RoBERTa ({self.model_name}) on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="multi_label_classification",
        ).to(self.device)

    # ── Dataset preparation ─────────────────────────────────────────────────
    def prepare_dataset(self, df, max_length: int = 128):
        texts = df["clean_text"].tolist()
        labels = df[self.categories].values.tolist()
        encodings = self.tokenizer(
            texts, truncation=True, padding=True, max_length=max_length
        )
        return NewsDataset(encodings, labels)

    # ── Training ─────────────────────────────────────────────────────────────
    def train(self, df, epochs: int = 3, batch_size: int = 8, lr: float = 2e-5):
        logger.info("Training XLM-RoBERTa...")
        dataset = self.prepare_dataset(df)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in dataloader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg = total_loss / len(dataloader)
            logger.info(
                f"XLM-RoBERTa Epoch {epoch + 1}/{epochs} | Loss: {avg:.4f}"
            )

    # ── Evaluation ───────────────────────────────────────────────────────────
    def evaluate(self, df):
        logger.info("Evaluating XLM-RoBERTa...")
        dataset = self.prepare_dataset(df)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]

                outputs = self.model(input_ids, attention_mask=attention_mask)
                preds = torch.sigmoid(outputs.logits) > 0.5

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        precision = precision_score(all_labels, all_preds, average="micro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="micro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
        h_loss = hamming_loss(all_labels, all_preds)

        print("\n--- XLM-RoBERTa Evaluation ---")
        print(
            f"Precision: {precision:.4f} | Recall: {recall:.4f} | "
            f"F1: {f1:.4f} | Hamming Loss: {h_loss:.4f}"
        )
        return {"precision": precision, "recall": recall, "f1": f1, "hamming_loss": h_loss}

    # ── Persistence ──────────────────────────────────────────────────────────
    def save(self, output_dir: str = "models/saved_xlmroberta"):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"XLM-RoBERTa saved to {output_dir}")

    def predict(self, texts: list) -> list:
        self.model.eval()
        encodings = self.tokenizer(
            texts, truncation=True, padding=True, max_length=128, return_tensors="pt"
        )
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            preds = torch.sigmoid(outputs.logits) > 0.5

        results = []
        for pred_array in preds.cpu().numpy():
            tags = [self.categories[i] for i, val in enumerate(pred_array) if val]
            results.append(tags if tags else ["non_crime"])
        return results
