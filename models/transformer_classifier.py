"""
Transformer-based multi-label classifier (default: bert-base-multilingual-cased).
Uses HuggingFace Transformers + a simple linear classification head.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from utils.helpers import setup_logging

logger = setup_logging()


class TransformerClassifier:
    """Fine-tune a multilingual BERT model for multi-label crime classification."""

    def __init__(self, categories: list,
                 model_name: str = "bert-base-multilingual-cased",
                 save_dir: str = "models/saved_transformer",
                 max_length: int = 128):
        self.categories = [c for c in categories if c != 'non_crime']
        self.all_categories = categories
        self.model_name = model_name
        self.save_dir = save_dir
        self.max_length = max_length
        self.num_labels = len(self.categories)
        self._tokenizer = None
        self._model = None
        self._fitted = False

    # ------------------------------------------------------------------
    def _lazy_load(self):
        """Import heavy deps only when actually training/predicting."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                problem_type="multi_label_classification",
                ignore_mismatched_sizes=True,
            )
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)

    # ------------------------------------------------------------------
    def _extract_labels(self, df: pd.DataFrame):
        rows = []
        for _, row in df.iterrows():
            active = [c for c in self.categories if row.get(c, 0) == 1]
            rows.append(active if active else [self.categories[0]])
        return rows

    # ------------------------------------------------------------------
    def _labels_to_tensor(self, label_lists):
        import torch
        cat_index = {c: i for i, c in enumerate(self.categories)}
        Y = torch.zeros(len(label_lists), self.num_labels)
        for i, labs in enumerate(label_lists):
            for l in labs:
                if l in cat_index:
                    Y[i, cat_index[l]] = 1.0
        return Y

    # ------------------------------------------------------------------
    def train(self, train_df: pd.DataFrame, epochs: int = 2,
              batch_size: int = 16, text_col: str = "clean_text"):
        self._lazy_load()
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, TensorDataset

        logger.info(f"Fine-tuning {self.model_name} for {epochs} epoch(s) ...")
        texts = train_df[text_col].fillna("").tolist()
        enc = self._tokenizer(texts, truncation=True, padding=True,
                              max_length=self.max_length, return_tensors="pt")
        label_lists = self._extract_labels(train_df)
        Y = self._labels_to_tensor(label_lists)

        dataset = TensorDataset(enc["input_ids"], enc["attention_mask"], Y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = AdamW(self._model.parameters(), lr=2e-5)
        self._model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                ids, mask, labels = [b.to(self._device) for b in batch]
                optimizer.zero_grad()
                outputs = self._model(input_ids=ids, attention_mask=mask, labels=labels)
                outputs.loss.backward()
                optimizer.step()
                total_loss += outputs.loss.item()
            logger.info(f"  Epoch {epoch + 1}/{epochs} | loss={total_loss / len(loader):.4f}")

        self._fitted = True
        logger.info(f"Transformer ({self.model_name}) training complete.")

    # ------------------------------------------------------------------
    def evaluate(self, test_df: pd.DataFrame, batch_size: int = 32,
                 text_col: str = "clean_text", threshold: float = 0.5):
        if not self._fitted:
            logger.warning("Transformer not trained — skipping evaluation.")
            return
        import torch
        texts = test_df[text_col].fillna("").tolist()
        enc = self._tokenizer(texts, truncation=True, padding=True,
                              max_length=self.max_length, return_tensors="pt")
        self._model.eval()
        with torch.no_grad():
            logits = self._model(
                input_ids=enc["input_ids"].to(self._device),
                attention_mask=enc["attention_mask"].to(self._device),
            ).logits
        Y_pred = (torch.sigmoid(logits).cpu().numpy() >= threshold).astype(int)
        label_lists = self._extract_labels(test_df)
        cat_index = {c: i for i, c in enumerate(self.categories)}
        Y_true = np.zeros((len(label_lists), self.num_labels), dtype=int)
        for i, labs in enumerate(label_lists):
            for l in labs:
                if l in cat_index:
                    Y_true[i, cat_index[l]] = 1
        report = classification_report(Y_true, Y_pred, target_names=self.categories, zero_division=0)
        print(report)

    # ------------------------------------------------------------------
    def predict(self, texts: list, threshold: float = 0.5) -> list:
        if not self._fitted:
            raise RuntimeError("Model not trained yet.")
        import torch
        enc = self._tokenizer(texts, truncation=True, padding=True,
                              max_length=self.max_length, return_tensors="pt")
        self._model.eval()
        with torch.no_grad():
            logits = self._model(
                input_ids=enc["input_ids"].to(self._device),
                attention_mask=enc["attention_mask"].to(self._device),
            ).logits
        probs = torch.sigmoid(logits).cpu().numpy()
        results = []
        for row in probs:
            preds = [self.categories[i] for i, p in enumerate(row) if p >= threshold]
            results.append(preds if preds else ["non_crime"])
        return results

    # ------------------------------------------------------------------
    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self._model.save_pretrained(self.save_dir)
        self._tokenizer.save_pretrained(self.save_dir)
        logger.info(f"Transformer model saved to {self.save_dir}")
