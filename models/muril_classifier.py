"""
MuRIL (Multilingual Representations for Indian Languages) classifier.
Google's MuRIL model is pre-trained on 17 Indian languages + English,
making it the best choice for Hindi/regional language news articles in CRIMSON-India.
Falls back to HindiBERT if MuRIL is unavailable.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
import numpy as np
import os
from utils.helpers import setup_logging, get_device, NewsDataset

logger = setup_logging()

# Primary: MuRIL covers 17 Indian languages. Fallback: Hindi BERT
MURIL_MODEL = "google/muril-base-cased"
HINDI_BERT_FALLBACK = "monsoon-nlp/hindi-bert"


class MuRILClassifier:
    """
    Fine-tuned MuRIL for multi-label crime/accident classification.
    Specifically designed for Hindi and other Indian language news articles.
    Architecture identical to TransformerClassifier but uses MuRIL weights.
    """

    def __init__(self, categories, model_name: str = MURIL_MODEL):
        self.categories = categories
        self.device = get_device()
        self.num_labels = len(categories)
        self.model_name = model_name

        logger.info(f"Initialising MuRIL ({self.model_name}) on {self.device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                problem_type="multi_label_classification",
            ).to(self.device)
        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}: {e}. Falling back to {HINDI_BERT_FALLBACK}")
            self.model_name = HINDI_BERT_FALLBACK
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                problem_type="multi_label_classification",
            ).to(self.device)

    def prepare_dataset(self, df, max_length: int = 128):
        texts = df["clean_text"].tolist()
        labels = df[self.categories].values.tolist()
        encodings = self.tokenizer(
            texts, truncation=True, padding=True, max_length=max_length
        )
        return NewsDataset(encodings, labels)

    def train(self, df, epochs: int = 3, batch_size: int = 8, lr: float = 2e-5):
        logger.info("Training MuRIL classifier...")
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

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg = total_loss / len(dataloader)
            logger.info(f"MuRIL Epoch {epoch + 1}/{epochs} | Loss: {avg:.4f}")

    def evaluate(self, df):
        logger.info("Evaluating MuRIL classifier...")
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

        print("\n--- MuRIL Evaluation ---")
        print(
            f"Precision: {precision:.4f} | Recall: {recall:.4f} | "
            f"F1: {f1:.4f} | Hamming Loss: {h_loss:.4f}"
        )
        return {"precision": precision, "recall": recall, "f1": f1, "hamming_loss": h_loss}

    def save(self, output_dir: str = "models/saved_muril"):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"MuRIL saved to {output_dir}")

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
