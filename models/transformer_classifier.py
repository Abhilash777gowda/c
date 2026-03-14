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

class TransformerClassifier:
    def __init__(self, categories, model_name="bert-base-multilingual-cased"):
        self.categories = categories
        self.device = get_device()
        self.num_labels = len(categories)
        self.model_name = model_name
        
        logger.info(f"Initializing Transformer model ({model_name}) on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # HuggingFace handles problem_type internally adapting loss function specifically for BCEWithLogits
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels,
            problem_type="multi_label_classification"
        ).to(self.device)

    def prepare_dataset(self, df):
        texts = df['clean_text'].tolist()
        labels = df[self.categories].values.tolist()
        
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=128)
        dataset = NewsDataset(encodings, labels)
        return dataset

    def train(self, df, epochs=3, batch_size=8, lr=2e-5):
        logger.info("Training Transformer Model...")
        dataset = self.prepare_dataset(df)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = AdamW(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            logger.info(f"Transformer Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")

    def evaluate(self, df):
        logger.info("Evaluating Transformer Model...")
        dataset = self.prepare_dataset(df)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.sigmoid(logits) > 0.5
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='micro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        h_loss = hamming_loss(all_labels, all_preds)

        print("\n--- Transformer Evaluation ---")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | Hamming Loss: {h_loss:.4f}")
        return {"precision": precision, "recall": recall, "f1": f1, "hamming_loss": h_loss}

    def save(self, output_dir="models/saved_transformer"):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Transformer Model saved to {output_dir}")

    def predict(self, texts):
        self.model.eval()
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.sigmoid(logits) > 0.5
            
        preds_numpy = preds.cpu().numpy()
        
        results = []
        for pred_array in preds_numpy:
            tags = [self.categories[i] for i, val in enumerate(pred_array) if val]
            results.append(tags)
            
        return results
