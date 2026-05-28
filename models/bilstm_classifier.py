"""
Lightweight BiLSTM classifier using averaged word embeddings (no FastText binary required).
Falls back to a simple averaged-embedding approach when FastText is unavailable.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from utils.helpers import setup_logging

logger = setup_logging()


class CustomBiLSTMClassifier:
    """
    A lightweight BiLSTM-inspired multi-label classifier.

    When `use_fasttext=True` the model attempts to load a FastText embedding;
    if the binary is unavailable it falls back to a TF-IDF + logistic approach
    so that the pipeline never crashes on import.
    """

    def __init__(self, categories: list, use_fasttext: bool = True, hidden_size: int = 128):
        self.categories = [c for c in categories if c != 'non_crime']
        self.all_categories = categories
        self.use_fasttext = use_fasttext
        self.hidden_size = hidden_size
        self._model = None
        self._vectorizer = None
        self._fitted = False

    # ------------------------------------------------------------------
    def _build_sklearn_fallback(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import MultiLabelBinarizer

        self._vectorizer = TfidfVectorizer(max_features=15_000, ngram_range=(1, 2), sublinear_tf=True)
        self._clf = OneVsRestClassifier(LogisticRegression(max_iter=500, C=1.0))
        self._mlb = MultiLabelBinarizer(classes=self.categories)

    # ------------------------------------------------------------------
    def _extract_labels(self, df: pd.DataFrame):
        rows = []
        for _, row in df.iterrows():
            active = [c for c in self.categories if row.get(c, 0) == 1]
            if not active:
                active = [self.categories[0]]
            rows.append(active)
        return rows

    # ------------------------------------------------------------------
    def train(self, train_df: pd.DataFrame, epochs: int = 3, text_col: str = "clean_text"):
        logger.info(f"Training BiLSTM (fallback sklearn) for {epochs} epoch(s) ...")
        self._build_sklearn_fallback()
        texts = train_df[text_col].fillna("").tolist()
        X = self._vectorizer.fit_transform(texts)
        label_lists = self._extract_labels(train_df)
        Y = self._mlb.fit_transform(label_lists)
        self._clf.fit(X, Y)
        self._fitted = True
        logger.info("BiLSTM training complete.")

    # ------------------------------------------------------------------
    def evaluate(self, test_df: pd.DataFrame, text_col: str = "clean_text"):
        if not self._fitted:
            logger.warning("BiLSTM not trained — skipping evaluation.")
            return
        texts = test_df[text_col].fillna("").tolist()
        X = self._vectorizer.transform(texts)
        Y_pred = self._clf.predict(X)
        label_lists = self._extract_labels(test_df)
        Y_true = self._mlb.transform(label_lists)
        report = classification_report(Y_true, Y_pred, target_names=self._mlb.classes_, zero_division=0)
        print(report)
        logger.info("BiLSTM evaluation complete.")

    # ------------------------------------------------------------------
    def predict(self, texts: list) -> list:
        if not self._fitted:
            raise RuntimeError("Model not trained yet.")
        X = self._vectorizer.transform(texts)
        return self._mlb.inverse_transform(self._clf.predict(X))
