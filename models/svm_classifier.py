"""
Baseline SVM + TF-IDF multi-label classifier for CRIMSON-India.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from utils.helpers import setup_logging

logger = setup_logging()


class BaselineSVM:
    """Multi-label SVM classifier using TF-IDF features."""

    def __init__(self, categories: list, save_path: str = "models/baseline_svm.pkl"):
        self.categories = [c for c in categories if c != 'non_crime']
        self.all_categories = categories
        self.save_path = save_path
        self.vectorizer = TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents='unicode',
        )
        self.clf = OneVsRestClassifier(LinearSVC(max_iter=2000, C=1.0))
        self.mlb = MultiLabelBinarizer(classes=self.categories)
        self._fitted = False

    # ------------------------------------------------------------------
    def _extract_labels(self, df: pd.DataFrame):
        """Return list-of-lists of active category labels per row."""
        rows = []
        for _, row in df.iterrows():
            active = [c for c in self.categories if row.get(c, 0) == 1]
            if not active:
                active = ['non_crime'] if 'non_crime' in self.all_categories else []
            rows.append(active)
        return rows

    # ------------------------------------------------------------------
    def train(self, train_df: pd.DataFrame, text_col: str = "clean_text"):
        logger.info("Training SVM baseline ...")
        texts = train_df[text_col].fillna("").tolist()
        X = self.vectorizer.fit_transform(texts)

        label_lists = self._extract_labels(train_df)
        # Filter to only categories tracked by mlb
        label_lists = [[l for l in ls if l in self.categories] or ['theft'] for ls in label_lists]
        Y = self.mlb.fit_transform(label_lists)

        self.clf.fit(X, Y)
        self._fitted = True
        logger.info("SVM training complete.")

    # ------------------------------------------------------------------
    def evaluate(self, test_df: pd.DataFrame, text_col: str = "clean_text"):
        if not self._fitted:
            logger.warning("SVM not trained — skipping evaluation.")
            return
        texts = test_df[text_col].fillna("").tolist()
        X = self.vectorizer.transform(texts)
        Y_pred = self.clf.predict(X)

        label_lists = self._extract_labels(test_df)
        label_lists = [[l for l in ls if l in self.categories] or ['theft'] for ls in label_lists]
        Y_true = self.mlb.transform(label_lists)

        report = classification_report(Y_true, Y_pred, target_names=self.mlb.classes_, zero_division=0)
        print(report)
        logger.info("SVM evaluation complete.")

    # ------------------------------------------------------------------
    def predict(self, texts: list) -> list:
        """Return predicted label lists for a list of raw text strings."""
        if not self._fitted:
            raise RuntimeError("Model not trained yet.")
        X = self.vectorizer.transform(texts)
        Y = self.clf.predict(X)
        return self.mlb.inverse_transform(Y)

    # ------------------------------------------------------------------
    def save(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "clf": self.clf, "mlb": self.mlb}, f)
        logger.info(f"SVM model saved to {self.save_path}")

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, categories: list, save_path: str = "models/baseline_svm.pkl"):
        obj = cls(categories, save_path)
        with open(save_path, "rb") as f:
            data = pickle.load(f)
        obj.vectorizer = data["vectorizer"]
        obj.clf = data["clf"]
        obj.mlb = data["mlb"]
        obj._fitted = True
        return obj
