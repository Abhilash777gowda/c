import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
import pickle
from utils.helpers import setup_logging

logger = setup_logging()

class BaselineSVM:
    def __init__(self, categories):
        self.categories = categories
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        # MultiOutputClassifier behavior achieved via OneVsRestClassifier for multi-label SVM
        self.classifier = OneVsRestClassifier(LinearSVC(class_weight='balanced'))

    def train(self, df):
        logger.info("Training Baseline Model (TF-IDF + SVM)...")
        X = self.vectorizer.fit_transform(df['clean_text'])
        y = df[self.categories].values
        
        self.classifier.fit(X, y)
        logger.info("Baseline SVM training completed.")

    def evaluate(self, df):
        logger.info("Evaluating Baseline SVM...")
        X = self.vectorizer.transform(df['clean_text'])
        y_true = df[self.categories].values
        
        y_pred = self.classifier.predict(X)
        
        report = classification_report(y_true, y_pred, target_names=self.categories, zero_division=0)
        h_loss = hamming_loss(y_true, y_pred)
        
        logger.info(f"Hamming Loss: {h_loss}")
        print("\n--- Baseline SVM Evaluation ---")
        print(f"Hamming Loss: {h_loss:.4f}")
        print(report)
        return {"hamming_loss": h_loss, "report": report}

    def save(self, filepath="models/baseline_svm.pkl"):
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({'vectorizer': self.vectorizer, 'classifier': self.classifier}, f)
            logger.info(f"Baseline SVM saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save SVM: {e}")
