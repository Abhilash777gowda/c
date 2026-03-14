"""
Fast inference module using the pre-trained SVM model.
Used in the dashboard for real-time article classification without GPU.
"""
import os
import pickle
import pandas as pd
from utils.helpers import setup_logging

logger = setup_logging()

CRIME_CATEGORIES = ['theft', 'assault', 'accident', 'drug_crime', 'cybercrime', 'non_crime']
SVM_MODEL_PATH = "models/baseline_svm.pkl"


def load_svm_model():
    """Load the saved TF-IDF + SVM pipeline from disk."""
    if not os.path.exists(SVM_MODEL_PATH):
        logger.warning(f"SVM model not found at {SVM_MODEL_PATH}. Run main.py --use-synthetic first.")
        return None, None
    try:
        with open(SVM_MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        logger.info("SVM model loaded for real-time inference.")
        return bundle["vectorizer"], bundle["classifier"]
    except Exception as e:
        logger.error(f"Failed to load SVM model: {e}")
        return None, None


def classify_articles(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """
    Classify articles using the saved SVM model.
    Adds multi-hot crime category columns to the DataFrame.

    Args:
        df: DataFrame with a `clean_text` column
        text_col: Name of the column holding cleaned article text

    Returns:
        DataFrame with added columns for each crime category (0 or 1)
    """
    vectorizer, classifier = load_svm_model()

    # Initialise all category columns to 0 as default
    for cat in CRIME_CATEGORIES:
        df[cat] = 0

    if vectorizer is None or classifier is None:
        logger.warning("Classifier unavailable — articles will have no labels.")
        return df

    try:
        texts = df[text_col].fillna("").tolist()
        X = vectorizer.transform(texts)
        preds = classifier.predict(X)  # shape: (n_samples, n_categories)

        for i, cat in enumerate(CRIME_CATEGORIES):
            df[cat] = preds[:, i]

        logger.info(f"Classified {len(df)} articles with SVM model.")
    except Exception as e:
        logger.error(f"Classification failed: {e}")

    return df
