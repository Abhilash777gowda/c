import os
import pickle
import pandas as pd
import streamlit as st
from utils.helpers import setup_logging
from transformers import pipeline
import torch

logger = setup_logging()

CRIME_CATEGORIES = [
    'murder', 'rape', 'kidnapping', 'sexual_harassment', 'crime_against_children', 
    'theft', 'burglary', 'robbery', 'fraud_cheating', 'accident', 'non_crime'
]

# Mapping keys to descriptive labels for better Zero-Shot inference
DESCRIPTIVE_LABELS = {
    'murder': 'murder and homicide',
    'rape': 'rape and sexual assault',
    'kidnapping': 'kidnapping and abduction',
    'accident': 'road or vehicle accident',
    'sexual_harassment': 'sexual harassment',
    'theft': 'theft',
    'burglary': 'burglary and housebreaking',
    'robbery': 'robbery and dacoity',
    'fraud_cheating': 'fraud, cheating, and forgery',
    'crime_against_children': 'crime against children',
    'non_crime': 'general news not related to crime'
}

_zs_classifier = None

@st.cache_resource(show_spinner=False)
def get_zeroshot_classifier():
    """Load a robust multilingual zero-shot classifier."""
    global _zs_classifier
    if _zs_classifier is None:
        try:
            logger.info("Loading Multilingual Zero-Shot Classifier (mDeBERTa)...")
            # This model supports 100+ languages including English, Hindi, Kannada, etc.
            _zs_classifier = pipeline(
                "zero-shot-classification", 
                model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.error(f"Failed to load Zero-Shot model: {e}")
            # Fallback to a smaller/standard one if needed
            _zs_classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
    return _zs_classifier


def classify_articles(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """
    Classify articles using a Multilingual Zero-Shot pipeline.
    This allows detecting any number of crime categories without retraining.
    """
    if df.empty:
        return df

    # Initialize all category columns to 0
    for cat in CRIME_CATEGORIES:
        df[cat] = 0

    try:
        classifier = get_zeroshot_classifier()
        labels = list(DESCRIPTIVE_LABELS.values())
        label_to_key = {v: k for k, v in DESCRIPTIVE_LABELS.items()}

        for idx, row in df.iterrows():
            try:
                text = str(row[text_col]).strip()
                if not text or len(text) < 10:
                    continue

                # Run zero-shot inference
                # multi_label=True allows mapping to multiple crime types if relevant
                result = classifier(text, labels, multi_label=True)
                
                # Map high-confidence results back to our category keys
                for label, score in zip(result['labels'], result['scores']):
                    if score > 0.4:  # Threshold for detection
                        key = label_to_key.get(label)
                        if key:
                            df.at[idx, key] = 1
            except Exception as inner_e:
                logger.warning(f"Failed to classify article at index {idx}: {inner_e}")
                continue
            
            # If no crime categories found, mark as non_crime
            crime_found = any(df.at[idx, k] == 1 for k in CRIME_CATEGORIES if k != 'non_crime')
            if not crime_found:
                df.at[idx, 'non_crime'] = 1

        logger.info(f"Classified {len(df)} articles using Multilingual Zero-Shot Pipeline.")
    except Exception as e:
        logger.error(f"Zero-Shot Classification failed: {e}")

    return df

