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
            import sys
            import os
            # If deployed to Streamlit Community Cloud (typically Linux with 1GB RAM limit),
            # ANY transformer model (even DistilBERT) causes PyTorch/HF overhead to hit OOM.
            # We return a flag to use a zero-memory Heuristic Keyword approach instead.
            if sys.platform == 'linux':
                logger.info("Cloud deployment detected. Forcing Zero-Memory Heuristic Classifier to prevent OOM!")
                return "HEURISTIC"
            else:
                logger.info("Loading Multilingual Zero-Shot Classifier (mDeBERTa)...")
                # This model supports 100+ languages including English, Hindi, Kannada, etc.
                _zs_classifier = pipeline(
                    "zero-shot-classification", 
                    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
                    device=0 if torch.cuda.is_available() else -1
                )
        except Exception as e:
            logger.error(f"Failed to load Zero-Shot model: {e}")
            return "HEURISTIC"
    return _zs_classifier


def classify_articles(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """
    Classify articles using a Multilingual Zero-Shot pipeline (locally)
    or a lightning-fast keyword heuristic (on Streamlit Cloud) to prevent crashes.
    """
    if df.empty:
        return df

    # Initialize all category columns to 0
    for cat in CRIME_CATEGORIES:
        df[cat] = 0

    try:
        classifier = get_zeroshot_classifier()
        
        if classifier == "HEURISTIC":
            logger.info("Running Heuristic Keyword Classification...")
            KEYWORDS = {
                'murder': ['murder', 'kill', 'shot', 'homicide', 'stab', 'slain', 'hatya', 'kolai', 'dead body', 'ಹತ್ಯೆ', 'ಕೊಲೆ', 'ಮರ್ಡರ್', 'हत्या', 'कत्ल', 'மரண', 'హత్య'],
                'rape': ['rape', 'sexual assault', 'gangrape', 'balatkar', 'ಅತ್ಯಾಚಾರ', 'ರೇಪ್', 'बलात्कार', 'दुष्कर्म', 'கற்பழிப்பு', 'అత్యాచారం'],
                'kidnapping': ['kidnap', 'abduct', 'hostage', 'apaharan', 'ಅಪಹರಣ', 'किडनैप', 'अपहरण', 'கடத்தல்', 'కిడ్నాప్'],
                'sexual_harassment': ['harassment', 'molest', 'eve-teasing', 'outrage', 'ಕಿರುಕುಳ', 'छेड़छाड़', 'துன்புறுத்தல்', 'వేధింపు'],
                'crime_against_children': ['pocso', 'child abuse', 'minor girl', 'minor boy', 'ಮಕ್ಕಳ', 'बच्चे', 'குழந்தை', 'పిల్లల'],
                'theft': ['theft', 'stolen', 'thief', 'chori', 'stealing', 'ಕಳ್ಳತನ', 'ಚೋರಿ', 'चोरी', 'திருட்டு', 'దొంగతనం'],
                'burglary': ['burglary', 'break-in', 'looted', 'housebreak', 'ದರೋಡೆ', 'सेंधमारी', 'கொள்ளை', 'దోపిడీ'],
                'robbery': ['robbery', 'robbed', 'dacoity', 'mugged', 'snatch', 'ಲೂಟಿ', 'लूट', 'डकैती', 'வழிப்பறி', 'దోపిడీ'],
                'fraud_cheating': ['fraud', 'cheat', 'scam', 'dupe', 'fake', 'phishing', 'cybercrime', 'ವಂಚನೆ', 'மோசடி', 'धोखाधड़ी', 'ठगी', 'మోసం'],
                'accident': ['accident', 'crash', 'collide', 'collision', 'mishap', 'durghatna', 'run over', 'ಅಪಘಾತ', 'ಡಿಕ್ಕಿ', 'ದುರಂತ', 'दुर्घटना', 'हादसा', 'விபத்து', 'ప్రమాదం']
            }
            
            for idx, row in df.iterrows():
                text = str(row.get('title', '')) + " " + str(row.get(text_col, ''))
                text = text.lower()
                crime_found = False
                
                if len(text) > 5:
                    for cat, words in KEYWORDS.items():
                        if any(w in text for w in words):
                            df.at[idx, cat] = 1
                            crime_found = True
                            
                if not crime_found:
                    df.at[idx, 'non_crime'] = 1
                    
            logger.info("Heuristic classification completed successfully.")
            return df

        # Fallback to standard Zero-Shot (Local PC)
        labels = list(DESCRIPTIVE_LABELS.values())
        label_to_key = {v: k for k, v in DESCRIPTIVE_LABELS.items()}

        total = len(df)
        progress_bar = st.progress(0, text=f"AI Classification in progress... (0/{total} articles)")

        for i, (idx, row) in enumerate(df.iterrows()):
            if i % 2 == 0 or i == total - 1:
                progress_bar.progress(min((i + 1) / total, 1.0), text=f"AI Classification in progress... ({i + 1}/{total} articles)")
            try:
                text = str(row[text_col]).strip()
                if not text or len(text) < 10:
                    continue

                # Run zero-shot inference
                result = classifier(text, labels, multi_label=True)
                
                for label, score in zip(result['labels'], result['scores']):
                    if score > 0.4:
                        key = label_to_key.get(label)
                        if key:
                            df.at[idx, key] = 1
            except Exception as inner_e:
                logger.warning(f"Failed to classify article at index {idx}: {inner_e}")
                continue
            
            crime_found = any(df.at[idx, k] == 1 for k in CRIME_CATEGORIES if k != 'non_crime')
            if not crime_found:
                df.at[idx, 'non_crime'] = 1

        progress_bar.empty()
        logger.info(f"Classified {len(df)} articles using Multilingual Zero-Shot Pipeline.")
    except Exception as e:
        logger.error(f"Classification failed: {e}")

    return df

