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
    """Load the multilingual zero-shot classifier, falling back to heuristic if RAM is insufficient."""
    global _zs_classifier
    if _zs_classifier is not None:
        return _zs_classifier
    try:
        logger.info("Loading mDeBERTa multilingual Zero-Shot classifier...")
        clf = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            device=-1,  # CPU
        )
        _zs_classifier = clf
        logger.info("Zero-Shot model loaded successfully.")
        return clf
    except Exception as e:
        logger.error(f"Failed to load Zero-Shot model ({e}). Falling back to heuristic classifier.")
        return "HEURISTIC"


# Expanded multilingual keyword table — scanned against RAW title + text (NOT clean_text)
_HEURISTIC_KEYWORDS = {
    'murder': [
        'murder', 'killed', 'kill', 'shot dead', 'shot and killed', 'stabbed', 'stabbing',
        'homicide', 'slain', 'found dead', 'body found', 'dead body', 'corpse', 'hanged',
        'lynched', 'beaten to death', 'death', 'dies', 'died', 'mob lynching',
        'hatya', 'kolai', 'ಹತ್ಯೆ', 'ಕೊಲೆ', 'ಮರ್ಡರ್', 'हत्या', 'कत्ल', 'मारा गया',
        'హత్య', 'హతుడు', 'చంపబడ్డాడు',
    ],
    'rape': [
        'rape', 'raped', 'gang rape', 'gangrape', 'sexual assault', 'sexually assaulted',
        'sexual abuse', 'balatkar', 'ಅತ್ಯಾಚಾರ', 'ರೇಪ್', 'बलात्कार', 'दुष्कर्म', 'यौन उत्पीड़न',
        'కற్పழிப்பு', 'అత్యాచారం',
    ],
    'kidnapping': [
        'kidnap', 'kidnapped', 'abduct', 'abducted', 'abduction', 'hostage', 'missing child',
        'missing girl', 'missing boy', 'apaharan', 'ಅಪಹರಣ', 'किडनैप', 'अपहरण', 'गुमशुदा',
        'కడత్తல்', 'కిడ్నాప్', 'కిడ్నాప్',
    ],
    'sexual_harassment': [
        'harassment', 'harassed', 'molest', 'molestation', 'molested', 'eve teasing',
        'eve-teasing', 'outrage of modesty', 'sexual harassment', 'stalking', 'stalked',
        'ಕಿರುಕುಳ', 'छेड़छाड़', 'துன்புறுத்தல்', 'వేధింపు', 'పీడన',
    ],
    'crime_against_children': [
        'pocso', 'child abuse', 'child sexual', 'minor girl', 'minor boy', 'minor raped',
        'child traffick', 'children traffick', 'child labour', 'ಮಕ್ಕಳ', 'बच्चे', 'नाबालिग',
        'குழந்தை', 'పిల్లల', 'బాలల',
    ],
    'theft': [
        'theft', 'stolen', 'thief', 'thieves', 'stealing', 'snatched', 'pickpocket',
        'shoplifting', 'loot', 'looted', 'chori', 'ಕಳ್ಳತನ', 'ಚೋರಿ', 'चोरी', 'திருட்டு',
        'దొంగతనం', 'దొంగ',
    ],
    'burglary': [
        'burglary', 'burgled', 'break-in', 'broke in', 'broken into', 'housebreak',
        'housebreaking', 'ದರೋಡೆ', 'सेंधमारी', 'கொள்ளை', 'దోపిడీ',
    ],
    'robbery': [
        'robbery', 'robbed', 'dacoity', 'dacoit', 'dacoits', 'mugged', 'snatched', 'snatch',
        'armed robbery', 'bank robbery', 'ಲೂಟಿ', 'लूट', 'डकैती', 'வழிப்பறி', 'దోపిడీ',
    ],
    'fraud_cheating': [
        'fraud', 'fraudulent', 'cheated', 'cheating', 'scam', 'scammed', 'duped', 'fake',
        'phishing', 'cyber crime', 'cybercrime', 'online fraud', 'ponzi', 'forgery', 'forged',
        'swindled', 'conned', 'sting', 'sting operation', 'blackmail', 'extortion',
        'ವಂಚನೆ', 'மோசடி', 'धोखाधड़ी', 'ठगी', 'మోసం',
    ],
    'accident': [
        'accident', 'accidents', 'crashed', 'crash', 'collision', 'collided', 'collide',
        'road accident', 'vehicle accident', 'car accident', 'bike accident', 'mishap',
        'run over', 'hit and run', 'fatally injured', 'injured in', 'durghatna', 'hादसा',
        'ಅಪಘಾತ', 'ಡಿಕ್ಕಿ', 'ದುರಂತ', 'दुर्घटना', 'हादसा', 'விபத்து', 'ప్రమాదం',
    ],
}


def _heuristic_classify(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """Fast keyword-based classifier. Scans both raw title+text AND clean_text."""
    logger.info("Running Heuristic Keyword Classification...")
    for idx, row in df.iterrows():
        # Scan raw title + raw text + clean_text for maximum recall
        raw_text = (
            str(row.get('title', '')) + " " +
            str(row.get('text', '')) + " " +
            str(row.get(text_col, ''))
        ).lower()

        crime_found = False
        if len(raw_text) > 5:
            for cat, words in _HEURISTIC_KEYWORDS.items():
                if any(w in raw_text for w in words):
                    df.at[idx, cat] = 1
                    crime_found = True

        if not crime_found:
            df.at[idx, 'non_crime'] = 1

    logger.info("Heuristic classification completed successfully.")
    return df


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
            return _heuristic_classify(df, text_col)

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

