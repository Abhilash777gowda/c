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
        
    import sys
    # Instantly trigger heuristic mode on Streamlit Cloud or forced overrides to prevent memory leaks/crashes
    if os.environ.get('FORCE_HEURISTIC') == 'true' or os.environ.get('HOSTNAME') == 'streamlit-cloud' or sys.platform == 'linux':
        logger.info("Forcing Zero-Memory Heuristic Classifier (Cloud deployment / override detected).")
        return "HEURISTIC"

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
        'murder', 'murdered', 'kills', 'killed', 'kill', 'shot dead', 'shot and killed',
        'stabbed', 'stabbing', 'homicide', 'slain', 'found dead', 'body found', 'dead body',
        'corpse', 'hanged', 'lynched', 'beaten to death', 'mob lynching', 'dies in attack',
        'man dead', 'woman dead', 'youth dead', 'dies after', 'fatal attack', 'charred body',
        'encounter killing', 'hatya', 'kolai',
        # Hindi
        'हत्या', 'कत्ल', 'मारा गया', 'वध', 'हत्यारों', 'लाश', 'शव', 'मृत', 'गोली मार', 'गोली',
        'छुरा', 'छुरा मार', 'जान से मारा', 'हत्यारों ने', 'डूब', 'डूबा', 'नदी में', 'कुएं में',
        'मौत', 'शव मिला', 'शव बरामद',
        # Kannada
        'ಹತ್ಯೆ', 'ಕೊಲೆ', 'ಮರ್ಡರ್', 'ಹೆಣ', 'ಶವ',
        # Telugu
        'హత్య', 'హతుడు', 'చంపబడ్డాడు', 'శవం',
        # Tamil
        'கொலை', 'இறப்பு', 'படுகொலை', 'சடலம்'
    ],
    'rape': [
        'rape', 'raped', 'gang rape', 'gangrape', 'gang-rape', 'sexual assault',
        'sexually assaulted', 'sexual abuse', 'balatkar', 'minor raped',
        'woman raped', 'girl raped', 'student raped',
        # Kannada
        'ಅತ್ಯಾಚಾರ', 'ರೇಪ್', 'ಲೈಂಗಿಕ ದೌರ್ಜನ್ಯ',
        # Hindi
        'बलात्कार', 'दुष्कर्म', 'यौन उत्पीड़न', 'यौन शोषण',
        # Telugu
        'అత్యాచారం', 'లైంగిక దాడి',
        # Tamil
        'கற்பழிப்பு', 'பாலியல் வன்கொடுமை'
    ],
    'kidnapping': [
        'kidnap', 'kidnapped', 'kidnapping', 'abduct', 'abducted', 'abduction', 'hostage',
        'missing child', 'missing girl', 'missing boy', 'child missing', 'woman missing',
        'apaharan', 
        # Kannada
        'ಅಪಹರಣ', 'ಕಿಡ್ನ್ಯಾಪ್', 'ನಾಪತ್ತೆ',
        # Hindi
        'किडनैप', 'अपहरण', 'गुमशुदा', 'बंधक',
        # Telugu
        'కిడ్నాప్', 'అపహరణ', 'లాక్కెళ్ళారు',
        # Tamil
        'கடத்தல்', 'காணாமல்'
    ],
    'sexual_harassment': [
        'harassment', 'harassed', 'molest', 'molestation', 'molested', 'eve teasing',
        'eve-teasing', 'outrage of modesty', 'sexual harassment', 'stalking', 'stalked',
        # Kannada
        'ಕಿರುಕುಳ', 'ಲೈಂಗಿಕ ಕಿರುಕುಳ',
        # Hindi
        'छेड़छाड़', 'उत्पीड़न', 'यौन प्रताड़ना',
        # Telugu
        'వేధింపు', 'పీడన',
        # Tamil
        'துன்புறுத்தல்', 'ஈவ்டீசிங்'
    ],
    'crime_against_children': [
        'pocso', 'child abuse', 'child sexual', 'minor girl', 'minor boy', 'minor raped',
        'child traffick', 'children traffick', 'child labour', 'juvenile', 'child victim',
        'school girl', 'teenage girl',
        # Kannada
        'ಮಕ್ಕಳ', 'ಬಾಲಕ', 'ಬಾಲಕಿ', 'ಅಪ್ರಾಪ್ತ',
        # Hindi
        'बच्चे', 'नाबालिग', 'बाल विवाह', 'बाल श्रम',
        # Telugu
        'పిల్లల', 'బాలల', 'మైనర్',
        # Tamil
        'குழந்தை', 'சிறுமி', 'சிறுவன்'
    ],
    'theft': [
        'theft', 'stolen', 'thief', 'thieves', 'stealing', 'snatched', 'pickpocket',
        'shoplifting', 'loot', 'looted', 'chori', 'vehicle theft', 'bike theft',
        'jewellery stolen', 'cash stolen', 'mobile stolen',
        # Kannada
        'ಕಳ್ಳತನ', 'ಚೋರಿ', 'ಕಳ್ಳ', 'ಕಳವು',
        # Hindi
        'चोरी', 'चोर गिरफ्तार', 'सामान चोरी',
        # Telugu
        'దొంగతనం', 'దొంగ', 'చౌర్యం',
        # Tamil
        'திருட்டு', 'களவாடப்பட்டது'
    ],
    'burglary': [
        'burglary', 'burgled', 'break-in', 'broke in', 'broken into', 'housebreak',
        'housebreaking', 'house robbery', 'house looted', 'home invasion',
        # Kannada
        'ದರೋಡೆ', 'ಕನ್ನ ಹಾಕಿದ',
        # Hindi
        'सेंधमारी', 'घर में चोरी',
        # Telugu
        'దోపిడీ', 'ఇంట్లో దొంగతనం',
        # Tamil
        'கொள்ளை', 'வீடு புகுந்து'
    ],
    'robbery': [
        'robbery', 'robbed', 'dacoity', 'dacoit', 'dacoits', 'mugged', 'snatched',
        'armed robbery', 'bank robbery', 'chain snatching', 'snatching incident',
        # Kannada
        'ಲೂಟಿ', 'ಸರಗಳ್ಳತನ', 'ಸರ ಅಪಹರಣ',
        # Hindi
        'लूट', 'डकैती', 'झपटमारी',
        # Telugu
        'దోపిడీ', 'దోపిడి',
        # Tamil
        'வழிப்பறி', 'பறிப்பு'
    ],
    'fraud_cheating': [
        'fraud', 'fraudulent', 'cheated', 'cheating', 'scam', 'scammed', 'duped', 'fake',
        'phishing', 'cyber crime', 'cybercrime', 'online fraud', 'ponzi', 'forgery', 'forged',
        'swindled', 'conned', 'blackmail', 'extortion', 'impersonation', 'fake call',
        'investment fraud', 'job fraud', 'matrimonial fraud', 'UPI fraud',
        # Kannada
        'ವಂಚನೆ', 'ವೆಂಚನೆ', 'ಮೋಸ', 'ಖೋಟಾ',
        # Hindi
        'धोखाधड़ी', 'ठगी', 'फर्जीवाड़ा', 'घोटाला', 'साइबर अपराध',
        # Telugu
        'మోసం', 'ఫోర్జరీ', 'కుంభకోణం',
        # Tamil
        'மோசடி', 'ஏமாற்று', 'போலி'
    ],
    'accident': [
        'accident', 'accidents', 'crashed', 'crash', 'collision', 'collided', 'collide',
        'road accident', 'vehicle accident', 'car accident', 'bike accident', 'mishap',
        'run over', 'hit and run', 'fatally injured', 'injured in', 'durghatna',
        'highway accident', 'truck accident', 'bus accident', 'falls from', 'fell from',
        # Hindi
        'दुर्घटना', 'हादसा', 'यमुना', 'नदी में', 'डूबा', 'डूबने', 'तालाब', 'हादसे में', 'पलटी', 'घायल',
        # Kannada
        'ಅಪಘಾತ', 'ಡಿಕ್ಕಿ', 'ದುರಂತ', 'ಬಲಿ', 'ಗಾಯ',
        # Telugu
        'ప్రమాదం', 'ఢీకొట్టింది', 'దుర్మరణం',
        # Tamil
        'விபத்து', 'மோதியது', 'உயிரிழப்பு'
    ],
}

# High-signal Indian crime-reporting phrases — if these appear,
# the article is almost certainly crime news
_CRIME_INDICATOR_PHRASES = [
    ('murder',         ['encounter', 'gang war', 'contract killing', 'supari killing',
                        'murder accused', 'murder case', 'murder fir', 'murder arrested',
                        'murder suspect', 'dead body recovered', 'body recovered',
                        'unidentified body', 'murder confession']),
    ('fraud_cheating', ['arrested for fraud', 'fir for fraud', 'cyber fraud arrested',
                        'online scam', 'cheating case', 'cheating arrested']),
    ('robbery',        ['held for robbery', 'arrested for robbery', 'robbery accused']),
    ('theft',          ['arrested for theft', 'theft case', 'theft accused']),
    ('rape',           ['arrested for rape', 'rape accused', 'rape case', 'rape fir',
                        'rape survivor', 'rape victim']),
    ('kidnapping',     ['kidnapping accused', 'kidnapping case', 'child recovered',
                        'rescued from kidnappers', 'ransom demand', 'ransom paid']),
    # Hindi high-signal phrases
    ('murder',         ['गिरफ्तार', 'हत्यारा', 'हत्यारों को', 'टारगेट किलिंग', 'एनकाउंटर',
                        'गैंग वार', 'आरोपी गिरफ्तार', 'हमलावर', 'गोली चलाई']),
    ('rape',           ['बलात्कार आरोपी', 'दुष्कर्म आरोपी', 'दुष्कर्म का मामला']),
    ('theft',          ['चोरी का मामला', 'चोर गिरफ्तार', 'लूट का मामला']),
    ('fraud_cheating', ['धोखाधड़ी का मामला', 'ठग गिरफ्तार', 'साइबर ठगी']),
    ('accident',       ['सड़क दुर्घटना', 'हादसे में घायल', 'हादसे में मौत', 'ट्रक की टक्कर',
                        'कार दुर्घटना', 'बाइक हादसा']),
]

def _heuristic_classify(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """Fast keyword-based classifier. Scans raw title+text+clean_text and indicator phrases."""
    logger.info("Running Heuristic Keyword Classification...")
    for idx, row in df.iterrows():
        # Scan raw title + raw text + clean_text for maximum recall
        raw_text = (
            str(row.get('title', '')) + " " +
            str(row.get('text', '')) + " " +
            str(row.get(text_col, ''))
        ).lower()

        # Sanity check: replace nan strings from pandas conversion
        raw_text = raw_text.replace(" nan ", " ")

        crime_found = False
        if len(raw_text) > 5:
            # Primary keyword scan
            for cat, words in _HEURISTIC_KEYWORDS.items():
                if any(w in raw_text for w in words):
                    df.at[idx, cat] = 1
                    crime_found = True

            # High-signal indicator phrases scan
            if not crime_found:
                for cat, phrases in _CRIME_INDICATOR_PHRASES:
                    if any(p in raw_text for p in phrases):
                        df.at[idx, cat] = 1
                        crime_found = True
                        break

        if not crime_found:
            df.at[idx, 'non_crime'] = 1

    logger.info("Heuristic classification completed successfully.")
    return df


def classify_articles(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """
    Classify articles using a Multilingual Zero-Shot pipeline (locally)
    with a lightning-fast keyword heuristic fallback to prevent false non-crime categorizations.
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

        # Multilingual Zero-Shot (Local PC)
        labels = list(DESCRIPTIVE_LABELS.values())
        label_to_key = {v: k for k, v in DESCRIPTIVE_LABELS.items()}

        total = len(df)
        progress_bar = st.progress(0, text=f"AI Classification in progress... (0/{total} articles)")

        for i, (idx, row) in enumerate(df.iterrows()):
            if i % 2 == 0 or i == total - 1:
                progress_bar.progress(min((i + 1) / total, 1.0), text=f"AI Classification in progress... ({i + 1}/{total} articles)")
            
            crime_found = False
            try:
                text = str(row[text_col]).strip()
                if text == "nan":
                    text = ""

                if len(text) >= 10:
                    result = classifier(text, labels, multi_label=True)
                    for label, score in zip(result['labels'], result['scores']):
                        if score > 0.4:
                            key = label_to_key.get(label)
                            if key:
                                df.at[idx, key] = 1
                                if key != 'non_crime':
                                    crime_found = True

                # --- Double-Layer Hybrid Fallback: Run Heuristic if AI did not detect a crime ---
                if not crime_found:
                    raw_text = (
                        str(row.get('title', '')) + " " +
                        str(row.get('text', '')) + " " +
                        str(row.get(text_col, ''))
                    ).lower().replace(" nan ", " ")
                    
                    for cat, words in _HEURISTIC_KEYWORDS.items():
                        if any(w in raw_text for w in words):
                            df.at[idx, cat] = 1
                            crime_found = True

                    if not crime_found:
                        for cat, phrases in _CRIME_INDICATOR_PHRASES:
                            if any(p in raw_text for p in phrases):
                                df.at[idx, cat] = 1
                                crime_found = True
                                break
            except Exception as inner_e:
                logger.warning(f"Failed to classify article at index {idx}: {inner_e}")
                # Fallback to heuristic on exception
                raw_text = (
                    str(row.get('title', '')) + " " +
                    str(row.get('text', ''))
                ).lower().replace(" nan ", " ")
                for cat, words in _HEURISTIC_KEYWORDS.items():
                    if any(w in raw_text for w in words):
                        df.at[idx, cat] = 1
                        crime_found = True
                        break

            if not crime_found:
                df.at[idx, 'non_crime'] = 1

        progress_bar.empty()
        logger.info(f"Classified {len(df)} articles using Multilingual Zero-Shot Pipeline.")
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return _heuristic_classify(df, text_col)

    return df

