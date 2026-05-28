import os
import re
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
    """Return the classifier to use.

    By default the fast keyword heuristic is used on ALL platforms so that
    classification is instant.  To opt-in to the heavy mDeBERTa zero-shot
    model (requires ~2 GB RAM and several minutes on first run) set the
    environment variable::

        USE_AI_CLASSIFIER=true

    before starting Streamlit.
    """
    global _zs_classifier
    if _zs_classifier is not None:
        return _zs_classifier

    import sys

    # --- Fast path (default): heuristic keyword classifier ---
    # Opt-out conditions: cloud deployment, forced override, OR simply not opted in.
    use_ai = os.environ.get('USE_AI_CLASSIFIER', 'false').lower() == 'true'
    force_heuristic = (
        os.environ.get('FORCE_HEURISTIC') == 'true'
        or os.environ.get('HOSTNAME') == 'streamlit-cloud'
        or sys.platform == 'linux'
        or not use_ai          # <-- DEFAULT: heuristic unless explicitly opted in
    )
    if force_heuristic:
        logger.info(
            "Using fast Heuristic Keyword Classifier. "
            "Set USE_AI_CLASSIFIER=true to enable the heavy AI model."
        )
        return "HEURISTIC"

    try:
        logger.info("Loading mDeBERTa multilingual Zero-Shot classifier (opt-in mode)...")
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

# ── Political context suppressor ────────────────────────────────────────────
# If an article title strongly matches ANY of these patterns it is almost
# certainly a political/governance story.  When such a title has NO crime
# keyword in the title itself (only in the RSS summary body), we override the
# classification and mark it as non_crime to prevent false positives.
_POLITICAL_TITLE_PATTERNS = re.compile(
    r'(?:'
    # English political roles & actions
    r'\b(?:chief minister|prime minister|home minister|finance minister|cm\b|pm\b'
    r'|mla\b|mp\b|governor|president|vice president|cabinet|election|minister'
    r'|parliament|assembly|lok sabha|rajya sabha|inaugurated|launched|announced'
    r'|press conference|political|party|congress|bjp|aap|jdu|bsp|ysrcp|dmc'
    r'|rally|yatra|padyatra|manifesto|coalition|budget|ordinance|legislation)\b'
    # Kannada political terms
    r'|ಮುಖ್ಯಮಂತ್ರಿ|ಪ್ರಧಾನ\s*ಮಂತ್ರಿ|ಮಂತ್ರಿ|ಶಾಸಕ|ಸಂಸದ|ಸಿಎಂ|ವಿಧಾನಸಭೆ'
    r'|ಸರ್ಕಾರ|ಆಶೀರ್ವಾದ|ಸ್ವಾಗತ|ಚುನಾವಣೆ|ಅಧಿವೇಶನ|ಒಕ್ಕೂಟ|ಪಕ್ಷ'
    # Hindi political terms
    r'|मुख्यमंत्री|प्रधानमंत्री|मंत्री|विधायक|सांसद|राज्यपाल|सरकार|आशीर्वाद'
    r'|राजनीति|विधानसभा|चुनाव|राजनेता|नेता|पार्टी|भाजपा|कांग्रेस'
    # Tamil political terms
    r'|முதலமைச்சர்|மந்திரி|சட்டமன்ற|பாராளுமன்ற'
    # Telugu political terms
    r'|ముఖ్యమంత్రి|మంత్రి|శాసనసభ|పార్లమెంట్'
    r')',
    re.IGNORECASE
)


def _heuristic_classify(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """Fast vectorised keyword-based classifier with political-context suppression.

    Strategy:
    1. Build a per-row corpus string (title + text + clean_text).
    2. Scan corpus against keyword lists with regex.
    3. For any match, also check if the article TITLE alone triggers the keyword.
       If the title is clearly political AND the crime keyword only appears in the
       summary/body (not the title), suppress the crime label → non_crime.
    """
    logger.info("Running fast Heuristic Keyword Classification (%d articles)...", len(df))

    # Ensure all category columns exist and start at 0
    for cat in CRIME_CATEGORIES:
        if cat not in df.columns:
            df[cat] = 0

    # ── Build search strings ─────────────────────────────────────────────────
    title_series: pd.Series = (
        df.get('title', pd.Series([''] * len(df), index=df.index)).fillna('').astype(str)
    ).str.lower()

    corpus: pd.Series = (
        title_series
        + " "
        + df.get('text', pd.Series([''] * len(df), index=df.index)).fillna('').astype(str).str.lower()
        + " "
        + df.get(text_col, pd.Series([''] * len(df), index=df.index)).fillna('').astype(str).str.lower()
    ).str.replace(' nan ', ' ', regex=False)

    # ── Precompile per-category patterns ────────────────────────────────────
    cat_patterns = {
        cat: re.compile('|'.join(re.escape(w) for w in words), re.IGNORECASE)
        for cat, words in _HEURISTIC_KEYWORDS.items()
    }

    # Strong crime-action words that override the political suppressor.
    # If these appear IN the title, it's real crime news regardless of political context
    # (e.g. "CM arrested", "Minister killed", "MLA shot dead").
    _CRIME_ACTION_IN_TITLE = re.compile(
        r'\b(?:arrested|detained|killed|shot|stabbed|murdered|accused|chargesheet'
        r'|fir|raped|abducted|kidnapped|robbed|looted|cheated|defrauded|held'
        r'|convicted|sentenced|jailed|nabbed|caught)\b'
        r'|(?:\u0917\u093f\u0930\u092b\u094d\u0924\u093e\u0930|\u0917\u093f\u0930\u092b\u094d\u0924|\u0906\u0930\u094b\u092a\u0940|\u092e\u093e\u0930\u093e \u0917\u092f\u093e)'  # Hindi: arrested/accused/killed
        r'|(?:\u0c17\u0c3f\u0c30\u0c2b\u0c4d\u0c24\u0c3e\u0c30\u0c4d|\u0c39\u0c24\u0c4d\u0c2f)'            # Telugu: arrested/murder
        r'|(?:\u0cac\u0c82\u0ca7\u0cbf\u0ca4|\u0c95\u0cca\u0cb2\u0cc6|\u0cb9\u0ca4\u0ccd\u0caf\u0cc6)'       # Kannada: arrested/murder
        r'|(?:\u0b95\u0bc8\u0ba4\u0bc1|\u0b95\u0bc6\u0bbe\u0bb2\u0bc8)',                          # Tamil: arrested/murder
        re.IGNORECASE
    )

    # ── Vectorised primary keyword scan ─────────────────────────────────────
    crime_mask = pd.Series(False, index=df.index)
    # Pre-compute which titles have a strong crime action (overrides political suppressor)
    crime_action_in_title = title_series.str.contains(_CRIME_ACTION_IN_TITLE, regex=True, na=False)

    for cat, pat in cat_patterns.items():
        matched_full = corpus.str.contains(pat, regex=True, na=False)

        if matched_full.any():
            # For rows that matched in the full corpus, check if the crime keyword
            # also appears in the title.  If NOT in title AND title is clearly
            # political AND no strong crime action word in title → suppress.
            matched_in_title = title_series.str.contains(pat, regex=True, na=False)
            political_title  = title_series.str.contains(_POLITICAL_TITLE_PATTERNS, regex=True, na=False)

            # Suppress: matched only in body + political title + no crime action
            suppressed = matched_full & ~matched_in_title & political_title & ~crime_action_in_title
            confirmed  = matched_full & ~suppressed

            df.loc[confirmed, cat] = 1
            crime_mask |= confirmed

    # ── High-signal indicator phrases (fallback for unmatched rows) ─────────
    no_crime_idx = df.index[~crime_mask]
    if len(no_crime_idx) > 0:
        political_mask = title_series.str.contains(_POLITICAL_TITLE_PATTERNS, regex=True, na=False)
        for cat, phrases in _CRIME_INDICATOR_PHRASES:
            pat = re.compile('|'.join(re.escape(p) for p in phrases), re.IGNORECASE)
            matched = corpus.loc[no_crime_idx].str.contains(pat, regex=True, na=False)
            matched_idx = matched[matched].index
            if len(matched_idx) > 0:
                # Apply political suppression here too, but NOT when a strong
                # crime-action word (arrested, killed, etc.) is in the title.
                title_match       = title_series.loc[matched_idx].str.contains(pat, regex=True, na=False)
                is_political      = political_mask.loc[matched_idx]
                has_crime_action  = crime_action_in_title.loc[matched_idx]
                confirmed_idx     = matched_idx[title_match | ~is_political | has_crime_action]
                if len(confirmed_idx) > 0:
                    df.loc[confirmed_idx, cat] = 1
                    crime_mask.loc[confirmed_idx] = True
                    no_crime_idx = df.index[~crime_mask]

    # ── Mark non-crime rows ──────────────────────────────────────────────────
    df.loc[~crime_mask, 'non_crime'] = 1
    df.loc[crime_mask,  'non_crime'] = 0

    crime_count = int(crime_mask.sum())
    logger.info(
        "Heuristic classification done: %d crime / %d non-crime out of %d articles.",
        crime_count, len(df) - crime_count, len(df)
    )
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

            if crime_found:
                df.at[idx, 'non_crime'] = 0
            else:
                df.at[idx, 'non_crime'] = 1

        progress_bar.empty()
        logger.info(f"Classified {len(df)} articles using Multilingual Zero-Shot Pipeline.")
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return _heuristic_classify(df, text_col)

    return df

