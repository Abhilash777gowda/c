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
    """Return HEURISTIC by default; opt-in to heavy AI via USE_AI_CLASSIFIER=true."""
    global _zs_classifier
    if _zs_classifier is not None:
        return _zs_classifier

    import sys
    use_ai = os.environ.get('USE_AI_CLASSIFIER', 'false').lower() == 'true'
    force_heuristic = (
        os.environ.get('FORCE_HEURISTIC') == 'true'
        or os.environ.get('HOSTNAME') == 'streamlit-cloud'
        or sys.platform == 'linux'
        or not use_ai
    )
    if force_heuristic:
        logger.info("Using fast Heuristic Keyword Classifier. Set USE_AI_CLASSIFIER=true to enable AI model.")
        return "HEURISTIC"

    try:
        logger.info("Loading mDeBERTa multilingual Zero-Shot classifier (opt-in mode)...")
        clf = pipeline("zero-shot-classification",
                       model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
                       device=-1)
        _zs_classifier = clf
        logger.info("Zero-Shot model loaded successfully.")
        return clf
    except Exception as e:
        logger.error(f"Failed to load Zero-Shot model ({e}). Falling back to heuristic.")
        return "HEURISTIC"


# ---------------------------------------------------------------------------
# Keyword tables
# ---------------------------------------------------------------------------
_HEURISTIC_KEYWORDS = {
    'murder': [
        'murder', 'murdered', 'kills', 'killed', 'kill', 'shot dead', 'shot and killed',
        'stabbed', 'stabbing', 'homicide', 'slain', 'found dead', 'body found', 'dead body',
        'corpse', 'hanged', 'lynched', 'beaten to death', 'mob lynching', 'dies in attack',
        'man dead', 'woman dead', 'youth dead', 'dies after', 'fatal attack', 'charred body',
        'encounter killing', 'hatya', 'kolai',
        'हत्या', 'कत्ल', 'मारा गया', 'वध', 'हत्यारों', 'लाश', 'शव', 'मृत', 'गोली मार',
        'छुरा मार', 'जान से मारा', 'हत्यारों ने', 'मौत', 'शव मिला', 'शव बरामद',
        'ಹತ್ಯೆ', 'ಕೊಲೆ', 'ಮರ್ಡರ್', 'ಹೆಣ', 'ಶವ',
        'హత్య', 'హతుడు', 'చంపబడ్డాడు', 'శవం',
        'கொலை', 'இறப்பு', 'படுகொலை', 'சடலம்'
    ],
    'rape': [
        'rape', 'raped', 'gang rape', 'gangrape', 'gang-rape', 'sexual assault',
        'sexually assaulted', 'sexual abuse', 'balatkar', 'minor raped',
        'woman raped', 'girl raped', 'student raped',
        'ಅತ್ಯಾಚಾರ', 'ರೇಪ್', 'ಲೈಂಗಿಕ ದೌರ್ಜನ್ಯ',
        'बलात्कार', 'दुष्कर्म', 'यौन उत्पीड़न', 'यौन शोषण',
        'అత్యాచారం', 'లైంగిక దాడి',
        'கற்பழிப்பு', 'பாலியல் வன்கொடுமை'
    ],
    'kidnapping': [
        'kidnap', 'kidnapped', 'kidnapping', 'abduct', 'abducted', 'abduction', 'hostage',
        'missing child', 'missing girl', 'missing boy', 'child missing', 'woman missing', 'apaharan',
        'ಅಪಹರಣ', 'ಕಿಡ್ನ್ಯಾಪ್', 'ನಾಪತ್ತೆ',
        'किडनैप', 'अपहरण', 'गुमशुदा', 'बंधक',
        'కిడ్నాప్', 'అపహరణ',
        'கடத்தல்', 'காணாமல்'
    ],
    'sexual_harassment': [
        'harassment', 'harassed', 'molest', 'molestation', 'molested', 'eve teasing',
        'eve-teasing', 'outrage of modesty', 'sexual harassment', 'stalking', 'stalked',
        'ಕಿರುಕುಳ', 'ಲೈಂಗಿಕ ಕಿರುಕುಳ',
        'छेड़छाड़', 'उत्पीड़न', 'यौन प्रताड़ना',
        'వేధింపు', 'పీడన',
        'துன்புறுத்தல்', 'ஈவ்டீசிங்'
    ],
    'crime_against_children': [
        'pocso', 'child abuse', 'child sexual', 'minor girl', 'minor boy', 'minor raped',
        'child traffick', 'children traffick', 'child labour', 'juvenile', 'child victim',
        'school girl', 'teenage girl',
        'ಮಕ್ಕಳ', 'ಬಾಲಕ', 'ಬಾಲಕಿ', 'ಅಪ್ರಾಪ್ತ',
        'बच्चे', 'नाबालिग', 'बाल विवाह', 'बाल श्रम',
        'పిల్లల', 'బాలల', 'మైనర్',
        'குழந்தை', 'சிறுமி', 'சிறுவன்'
    ],
    'theft': [
        'theft', 'stolen', 'thief', 'thieves', 'stealing', 'snatched', 'pickpocket',
        'shoplifting', 'loot', 'looted', 'chori', 'vehicle theft', 'bike theft',
        'jewellery stolen', 'cash stolen', 'mobile stolen',
        'ಕಳ್ಳತನ', 'ಚೋರಿ', 'ಕಳ್ಳ', 'ಕಳವು',
        'चोरी', 'चोर गिरफ्तार', 'सामान चोरी',
        'దొంగతనం', 'దొంగ', 'చౌర్యం',
        'திருட்டு', 'களவாடப்பட்டது'
    ],
    'burglary': [
        'burglary', 'burgled', 'break-in', 'broke in', 'broken into', 'housebreak',
        'housebreaking', 'house robbery', 'house looted', 'home invasion',
        'ದರೋಡೆ', 'ಕನ್ನ ಹಾಕಿದ',
        'सेंधमारी', 'घर में चोरी',
        'దోపిడీ', 'ఇంట్లో దొంగతనం',
        'கொள்ளை', 'வீடு புகுந்து'
    ],
    'robbery': [
        'robbery', 'robbed', 'dacoity', 'dacoit', 'dacoits', 'mugged', 'snatched',
        'armed robbery', 'bank robbery', 'chain snatching', 'snatching incident',
        'ಲೂಟಿ', 'ಸರಗಳ್ಳತನ', 'ಸರ ಅಪಹರಣ',
        'लूट', 'डकैती', 'झपटमारी',
        'దోపిడి',
        'வழிப்பறி', 'பறிப்பு'
    ],
    'fraud_cheating': [
        'fraud', 'fraudulent', 'cheated', 'cheating', 'scam', 'scammed', 'duped', 'fake',
        'phishing', 'cyber crime', 'cybercrime', 'online fraud', 'ponzi', 'forgery', 'forged',
        'swindled', 'conned', 'blackmail', 'extortion', 'impersonation', 'fake call',
        'investment fraud', 'job fraud', 'matrimonial fraud', 'UPI fraud',
        'ವಂಚನೆ', 'ಮೋಸ', 'ಖೋಟಾ',
        'धोखाधड़ी', 'ठगी', 'फर्जीवाड़ा', 'घोटाला', 'साइबर अपराध',
        'మోసం', 'కుంభకోణం',
        'மோசடி', 'ஏமாற்று', 'போலி'
    ],
    'accident': [
        'accident', 'accidents', 'crashed', 'crash', 'collision', 'collided', 'collide',
        'road accident', 'vehicle accident', 'car accident', 'bike accident', 'mishap',
        'run over', 'hit and run', 'fatally injured', 'injured in', 'durghatna',
        'highway accident', 'truck accident', 'bus accident', 'falls from', 'fell from',
        'दुर्घटना', 'हादसा', 'हादसे में', 'पलटी', 'घायल',
        'ಅಪಘಾತ', 'ಡಿಕ್ಕಿ', 'ದುರಂತ', 'ಬಲಿ', 'ಗಾಯ',
        'ప్రమాదం', 'ఢీకొట్టింది', 'దుర్మరణం',
        'விபத்து', 'மோதியது', 'உயிரிழப்பு'
    ],
}

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
    ('murder',         ['गिरफ्तार', 'हत्यारा', 'हत्यारों को', 'टारगेट किलिंग', 'एनकाउंटर',
                        'गैंग वार', 'आरोपी गिरफ्तार', 'हमलावर', 'गोली चलाई']),
    ('rape',           ['बलात्कार आरोपी', 'दुष्कर्म आरोपी', 'दुष्कर्म का मामला']),
    ('theft',          ['चोरी का मामला', 'चोर गिरफ्तार', 'लूट का मामला']),
    ('fraud_cheating', ['धोखाधड़ी का मामला', 'ठग गिरफ्तार', 'साइबर ठगी']),
    ('accident',       ['सड़क दुर्घटना', 'हादसे में घायल', 'हादसे में मौत',
                        'ट्रक की टक्कर', 'कार दुर्घटना', 'बाइक हादसा']),
]

# Political context: if title matches these AND has no crime keyword → non_crime
_POLITICAL_TITLE_PAT = re.compile(
    r'(?:'
    r'\b(?:chief minister|prime minister|home minister|finance minister|cm\b|pm\b'
    r'|mla\b|mp\b|governor|president|vice president|cabinet|election|minister'
    r'|parliament|assembly|lok sabha|rajya sabha|inaugurated|launched|announced'
    r'|press conference|political|party|congress|bjp|aap|jdu|bsp|ysrcp'
    r'|rally|yatra|padyatra|manifesto|coalition|budget|ordinance|legislation)\b'
    r'|ಮುಖ್ಯಮಂತ್ರಿ|ಪ್ರಧಾನ\s*ಮಂತ್ರಿ|ಮಂತ್ರಿ|ಶಾಸಕ|ಸಂಸದ|ಸಿಎಂ|ವಿಧಾನಸಭೆ'
    r'|ಸರ್ಕಾರ|ಆಶೀರ್ವಾದ|ಸ್ವಾಗತ|ಚುನಾವಣೆ|ಅಧಿವೇಶನ|ಒಕ್ಕೂಟ|ಪಕ್ಷ'
    r'|मुख्यमंत्री|प्रधानमंत्री|मंत्री|विधायक|सांसद|राज्यपाल|सरकार|आशीर्वाद'
    r'|राजनीति|विधानसभा|चुनाव|राजनेता|नेता|पार्टी|भाजपा|कांग्रेस'
    r'|முதலமைச்சர்|மந்திரி|சட்டமன்ற|பாராளுமன்ற'
    r'|ముఖ్యమంత్రి|మంత్రి|శాసనసభ|పార్లమెంట్'
    r')',
    re.IGNORECASE
)

# Crime-action override: if these appear IN THE TITLE, suppress political filter
# (e.g. "CM arrested", "Minister shot dead" = real crime news)
_CRIME_ACTION_PAT = re.compile(
    r'\b(?:arrested|detained|killed|shot|stabbed|murdered|accused|chargesheeted'
    r'|fir|raped|abducted|kidnapped|robbed|looted|cheated|defrauded|held'
    r'|convicted|sentenced|jailed|nabbed|caught)\b'
    r'|(?:गिरफ्तार|गिरफ्त|आरोपी|मारा गया)'
    r'|(?:గిరఫ్తార్|హత్య)'
    r'|(?:ಬಂಧಿತ|ಕೊಲೆ|ಹತ್ಯೆ)'
    r'|(?:கைது|கொலை)',
    re.IGNORECASE
)


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a column as plain Python object-dtype Series (no Arrow backend)."""
    return (
        df.get(col, pd.Series([''] * len(df), index=df.index))
        .fillna('').astype(str).astype(object)
    )


def _contains(series: pd.Series, pat: re.Pattern) -> pd.Series:
    """
    Safe str.contains wrapper.

    Streamlit Cloud (pandas >= 2 + pyarrow) uses Arrow-backed string arrays
    that reject compiled re.Pattern objects. We pass the raw pattern string
    and flags separately, which works on every backend.
    """
    # Strip re.UNICODE from flags — pandas str.contains doesn't accept it
    flags = pat.flags & ~re.UNICODE
    return series.str.contains(pat.pattern, flags=flags, regex=True, na=False)


def _heuristic_classify(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """Fast vectorised keyword classifier with political-context suppression.

    Fixes on Streamlit Cloud (PyArrow string backend):
    - Series cast to object dtype before regex ops
    - Pattern strings + flags= passed instead of compiled re.Pattern objects
    """
    logger.info("Running fast Heuristic Keyword Classification (%d articles)...", len(df))

    for cat in CRIME_CATEGORIES:
        if cat not in df.columns:
            df[cat] = 0

    # Build object-dtype series (avoids PyArrow backend issues)
    title_series = _safe_series(df, 'title').str.lower()
    corpus = (
        title_series
        + " "
        + _safe_series(df, 'text').str.lower()
        + " "
        + _safe_series(df, text_col).str.lower()
    ).str.replace(' nan ', ' ', regex=False).astype(object)
    title_series = title_series.astype(object)

    # Precompile per-category patterns
    cat_patterns = {
        cat: re.compile('|'.join(re.escape(w) for w in words), re.IGNORECASE)
        for cat, words in _HEURISTIC_KEYWORDS.items()
    }

    crime_mask = pd.Series(False, index=df.index)
    crime_action_in_title = _contains(title_series, _CRIME_ACTION_PAT)

    for cat, pat in cat_patterns.items():
        matched_full = _contains(corpus, pat)
        if not matched_full.any():
            continue

        matched_in_title = _contains(title_series, pat)
        political_title  = _contains(title_series, _POLITICAL_TITLE_PAT)

        # Suppress if: crime only in body, political title, no crime-action override
        suppressed = matched_full & ~matched_in_title & political_title & ~crime_action_in_title
        confirmed  = matched_full & ~suppressed

        df.loc[confirmed, cat] = 1
        crime_mask |= confirmed

    # Indicator phrases fallback for still-unmatched rows
    no_crime_idx = df.index[~crime_mask]
    if len(no_crime_idx) > 0:
        political_mask = _contains(title_series, _POLITICAL_TITLE_PAT)
        for cat, phrases in _CRIME_INDICATOR_PHRASES:
            pat = re.compile('|'.join(re.escape(p) for p in phrases), re.IGNORECASE)
            matched     = _contains(corpus.loc[no_crime_idx], pat)
            matched_idx = matched[matched].index
            if len(matched_idx) == 0:
                continue
            title_match      = _contains(title_series.loc[matched_idx], pat)
            is_political     = political_mask.loc[matched_idx]
            has_crime_action = crime_action_in_title.loc[matched_idx]
            confirmed_idx    = matched_idx[title_match | ~is_political | has_crime_action]
            if len(confirmed_idx) > 0:
                df.loc[confirmed_idx, cat] = 1
                crime_mask.loc[confirmed_idx] = True
                no_crime_idx = df.index[~crime_mask]

    df.loc[~crime_mask, 'non_crime'] = 1
    df.loc[crime_mask,  'non_crime'] = 0

    crime_count = int(crime_mask.sum())
    logger.info("Heuristic classification done: %d crime / %d non-crime out of %d articles.",
                crime_count, len(df) - crime_count, len(df))
    return df


def classify_articles(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """Classify articles — uses heuristic by default, AI model if opted in."""
    if df.empty:
        return df

    for cat in CRIME_CATEGORIES:
        df[cat] = 0

    try:
        classifier = get_zeroshot_classifier()
        if classifier == "HEURISTIC":
            return _heuristic_classify(df, text_col)

        labels = list(DESCRIPTIVE_LABELS.values())
        label_to_key = {v: k for k, v in DESCRIPTIVE_LABELS.items()}
        total = len(df)
        progress_bar = st.progress(0, text=f"AI Classification in progress... (0/{total} articles)")

        for i, (idx, row) in enumerate(df.iterrows()):
            if i % 2 == 0 or i == total - 1:
                progress_bar.progress(
                    min((i + 1) / total, 1.0),
                    text=f"AI Classification in progress... ({i + 1}/{total} articles)"
                )
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

            except Exception as inner_e:
                logger.warning(f"Failed to classify article at index {idx}: {inner_e}")
                raw_text = (
                    str(row.get('title', '')) + " " + str(row.get('text', ''))
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
