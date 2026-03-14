import re
import pandas as pd
from utils.helpers import setup_logging

logger = setup_logging()

# Try importing langdetect; graceful fallback if not installed
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not installed. Language detection will be skipped. "
                   "Run: pip install langdetect")

SUPPORTED_LANGS = {"en", "hi", "mr", "bn", "ta", "te", "kn", "ml", "gu", "pa"}


class TextCleaner:
    def __init__(self):
        pass

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        # Lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove special characters (keep alphanumeric + space + Devanagari script)
        text = re.sub(r'[^a-z0-9\s\u0900-\u097F]', '', text)
        # Collapse whitespace
        tokens = text.split()
        return ' '.join(tokens)

    def detect_language(self, text: str) -> str:
        """
        Detect the language of a text snippet.
        Returns an ISO 639-1 code (e.g. 'en', 'hi') or 'unknown'.
        Requires langdetect: pip install langdetect
        """
        if not LANGDETECT_AVAILABLE or not text or len(text.split()) < 5:
            return "en"  # Assume English if cannot detect
        try:
            lang = detect(text)
            return lang if lang in SUPPORTED_LANGS else "en"
        except Exception:
            return "en"

    def clean_dataset(self, input_path: str = "data/raw_news.csv",
                      output_path: str = "data/clean_news.csv") -> pd.DataFrame:
        logger.info(f"Cleaning dataset from {input_path}...")
        try:
            df = pd.read_csv(input_path)
            if 'text' not in df.columns:
                raise ValueError("Expected 'text' column in dataset.")

            df['clean_text'] = df['text'].apply(self.clean_text)
            df = df[df['clean_text'].str.len() > 0]

            # Language detection — routes articles to correct model downstream
            if LANGDETECT_AVAILABLE:
                logger.info("Running language detection on articles...")
                df['lang'] = df['clean_text'].apply(self.detect_language)
                lang_counts = df['lang'].value_counts().to_dict()
                logger.info(f"Language distribution: {lang_counts}")
            else:
                df['lang'] = 'en'

            df.to_csv(output_path, index=False)
            logger.info(f"Cleaned dataset saved to {output_path} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.error(f"Failed to clean dataset: {e}")
            raise
