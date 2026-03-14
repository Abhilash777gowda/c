import argparse
import pandas as pd
import os
from utils.helpers import setup_logging
from utils.data_annotator import generate_synthetic_dataset, CRIME_CATEGORIES
from scraper.news_scraper import NewsScraper
# RSSNewsScraper imported lazily inside --use-rss branch to avoid feedparser dependency at startup
from preprocessing.text_cleaner import TextCleaner
from models.svm_classifier import BaselineSVM
from models.bilstm_classifier import CustomBiLSTMClassifier
from models.transformer_classifier import TransformerClassifier
from models.xlmroberta_classifier import XLMRoBertaClassifier
from models.muril_classifier import MuRILClassifier
from analysis.trend_analysis import TrendAnalyzer
from analysis.correlation_analysis import CorrelationValidator
from data.ncrb_data import generate_ncrb_csv

logger = setup_logging()


def split_by_language(df: pd.DataFrame):
    """
    Route articles to language-appropriate models.
    Hindi/Indian-language articles → MuRIL
    English articles → XLM-RoBERTa / mBERT
    """
    if 'lang' not in df.columns:
        df['lang'] = 'en'

    indian_langs = {'hi', 'mr', 'bn', 'ta', 'te', 'kn', 'ml', 'gu', 'pa'}
    mask_hindi = df['lang'].isin(indian_langs)
    df_hindi = df[mask_hindi].copy().reset_index(drop=True)
    df_english = df[~mask_hindi].copy().reset_index(drop=True)

    logger.info(f"Language split → English/multilingual: {len(df_english)} | "
                f"Indian-language (Hindi/regional): {len(df_hindi)}")
    return df_english, df_hindi


def main():
    parser = argparse.ArgumentParser(description="CRIMSON-India Execution Pipeline")
    parser.add_argument("--use-synthetic", action="store_true",
                        help="Use synthetic data to bypass network scraping")
    parser.add_argument("--use-rss", action="store_true",
                        help="Use live RSS feeds for real-time data")
    parser.add_argument("--skip-muril", action="store_true",
                        help="Skip MuRIL training (saves time if no Hindi data)")
    parser.add_argument("--skip-xlmr", action="store_true",
                        help="Skip XLM-RoBERTa training (faster runs)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs for Transformer models")
    args = parser.parse_args()

    # Create output directories
    for d in ['data', 'plots', 'models/saved_transformer',
              'models/saved_xlmroberta', 'models/saved_muril']:
        os.makedirs(d, exist_ok=True)

    print("\n==============================================")
    print(" CRIMSON-India Framework Pipeline Initiated")
    print("==============================================\n")

    # ── Step 0: Ensure NCRB reference data exists ────────────────────────────
    if not os.path.exists("data/ncrb_stats.csv"):
        logger.info("Generating NCRB reference statistics CSV...")
        generate_ncrb_csv("data/ncrb_stats.csv")

    # ─────────────────────────────────────────────────────────────────────────
    # Steps 1-3: Data Acquisition, Preprocessing & Annotation
    # ─────────────────────────────────────────────────────────────────────────
    cleaner = TextCleaner()

    if args.use_synthetic:
        logger.info("Running in Synthetic Mode (End-to-End Pipeline Demo)")
        df = generate_synthetic_dataset("data/labeled_news.csv", num_samples=300)

    elif args.use_rss:
        logger.info("Running in Real-time RSS Mode")
        rss = RSSNewsScraper()
        df_raw = rss.scrape_to_csv("data/raw_news.csv", max_per_feed=20)
        df_raw['clean_text'] = df_raw['text'].apply(cleaner.clean_text)
        if hasattr(cleaner, 'detect_language'):
            df_raw['lang'] = df_raw['clean_text'].apply(cleaner.detect_language)
        # Fall back to synthetic labels since live data is unsupervised
        logger.info("Live scrape complete — generating synthetic labels for pipeline demo.")
        df = generate_synthetic_dataset("data/labeled_news.csv", num_samples=300)

    else:
        logger.info("Running in Scrape Mode (BeautifulSoup)")
        scraper = NewsScraper()
        demo_urls = [
            "https://timesofindia.indiatimes.com/world",
            "https://www.thehindu.com/news/",
        ]
        scraper.scrape_list(demo_urls, "data/raw_news.csv")
        cleaner.clean_dataset("data/raw_news.csv", "data/clean_news.csv")
        # Fall back to synthetic labels
        df = generate_synthetic_dataset("data/labeled_news.csv", num_samples=300)

    # ─────────────────────────────────────────────────────────────────────────
    # Train/Test split
    # ─────────────────────────────────────────────────────────────────────────
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split = int(0.8 * len(df_shuffled))
    train_df = df_shuffled.iloc[:split]
    test_df  = df_shuffled.iloc[split:]
    logger.info(f"Train: {len(train_df)} | Test: {len(test_df)}")

    # Language routing for transformer models
    train_en, train_hi = split_by_language(train_df)
    test_en,  test_hi  = split_by_language(test_df)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Baseline Models
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 4a. Baseline: SVM + TF-IDF ---")
    svm = BaselineSVM(CRIME_CATEGORIES)
    svm.train(train_df)
    svm.evaluate(test_df)
    svm.save()

    print("\n--- 4b. Baseline: BiLSTM (FastText embeddings) ---")
    bilstm = CustomBiLSTMClassifier(CRIME_CATEGORIES, use_fasttext=True)
    bilstm.train(train_df, epochs=3)
    bilstm.evaluate(test_df)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: Transformer Models (Core)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 5a. Core Transformer: mBERT (bert-base-multilingual-cased) ---")
    transformer = TransformerClassifier(CRIME_CATEGORIES,
                                        model_name="bert-base-multilingual-cased")
    transformer.train(train_df, epochs=args.epochs)
    transformer.evaluate(test_df)
    transformer.save()

    if not args.skip_xlmr:
        print("\n--- 5b. Core Transformer: XLM-RoBERTa (xlm-roberta-base) ---")
        xlmr = XLMRoBertaClassifier(CRIME_CATEGORIES)
        # Use English / multilingual split for training XLM-R
        train_xlmr = train_en if len(train_en) >= 10 else train_df
        test_xlmr  = test_en  if len(test_en)  >= 5  else test_df
        xlmr.train(train_xlmr, epochs=args.epochs)
        xlmr.evaluate(test_xlmr)
        xlmr.save()
    else:
        logger.info("Skipping XLM-RoBERTa (--skip-xlmr flag set).")

    if not args.skip_muril:
        print("\n--- 5c. Hindi Model: MuRIL (google/muril-base-cased) ---")
        muril = MuRILClassifier(CRIME_CATEGORIES)
        # Use Hindi/regional split; fall back to full dataset if insufficient
        train_muril = train_hi if len(train_hi) >= 10 else train_df
        test_muril  = test_hi  if len(test_hi)  >= 5  else test_df
        muril.train(train_muril, epochs=args.epochs)
        muril.evaluate(test_muril)
        muril.save()
    else:
        logger.info("Skipping MuRIL (--skip-muril flag set).")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 6: Prediction Pipeline (best model = XLM-RoBERTa or mBERT fallback)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 6. Prediction Pipeline ---")
    best_model = xlmr if not args.skip_xlmr else transformer
    sample_texts = test_df['clean_text'].head(3).tolist()
    sample_preds = best_model.predict(sample_texts)
    for text, labels in zip(sample_texts, sample_preds):
        print(f"  Article: {text[:70]}...")
        print(f"  Predicted: {labels}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 7: Trend Analysis
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 7. Trend Analysis ---")
    analyzer = TrendAnalyzer(CRIME_CATEGORIES)
    monthly_trends = analyzer.generate_trends(df, output_path="plots/crime_trends.png")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 8: Correlation Validation (vs NCRB stats)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 8. Correlation Validation (Pearson r vs NCRB) ---")
    validator = CorrelationValidator(CRIME_CATEGORIES)
    validator.calculate_correlation(monthly_trends)

    print("\n==============================================")
    print(" CRIMSON-India Framework Pipeline Completed!")
    print("==============================================\n")


if __name__ == "__main__":
    main()
