import sys
import os
import pandas as pd

sys.path.append('.')

# Set FORCE_HEURISTIC to true to run in local lightweight mode
os.environ['FORCE_HEURISTIC'] = 'true'

from scraper.rss_scraper import RSSNewsScraper
from utils.classifier_inference import classify_articles, CRIME_CATEGORIES

print("Fetching live articles from RSS feeds...")
scraper = RSSNewsScraper()
df_raw = scraper.scrape_all(max_per_feed=10)

if df_raw.empty:
    print("Error: No articles fetched. Please check internet connection.")
    sys.exit(1)

print(f"Successfully fetched {len(df_raw)} articles.")

# Clean text
from preprocessing.text_cleaner import TextCleaner
cleaner = TextCleaner()
df_raw['clean_text'] = df_raw['text'].apply(cleaner.clean_text)
df_raw = df_raw[df_raw['clean_text'].str.len() > 0].reset_index(drop=True)

print("Classifying fetched articles...")
df_classified = classify_articles(df_raw)

# Save results to review
output_lines = []
output_lines.append(f"=== LIVE RSS FEED CLASSIFICATION TEST ({len(df_classified)} articles) ===")

crime_count = 0
for idx, row in df_classified.iterrows():
    matched = [cat.upper() for cat in CRIME_CATEGORIES if row.get(cat, 0) == 1]
    is_crime = any(cat != 'NON_CRIME' for cat in matched)
    if is_crime:
        crime_count += 1
    output_lines.append(f"[{', '.join(matched)}] {row['source']} | {row['title']}")

output_lines.append(f"\nSummary: {crime_count} crime/accident articles found out of {len(df_classified)}")
output_lines.append("=================================================")

with open("scratch/live_rss_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print(f"Diagnostics complete. Found {crime_count} crime articles. Results saved to scratch/live_rss_results.txt")
