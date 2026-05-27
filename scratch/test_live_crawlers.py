import sys
import os
import pandas as pd

sys.path.append('.')

# Set FORCE_HEURISTIC to true for fast lightweight local testing
os.environ['FORCE_HEURISTIC'] = 'true'

from scraper.kannada_scraper import KannadaWebScraper
from scraper.hindi_scraper import HindiWebScraper
from scraper.regional_scraper import RegionalWebScraper
from utils.classifier_inference import classify_articles, CRIME_CATEGORIES

print("Running direct HTML crime web scrapers...")
df_kan = KannadaWebScraper().scrape_all(max_per_source=8)
df_hin = HindiWebScraper().scrape_all(max_per_source=8)
df_reg = RegionalWebScraper().scrape_all(max_per_source=8)

df_raw = pd.concat([df_kan, df_hin, df_reg], ignore_index=True)

if df_raw.empty:
    print("Error: No articles fetched by HTML scrapers.")
    sys.exit(1)

print(f"Successfully scraped {len(df_raw)} raw articles directly from regional crime sections.")

# Clean text
from preprocessing.text_cleaner import TextCleaner
cleaner = TextCleaner()
df_raw['clean_text'] = df_raw['title'].apply(cleaner.clean_text)
df_raw = df_raw[df_raw['clean_text'].str.len() > 0].reset_index(drop=True)

print("Classifying direct HTML regional crime articles...")
df_classified = classify_articles(df_raw)

# Save and analyze results
output_lines = []
output_lines.append(f"=== DIRECT REGIONAL CRIME HTML SCRAPERS RESULT ({len(df_classified)} articles) ===")

crime_count = 0
category_stats = {cat: 0 for cat in CRIME_CATEGORIES if cat != 'non_crime'}

for idx, row in df_classified.iterrows():
    matched = [cat.upper() for cat in CRIME_CATEGORIES if row.get(cat, 0) == 1]
    is_crime = any(cat != 'NON_CRIME' for cat in matched)
    if is_crime:
        crime_count += 1
        for cat in CRIME_CATEGORIES:
            if row.get(cat, 0) == 1 and cat != 'non_crime':
                category_stats[cat] += 1
    output_lines.append(f"[{', '.join(matched)}] {row['source']} | {row['title']}")

output_lines.append("\n--- Regional Crime Extraction Statistics ---")
output_lines.append(f"Total Crime/Accident Alerts: {crime_count} out of {len(df_classified)}")
for cat, cnt in category_stats.items():
    output_lines.append(f"  - {cat.upper()}: {cnt} matches")
output_lines.append("=================================================")

with open("scratch/live_crawlers_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print(f"Diagnostics complete. Found {crime_count} crime articles. Results saved to scratch/live_crawlers_results.txt")
