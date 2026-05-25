import sys, os
sys.path.insert(0, '.')
import pandas as pd
from scraper.rss_scraper import RSSNewsScraper
from utils.classifier_inference import _HEURISTIC_KEYWORDS, _heuristic_classify, CRIME_CATEGORIES

sc = RSSNewsScraper()
df = sc.scrape_all(max_per_feed=10)

# Manually count how many have crime keywords in raw title+text
crime_count = 0
for _, row in df.iterrows():
    txt = (str(row.get('title','')) + ' ' + str(row.get('text',''))).lower()
    for cat, words in _HEURISTIC_KEYWORDS.items():
        if any(w in txt for w in words):
            crime_count += 1
            break

print(f"Direct keyword match: {crime_count}/{len(df)} articles")

# Now run the actual classifier
for cat in CRIME_CATEGORIES:
    df[cat] = 0

if 'text' not in df.columns:
    df['text'] = df['title']

result = _heuristic_classify(df.copy())

crime_df = result[result['non_crime'] == 0]
non_crime_df = result[result['non_crime'] == 1]
print(f"After _heuristic_classify: CRIME={len(crime_df)}, NON-CRIME={len(non_crime_df)}")

print("\n=== CRIME articles detected ===")
for _, row in crime_df.head(8).iterrows():
    cats = [c for c in CRIME_CATEGORIES if c != 'non_crime' and row.get(c, 0) == 1]
    label = ", ".join(cats)
    print(f"  [{label}] {str(row['title'])[:90]}")

print("\n=== NON-CRIME sample (first 5) ===")
for _, row in non_crime_df.head(5).iterrows():
    print(f"  {str(row['title'])[:90]}")
