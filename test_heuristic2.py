import pandas as pd
import sys
import os

sys.path.append('.')

from scraper.rss_scraper import RSSNewsScraper
from scraper.regional_scraper import RegionalScraper
from utils.classifier_inference import classify_articles

def main():
    print("Fetching news...")
    articles = []
    
    # Fetch from one Kannada source
    scrapers = RegionalScraper.get_scrapers()
    for s in scrapers:
        if s.language == 'kn':
            articles.extend(s.scrape()[:5])
            break
            
    # Fetch from one Hindi source
    from scraper.hindi_scraper import HindiScraper
    hindi_scrapers = HindiScraper.get_scrapers()
    for s in hindi_scrapers:
        articles.extend(s.scrape()[:5])
        break
        
    df_raw = pd.DataFrame(articles)
    print(f"Fetched {len(df_raw)} articles.")
    
    if df_raw.empty:
        print("Empty.")
        return
        
    df_raw['clean_text'] = df_raw.apply(lambda x: str(x.get('title', '')) + " " + str(x.get('summary', '')), axis=1)
    
    # We will FORCE the heuristic classifier by setting the env variable
    os.environ['HOSTNAME'] = 'streamlit-cloud'
    
    print("Classifying...")
    df_classified = classify_articles(df_raw)
    
    # Let's see what it found
    crimes = df_classified[df_classified['non_crime'] == 0]
    print(f"Found {len(crimes)} crimes out of {len(df_raw)}.")
    
    for idx, row in df_raw.iterrows():
        text = row['clean_text']
        cats = [c for c in df_classified.columns if row.get(c, 0) == 1]
        print(f"[{','.join(cats)}] {text}")
        
if __name__ == "__main__":
    main()
