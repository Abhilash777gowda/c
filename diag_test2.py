import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, '.')
import feedparser, requests

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
feeds = {
    'TOI Crime':        'https://timesofindia.indiatimes.com/rssfeeds/1081479906.cms',
    'News18 Crime':     'https://www.news18.com/rss/crime.xml',
    'India Today Crime':'https://www.indiatoday.in/rss/1206515',
    'Amar Ujala':       'https://www.amarujala.com/rss/crime.xml',
    'NDTV Crime':       'https://feeds.feedburner.com/ndtvnews-india-news',
    'HT Crime':         'https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml',
}
from utils.classifier_inference import _HEURISTIC_KEYWORDS, _CRIME_INDICATOR_PHRASES

titles = []
for name, url in feeds.items():
    try:
        r = requests.get(url, headers=headers, timeout=10)
        feed = feedparser.parse(r.content)
        for e in feed.entries[:8]:
            t = getattr(e, 'title', '')
            titles.append((name, t))
    except Exception as ex:
        print(f"ERROR {name}: {ex}")

print(f"Total sample: {len(titles)} titles")
matched = 0
for src, t in titles:
    txt = t.lower()
    found = False
    for cat, words in _HEURISTIC_KEYWORDS.items():
        if any(w in txt for w in words):
            print(f"  [{cat}] [{src}] {t[:80]}")
            found = True; matched += 1; break
    if not found:
        for cat, phrases in _CRIME_INDICATOR_PHRASES:
            if any(p in txt for p in phrases):
                print(f"  [{cat}/indicator] [{src}] {t[:80]}")
                found = True; matched += 1; break
    if not found:
        print(f"  [NON-CRIME] [{src}] {t[:80]}")
print(f"\nMatched {matched}/{len(titles)} as crime from titles alone")
