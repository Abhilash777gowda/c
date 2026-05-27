import feedparser
import pandas as pd
from datetime import datetime
from utils.helpers import setup_logging

logger = setup_logging()

# Expanded RSS feeds including standard and regional versions of requested papers
RSS_FEEDS = {
    # TOI (verified — serves crime and general articles)
    "TOI Crime":            "https://timesofindia.indiatimes.com/rssfeeds/1081479906.cms",
    "Times of India":       "https://timesofindia.indiatimes.com/rssfeeds/2947300.cms",
    
    # The Hindu (National News)
    "The Hindu":            "https://www.thehindu.com/news/national/feeder/default.rss",
    
    # Amar Ujala verified crime news (Hindi)
    "Amar Ujala Crime":     "https://www.amarujala.com/rss/crime.xml",
    
    # National India news for diversity
    "NDTV India":           "https://feeds.feedburner.com/ndtvnews-india-news",
    "Hindustan Times":      "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml",
    
    # Regional / Karnataka News (Prajavani, Vijay Karnataka, Udayavani, Vijayavani)
    "Prajavani":            "https://www.prajavani.net/rssfeeds/karnataka.xml",
    "Vijay Karnataka":      "https://vijaykarnataka.com/rssfeeds/21283013.cms",
    "Udayavani":            "https://www.udayavani.com/category/state/feed",
    "Vijayavani":           "https://www.vijayavani.net/feed",
    
    # Other regional feeds
    "OneIndia Tamil":       "https://tamil.oneindia.com/rss/tamil-news-fb.xml",
    "OneIndia Telugu":      "https://telugu.oneindia.com/rss/telugu-news-fb.xml",
    "OneIndia Kannada":     "https://kannada.oneindia.com/rss/kannada-news-fb.xml",
}


class RSSNewsScraper:
    """
    Scrapes real-time Indian news articles from RSS feeds.
    No API key required — uses publicly available RSS endpoints.
    """

    def __init__(self, feeds: dict = None):
        self.feeds = feeds or RSS_FEEDS

    def _parse_date(self, entry) -> str:
        """Try to extract a clean YYYY-MM-DD date from a feed entry."""
        for attr in ("published_parsed", "updated_parsed"):
            t = getattr(entry, attr, None)
            if t:
                try:
                    return datetime(*t[:6]).strftime("%Y-%m-%d")
                except Exception:
                    pass
        return datetime.now().strftime("%Y-%m-%d")

    def _entry_to_dict(self, entry, source: str) -> dict:
        title = getattr(entry, "title", "").strip()
        summary = getattr(entry, "summary", "").strip()
        link = getattr(entry, "link", "").strip()
        date = self._parse_date(entry)

        # Combine title + summary as the article body for classification
        text = f"{title}. {summary}".strip()
        return {
            "title": title,
            "text": text,
            "date": date,
            "url": link,
            "source": source,
        }

    def scrape_all(self, max_per_feed: int = 20) -> pd.DataFrame:
        """
        Scrape up to `max_per_feed` articles from each configured RSS feed.
        Returns a DataFrame with columns: title, text, date, url, source.
        """
        records = []
        seen_urls = set()

        import requests
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        
        for source, url in self.feeds.items():
            try:
                logger.info(f"Fetching RSS feed: {source} ({url})")
                
                resp = requests.get(url, headers=headers, timeout=5)
                if resp.status_code != 200:
                    logger.warning(f"Status {resp.status_code} for {source}")
                    continue
                    
                feed = feedparser.parse(resp.content)
                fetched = 0
                for entry in feed.entries[:max_per_feed]:
                    record = self._entry_to_dict(entry, source)
                    if record["url"] in seen_urls or not record["text"]:
                        continue
                    seen_urls.add(record["url"])
                    records.append(record)
                    fetched += 1
                logger.info(f"  -> {fetched} articles from {source}")
            except Exception as e:
                logger.warning(f"Failed to fetch {source}: {e}")

        if not records:
            logger.warning("No articles fetched from any RSS feed.")
            return pd.DataFrame(columns=["title", "text", "date", "url", "source"])

        df = pd.DataFrame(records)
        logger.info(f"Total real-time articles fetched: {len(df)}")
        return df

    def scrape_to_csv(self, output_path: str = "data/raw_news.csv", max_per_feed: int = 20) -> pd.DataFrame:
        """Scrape and save raw articles to CSV."""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = self.scrape_all(max_per_feed=max_per_feed)
        if not df.empty:
            df.to_csv(output_path, index=False)
            logger.info(f"Real-time articles saved to {output_path} ({len(df)} rows)")
        return df
