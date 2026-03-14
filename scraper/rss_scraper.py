import feedparser
import pandas as pd
from datetime import datetime
from utils.helpers import setup_logging

logger = setup_logging()

# Free RSS feeds from major Indian news sources - no API key needed
RSS_FEEDS = {
    "NDTV India": "https://feeds.feedburner.com/ndtvnews-india-news",
    "Times of India": "https://timesofindia.indiatimes.com/rssfeedmostread.cms",
    "The Hindu": "https://www.thehindu.com/news/national/feeder/default.rss",
    "India Today": "https://www.indiatoday.in/rss/1206551",
    "Hindustan Times": "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml",
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

        for source, url in self.feeds.items():
            try:
                logger.info(f"Fetching RSS feed: {source} ({url})")
                feed = feedparser.parse(url)
                fetched = 0
                for entry in feed.entries[:max_per_feed]:
                    record = self._entry_to_dict(entry, source)
                    if record["url"] in seen_urls or not record["text"]:
                        continue
                    seen_urls.add(record["url"])
                    records.append(record)
                    fetched += 1
                logger.info(f"  → {fetched} articles from {source}")
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
