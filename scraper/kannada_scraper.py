import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from utils.helpers import setup_logging
import urllib.parse

logger = setup_logging()

class KannadaWebScraper:
    """
    Direct HTML scraper for Kannada news channels, bypassing RSS feeds.
    Provides real-time snippets for the Live News Feed and Geospatial map.
    """
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
        }
        
        self.sources = {
            "Prajavani": "https://www.prajavani.net/news",
            "Udayavani": "https://www.udayavani.com/news_menu/crime/685541b8d3279fb061cfdb24?lang=kn",
            "Vijayakarnataka": "https://vijaykarnataka.com/news/karnataka/",
            "Vijayavani": "https://www.vijayavani.net/"
        }

    def _scrape_site(self, source_name: str, url: str) -> list:
        records = []
        seen_titles = set()
        
        try:
            logger.info(f"Scraping HTML for {source_name} from {url}")
            response = requests.get(url, headers=self.headers, timeout=5)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # First, try to extract from heading tags (h2, h3, h4) which usually contain true headlines
            headings = soup.find_all(['h2', 'h3', 'h4'])
            valid_articles = []
            
            for h in headings:
                a_tag = h.find('a', href=True)
                if not a_tag:
                    # Sometimes the heading IS the link or wrapped by link
                    if h.name == 'a' and h.has_attr('href'):
                        a_tag = h
                    else:
                        parent = h.find_parent('a')
                        if parent and parent.has_attr('href'):
                            a_tag = parent
                
                if a_tag:
                    headline = h.get_text(strip=True)
                    url_href = a_tag['href']
                    
                    # A true headline in Kannada is usually longer than 15-20 characters
                    if len(headline) > 20 and 'javascript' not in url_href.lower() and 'tel:' not in url_href.lower():
                        valid_articles.append((headline, url_href))
                        
            # Fallback: if heading extraction failed (yielding < 2 articles), try finding long generic links
            if len(valid_articles) < 2:
                for a_tag in soup.find_all('a', href=True):
                    headline = a_tag.get_text(strip=True)
                    if len(headline) >= 25 and 'javascript' not in a_tag['href'].lower() and 'ಮತ್ತಷ್ಟು' not in headline:
                        valid_articles.append((headline, a_tag['href']))

            for headline, url_href in valid_articles:
                link = urllib.parse.urljoin(url, url_href)
                if headline not in seen_titles:
                    seen_titles.add(headline)
                    records.append({
                        "title": headline,
                        "text": headline,
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "url": link,
                        "source": source_name,
                    })
                        
        except Exception as e:
            logger.warning(f"Failed to scrape HTML for {source_name}: {e}")
            
        logger.info(f"  -> Extracted {len(records)} potential articles from {source_name}")
        return records

    def scrape_all(self, max_per_source: int = 20) -> pd.DataFrame:
        """
        Scrape top headlines directly from all configured Kannada news sites' HTML.
        Returns a DataFrame compatible with the main pipeline.
        """
        all_records = []
        for name, url in self.sources.items():
            records = self._scrape_site(name, url)
            # Cap at max_per_source
            all_records.extend(records[:max_per_source])
            
        if not all_records:
            logger.warning("No HTML articles fetched from any Kannada site.")
            return pd.DataFrame(columns=["title", "text", "date", "url", "source"])

        df = pd.DataFrame(all_records)
        logger.info(f"Total HTML live articles fetched: {len(df)}")
        return df
