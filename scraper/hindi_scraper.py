import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from utils.helpers import setup_logging

logger = setup_logging()

class HindiWebScraper:
    """
    Scrapes Hindi news from Amar Ujala and Dainik Bhaskar.
    Uses direct HTML parsing for better reliability than RSS.
    """
    SOURCES = {
        "Amar Ujala": "https://www.amarujala.com/india-news",
        "Dainik Bhaskar": "https://www.bhaskar.com/national/"
    }

    def scrape_source(self, source_name, url, max_results=10):
        articles = []
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                logger.error(f"Failed to fetch {source_name}: Status {response.status_code}")
                return []

            soup = BeautifulSoup(response.text, "html.parser")
            
            # Universal extraction targeting common news link patterns
            links = soup.find_all("a", href=True)
            for link in links:
                if len(articles) >= max_results:
                    break
                    
                title = ""
                # Search for titles inside headings or spans
                heading = link.find(['h1', 'h2', 'h3', 'h4', 'span'])
                if heading:
                    title = heading.get_text().strip()
                elif link.get('title'):
                    title = link.get('title').strip()
                else:
                    title = link.get_text().strip()
                
                href = link.get('href')
                if not href.startswith('http'):
                    base = "https://www.amarujala.com" if "amarujala" in url else "https://www.bhaskar.com"
                    href = base + (href if href.startswith('/') else '/' + href)

                if len(title) > 25 and href not in [a['url'] for a in articles]:
                    articles.append({
                        "title": title,
                        "text": title,  # Use title as text for classification if snippet is unavailable
                        "url": href,
                        "source": source_name,
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "language": "hi"
                    })
            
            logger.info(f"Scraped {len(articles)} articles from {source_name}")
        except Exception as e:
            logger.error(f"Error scraping {source_name}: {e}")
            
        return articles

    def scrape_all(self, max_per_source=10):
        all_articles = []
        for name, url in self.SOURCES.items():
            all_articles.extend(self.scrape_source(name, url, max_per_source))
        return pd.DataFrame(all_articles)

if __name__ == "__main__":
    scraper = HindiWebScraper()
    df = scraper.scrape_all()
    print(df.head())
