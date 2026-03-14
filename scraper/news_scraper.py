import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from utils.helpers import setup_logging

logger = setup_logging()

class NewsScraper:
    def __init__(self):
        # We will configure basic generic headers to avoid instant 403s
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def scrape_article(self, url):
        """Attempts to extract text form an arbitrary news url using generic HTML tags."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Naive heuristics: h1 is title, p tags contain text
            title_tag = soup.find('h1')
            title = title_tag.text.strip() if title_tag else "Unknown Title"

            paragraphs = soup.find_all('p')
            text = ' '.join([p.text.strip() for p in paragraphs if len(p.text.strip()) > 30])

            # In a genuine deployment, we would parse exact meta tags for dates.
            # Using current date for naive scrape
            date_published = datetime.now().strftime('%Y-%m-%d')

            return {
                'title': title,
                'text': text,
                'date': date_published,
                'url': url
            }
        except Exception as e:
            logger.warning(f"Error scraping {url}: {e}")
            return None

    def scrape_list(self, urls, output_path="data/raw_news.csv"):
        """Scrapes a list of urls and saves to csv."""
        logger.info(f"Starting scrape process for {len(urls)} URLs...")
        data = []
        for url in urls:
            article_data = self.scrape_article(url)
            if article_data and article_data['text']:
                data.append(article_data)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} scraped articles to {output_path}")
        return df
