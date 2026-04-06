import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from utils.helpers import setup_logging

logger = setup_logging()

class RegionalWebScraper:
    """
    Scrapes Tamil news from Dina Thanthi and Telugu news from Sakshi.
    """
    TAMIL_SOURCES = {
        "Dina Thanthi": "https://www.dailythanthi.com/News/India"
    }
    TELUGU_SOURCES = {
        "Sakshi": "https://www.sakshi.com/national"
    }

    def scrape_source(self, source_name, url, lang, max_results=10):
        articles = []
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                logger.error(f"Failed to fetch {source_name}: Status {response.status_code}")
                return []

            soup = BeautifulSoup(response.text, "html.parser")
            
            # Sakshi specific: headlines often in 'h2' or 'h3'
            # Dina Thanthi specific: links with certain classes
            links = soup.find_all("a", href=True)
            for link in links:
                if len(articles) >= max_results:
                    break
                    
                title = ""
                # Attempt to find the most relevant title
                heading = link.find(['h1', 'h2', 'h3', 'h4'])
                if heading:
                    title = heading.get_text().strip()
                else:
                    title = link.get_text().strip()
                
                href = link.get('href')
                if not href.startswith('http'):
                    base = "https://www.sakshi.com" if "sakshi" in url else "https://www.dailythanthi.com"
                    href = base + (href if href.startswith('/') else '/' + href)

                # Filter out short menu items or boilerplate
                if len(title) > 20 and href not in [a['url'] for a in articles]:
                    articles.append({
                        "title": title,
                        "text": title,
                        "url": href,
                        "source": source_name,
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "language": lang
                    })
            
            logger.info(f"Scraped {len(articles)} articles from {source_name} ({lang})")
        except Exception as e:
            logger.error(f"Error scraping {source_name}: {e}")
            
        return articles

    def scrape_all(self, max_per_source=8):
        all_articles = []
        for name, url in self.TAMIL_SOURCES.items():
            all_articles.extend(self.scrape_source(name, url, "ta", max_per_source))
        for name, url in self.TELUGU_SOURCES.items():
            all_articles.extend(self.scrape_source(name, url, "te", max_per_source))
        return pd.DataFrame(all_articles)

if __name__ == "__main__":
    scraper = RegionalWebScraper()
    df = scraper.scrape_all()
    print(df.head())
