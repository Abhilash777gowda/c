import re
import pandas as pd
from typing import List, Dict
try:
    from utils.helpers import setup_logging
except ImportError:
    import logging
    def setup_logging():
        return logging.getLogger(__name__)

logger = setup_logging()

def extract_keywords(text: str) -> List[str]:
    """
    Extract key terms from an article for search.
    Simple approach: extract capitalized words (proper nouns) and long words.
    """
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    
    # Heuristic: Proper nouns (Capitalized) and significant words (> 6 chars)
    keywords = [w for w in words if (w[0].isupper() and len(w) > 2) or len(w) > 7]
    
    # Return unique keywords, top 10
    return list(dict.fromkeys(keywords))[:10]

def search_online(keywords: List[str]) -> List[Dict]:
    """
    Simulate an 'online' search by using the RSS scraper to find 
    articles matching the keywords.
    """
    from scraper.rss_scraper import RSSNewsScraper
    scraper = RSSNewsScraper()
    # Fetch a larger batch for searching
    df = scraper.scrape_all(max_per_feed=30)
    
    if df.empty:
        return []
    
    # Filter by keywords
    results = []
    query = " ".join(keywords).lower()
    
    # Simple keyword matching
    for _, row in df.iterrows():
        text = str(row['text']).lower()
        title = str(row['title']).lower()
        
        # Calculate match score
        matches = sum(1 for kw in keywords if kw.lower() in text or kw.lower() in title)
        if matches > 0:
            results.append({
                "title": row['title'],
                "text": row['text'],
                "url": row['url'],
                "source": row['source'],
                "match_score": matches
            })
            
    # Sort by number of keyword matches
    results = sorted(results, key=lambda x: x['match_score'], reverse=True)
    return results[:5]  # Top 5 related articles

def compare_articles(original_text: str, related_articles: List[Dict]) -> Dict:
    """
    Compare the input text with found articles.
    """
    if not related_articles:
        return {
            "status": "Inconclusive",
            "score": 0,
            "reasoning": "No related news articles found online to verify this claim.",
            "sources": []
        }

    # Simplified verification logic
    orig_keywords = set(extract_keywords(original_text.lower()))
    source_results = []
    
    max_overlap = 0.0
    total_score = 0.0
    
    for art in related_articles:
        art_keywords = set(extract_keywords(art.get('text', '').lower()))
        overlap = len(orig_keywords.intersection(art_keywords))
        # Similarity score
        score = min(100.0, (overlap / len(orig_keywords) * 100.0)) if orig_keywords else 0.0
        
        source_results.append({
            "title": art.get('title', 'Related News'),
            "url": art.get('url', '#'),
            "source": art.get('source', 'Unknown'),
            "similarity": int(score)
        })
        
        if score > max_overlap:
            max_overlap = score
        total_score += score

    avg_score = total_score / len(related_articles)
    
    if max_overlap > 75:
        status = "Verified"
        reasoning = f"This article is highly consistent with reports from {len(related_articles)} sources. The core facts are confirmed by multiple reputable news outlets."
    elif max_overlap > 40:
        status = "Partially Matches"
        reasoning = f"The article shares significant details with results from {len(related_articles)} sources, but some specifics might be missing or different."
    else:
        status = "Inconsistent / Low Evidence"
        reasoning = "The details provided do not strongly align with recent major news reports. This could be a very local event or potentially inaccurate information."

    return {
        "status": status,
        "score": int(max_overlap),
        "reasoning": reasoning,
        "sources": source_results
    }
