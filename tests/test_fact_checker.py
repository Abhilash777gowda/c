from utils.fact_checker import extract_keywords, compare_articles
import pandas as pd

def test_fact_checker():
    # Mock data
    original_text = "A serious car accident occurred in Delhi near the Red Fort today involving three vehicles."
    keywords = extract_keywords(original_text)
    print(f"Keywords: {keywords}")
    
    related_mock = [
        {
            "title": "Major pile-up in Delhi: Three cars collide near Red Fort",
            "text": "A massive traffic jam was reported in Delhi today after three vehicles were involved in a collision near the historic Red Fort. Rescue teams are on site.",
            "url": "http://example.com/news1",
            "source": "Mock Times"
        }
    ]
    
    results = compare_articles(original_text, related_mock)
    print(f"Status: {results['status']}")
    print(f"Score: {results['score']}%")
    print(f"Reasoning: {results['reasoning']}")
    
    assert results['score'] > 50
    print("Test passed!")

if __name__ == "__main__":
    test_fact_checker()
