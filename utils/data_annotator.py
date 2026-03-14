import pandas as pd
import numpy as np
import random
from utils.helpers import setup_logging

logger = setup_logging()

CRIME_CATEGORIES = ['theft', 'assault', 'accident', 'drug_crime', 'cybercrime', 'non_crime']

def generate_synthetic_dataset(output_path="data/labeled_news.csv", num_samples=200):
    """
    Generates a synthetic labeled dataset mimicking Indian news crime articles.
    This helps us test the ML pipeline robustly without scraping real data constantly.
    """
    logger.info(f"Generating synthetic structured data ({num_samples} samples)...")
    
    # Vocabulary clusters related to categories to make TF-IDF / embeddings have SOME signal
    vocab = {
        'theft': ['stolen', 'robbery', 'thief', 'money', 'jewelry', 'house', 'break-in', 'snatched'],
        'assault': ['attacked', 'injured', 'hospital', 'beat', 'weapon', 'fight', 'police', 'arrested'],
        'accident': ['car', 'truck', 'highway', 'collision', 'dead', 'fatal', 'speeding', 'crashed'],
        'drug_crime': ['smuggling', 'heroin', 'cocaine', 'seized', 'narcotics', 'gang', 'raid'],
        'cybercrime': ['hacked', 'phishing', 'scam', 'online', 'bank', 'fraud', 'OTP', 'cyber'],
        'non_crime': ['election', 'minister', 'cricket', 'festival', 'celebration', 'movie', 'weather', 'economy']
    }
    
    data = []
    
    for i in range(num_samples):
        # Pick 1-2 primary categories
        num_cats = random.choice([1, 1, 1, 2, 2])
        chosen_cats = random.sample(CRIME_CATEGORIES, num_cats)
        
        # If non-crime is chosen, it should be the only label
        if 'non_crime' in chosen_cats:
            chosen_cats = ['non_crime']
            
        # Generate dummy text based on vocab
        text_words = []
        for cat in chosen_cats:
            text_words.extend(random.sample(vocab[cat], random.randint(3, 6)))
        
        # Add random noise words
        common_words = ['the', 'in', 'of', 'delhi', 'mumbai', 'police', 'said', 'today', 'man', 'woman']
        text_words.extend([random.choice(common_words) for _ in range(15)])
        random.shuffle(text_words)
        text = ' '.join(text_words)
        
        # Prepare row dict
        row = {
            'title': ' '.join(text_words[:3]).title(),
            'clean_text': text,
            'date': f"2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            'url': f"http://example.com/news/{i}"
        }
        
        # Setup multi-hot targets
        for cat in CRIME_CATEGORIES:
            row[cat] = 1 if cat in chosen_cats else 0
            
        data.append(row)
        
    df = pd.DataFrame(data)
    # Ensure directory exists just in case
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Synthetic dataset saved to {output_path}")
    return df
