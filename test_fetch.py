import sys
import pandas as pd
from app import fetch_live_news
from utils.classifier_inference import classify_articles

sys.path.append('.')

df_raw = pd.DataFrame([
    {'title': 'Man shot dead in Delhi', 'summary': 'Police investigate...'},
    {'title': 'Road accident in Mumbai', 'summary': 'Two injured...'},
    {'title': 'बेंगलुरु में हत्या', 'summary': 'भयानक...'}
])
df_raw['clean_text'] = df_raw.apply(lambda x: x['title'] + ' ' + x['summary'], axis=1)

import os
os.environ['FORCE_HEURISTIC'] = 'true'

print("Classifying...")
df_classified = classify_articles(df_raw)
print(df_classified[['title', 'murder', 'accident', 'non_crime']])
