import sys
import os
import pandas as pd

sys.path.append('.')

# Set env override to trigger fast heuristic classification immediately
os.environ['FORCE_HEURISTIC'] = 'true'

from utils.classifier_inference import classify_articles, CRIME_CATEGORIES

# Define highly realistic multilingual crime reporting test headlines
test_articles = [
    # English headlines
    {'title': 'Man shot dead in Delhi by contract killers', 'text': 'The victim was found with multiple gunshot wounds in Central Delhi.'},
    {'title': 'Gold jewelry stolen from bank locker in Chennai', 'text': 'Theft occurred over the weekend, police registering case.'},
    {'title': 'Cyber fraud gang arrested for duplicate UPI app scam', 'text': 'Three men swindled over 50 Lakhs from unsuspecting citizens.'},
    {'title': 'Five dead in terrible head-on truck collision on national highway', 'text': 'A major accident occurred due to poor visibility.'},
    
    # Kannada headlines (requested papers style)
    {'title': 'ಬೆಂಗಳೂರು: ರಾಜಕೀಯ ದ್ವೇಷಕ್ಕೆ ಯುವಕನ ಭೀಕರ ಕೊಲೆ', 'text': 'ಹತ್ಯೆ ಆರೋಪಿಗಳನ್ನು ಪೊಲೀಸರು ತಡರಾತ್ರಿ ಬಂಧಿಸಿದ್ದಾರೆ.'},
    {'title': 'ಮೈಸೂರು ಬ್ಯಾಂಕ್‌ನಲ್ಲಿ ಭಾರಿ ದರೋಡೆ: ಚಿನ್ನಾಭರಣ ಲೂಟಿ', 'text': 'ದರೋಡೆಕೋರರು ಭದ್ರತಾ ಸಿಬ್ಬಂದಿಯನ್ನು ಕಟ್ಟಿಹಾಕಿ ಲೂಟಿ ಮಾಡಿದ್ದಾರೆ.'},
    {'title': 'ಕಾರು ಮತ್ತು ಲಾರಿ ನಡುವೆ ಭೀಕರ ಅಪಘಾತ: ಸ್ಥಳದಲ್ಲೇ ಇಬ್ಬರು ಸಾವು', 'text': 'ರಾಷ್ಟ್ರೀಯ ಹೆದ್ದಾರಿಯಲ್ಲಿ ಸಂಭವಿಸಿದ ಅಪಘಾತ.'},
    
    # Hindi headlines
    {'title': 'दिल्ली में दिनदहाड़े महिला की गोली मारकर हत्या', 'text': 'हत्यारों की तलाश में पुलिस ने पांच टीमें गठित कीं।'},
    {'title': 'घर में घुसे चोरों ने लाखों की नगदी और जेवर चोरी किए', 'text': 'सेंधमारी की इस वारदात से इलाके में सनसनी फैल गई।'},
    
    # General non-crime news for negative test
    {'title': 'India wins spectacular cricket match against Australia', 'text': 'The stadium was packed as the captain hit the winning runs.'},
    {'title': 'Stock market hits all-time high amid positive global cues', 'text': 'Investors made massive gains today.'}
]

df_raw = pd.DataFrame(test_articles)
df_raw['clean_text'] = df_raw.apply(lambda x: str(x['title']) + ' ' + str(x['text']), axis=1)

output_lines = []
output_lines.append("=== STARTING DIRECT HEURISTIC CLASSIFIER TESTING ===")
df_classified = classify_articles(df_raw)

output_lines.append("\n--- RESULTS ---")
for idx, row in df_classified.iterrows():
    # Identify which columns were set to 1
    matched_crimes = [cat.upper() for cat in CRIME_CATEGORIES if row.get(cat, 0) == 1]
    output_lines.append(f"[{', '.join(matched_crimes)}] Headline: {row['title']}")
output_lines.append("=================================================\n")

# Save results to a UTF-8 encoded text file
with open("scratch/classification_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print("Testing complete. Results saved to scratch/classification_results.txt")
