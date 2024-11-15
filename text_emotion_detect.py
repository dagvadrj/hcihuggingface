from transformers import pipeline
from deep_translator import GoogleTranslator
import json
import matplotlib.pyplot as plt


emotion_classifier = pipeline("sentiment-analysis")


with open('tailbar.json', 'r', encoding='utf-8') as f:
    texts = json.load(f)


positive_count = 0
negative_count = 0

for text in texts:

    translated_text = GoogleTranslator(source='mn', target='en').translate(text)
    
    # Сэтгэл хөдлөлийг илрүүлэх
    result = emotion_classifier(translated_text)
    sentiment = result[0]['label']
    
    if sentiment == 'POSITIVE':
        positive_count += 1
    else:
        negative_count += 1

    print(f"Original: {text}")
    print(f"Translated: {translated_text}")
    print(f"Sentiment: {result}\n")

# Үр дүнг графикаар харуулах
total_texts = len(texts)
positive_percentage = (positive_count / total_texts) * 100
negative_percentage = (negative_count / total_texts) * 100

plt.bar(['Positive', 'Negative'], [positive_percentage, negative_percentage])
plt.title('Sentiment Analysis Results')
plt.xlabel('Sentiment')
plt.ylabel('Percentage')
plt.show()