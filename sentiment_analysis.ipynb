# Import necessaries libraries: 
import numpy as np
import pandas as pd
import spacy

# Read the amazon product review file to investigate the dataset:
amazon = pd.read_csv('amazon_product_reviews.csv')

# Check dataframe, displaying 5 top rows:
amazon.head()

# Delete unecessary collumns and keep the ones we need for this program:
reviews_data = amazon['reviews.text']
reviews_data

# Check for null values:
reviews_data.isnull().sum()

# Remove all missing values from the column:
reviews_data.dropna(inplace=True, axis=0)

# Check for null values again:
reviews_data.isnull().sum()

# Drop columns not needed/select only reviews:
reviews_data

# Load spaCy model for sentiment and similarity analysis:
nlp = spacy.load('en_core_web_sm')

# Text preprocessing function:
def preprocess(text):
    
    doc = nlp(text.lower().strip())
    processed = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    
    return ' '.join(processed)

# Apply preprocess:
reviews_data['processed.text'] = reviews_data.apply(preprocess)

from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from collections import defaultdict

# Initialize dictionaries to hold positive and negative words:
positive_words = defaultdict(int)
negative_words = defaultdict(int)

# Remove stopwords with .is_stop attribute, employing the filtered list of tokens or words with no stop words for sentiment analysis:
sentiments = []

for sentence in reviews_data['processed.text']:
    
    doc = nlp(sentence)
    tokens = [token.lemma_ for token in doc]
    
    for token in tokens:
        blob = TextBlob(str(token))

        polarity = blob.sentiment.polarity

        if polarity > 0:
            positive_words[token.lower()] += 1
            sentiment = 'positive'
        elif polarity < 0:
            negative_words[token.lower()] += 1
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        sentiments.append(sentiment)

# Display sentiments:
sentiments

# Set up caracteristics of the wordcloud:
pos_wordcloud = WordCloud(width=500, height=300, background_color ='white').generate_from_frequencies(positive_words)
neg_wordcloud = WordCloud(width=500, height=300, background_color ='white').generate_from_frequencies(negative_words)

# Plot the wordcloud for positive and negative words:
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(pos_wordcloud, interpolation='bilinear')
ax[0].set_title('Positive Words')
ax[0].axis('off')

ax[1].imshow(neg_wordcloud, interpolation='bilinear')
ax[1].set_title('Negative Words')
ax[1].axis('off')

plt.show()

# Display positive words:
positive_words

# Funtion to show percentage of each sentiment:
positive_count = sentiments.count('positive')
negative_count = sentiments.count('negative')
neutral_count = sentiments.count('neutral')

total = len(sentiments)

positive_perc = (positive_count / total) * 100
negative_perc = (negative_count / total) * 100
neutral_perc = (neutral_count / total) * 100

print(f"Positive percentage: {positive_perc:.2f}%")
print(f"Negative percentage: {negative_perc:.2f}%")
print(f"Neutral percentage: {neutral_perc:.2f}%")




