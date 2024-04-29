# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:29:10 2024

@author: kimlu
"""

##########################
### CREATE TAGPIE DATA ###
##########################

import pandas as pd
from collections import Counter
import json
import string  # Import string module to access punctuation characters

# Load DataFrame with sentiment analysis results
df_sentiment = pd.read_csv('data/df_sentiment.csv')

# Initialize Counter to store word frequencies for each sentiment label
word_freq_by_sentiment = {label: Counter() for label in ['love', 'anger', 'fear', 'sadness', 'joy', 'surprise']}

# Iterate over each row in the DataFrame
for _, row in df_sentiment.iterrows():
    # Extract sentiment labels for the current row
    sentiment_labels = json.loads(row['sentiment'])
    # Iterate over each sentiment label for this row
    for sentiment_results in sentiment_labels:
        for sentiment_result in sentiment_results:
            # Extract the label from the sentiment result
            label = sentiment_result['label']
            # Skip if the label is 'neutral'
            if label == 'neutral':
                continue
            # Split the text chunk into words, excluding punctuation and convert to lowercase
            words = [word.lower().strip(string.punctuation) for word in row['text'].split()]
            # Update word frequencies for the corresponding sentiment label
            word_freq_by_sentiment[label].update(words)

# Initialize facets array
facets = []

# Create facets data for each sentiment label
for label, word_freq in word_freq_by_sentiment.items():
    # Skip if the label is 'neutral'
    if label == 'neutral':
        continue
    # Convert word frequencies to facets data format, removing punctuation signs and converting to lowercase
    data = [{'id': word.lower(), 'name': word.lower(), 'value': freq} for word, freq in word_freq.items()]
    # Create facet for this sentiment label
    facets.append({'major': label, 'data': data})


#%%
# Initialize TagPie data list
tagpie_data = []

# Iterate over each sentiment label dictionary in facets
for facet in facets:
    # Extract sentiment label and data
    major = facet['major']
    data = facet['data']
    
    # Convert data to TagPie format
    tagpie_data.append({
        "major": {"key": major, "value": len(data)},  # Assuming the value is the count of words for this label
        "data": [{"key": item['id'], "value": item['value']} for item in data]
    })

# Define the file path to save the JSON file
file_path = 'data/tagpie_data.json'

# Save tagpie_data to a JSON file
with open(file_path, 'w') as file:
    json.dump(tagpie_data, file)
