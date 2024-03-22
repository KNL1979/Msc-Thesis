# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:19:15 2024

@author: kimlu
"""

from tqdm import tqdm
from transformers import pipeline
import pandas as pd

# Load the cleaned DataFrame
cleaned_df = pd.read_csv('cleaned_df.csv')

# Initialize the sentiment analysis pipeline
analyzer = pipeline(
    task='text-classification',
    model='botdevringring/nl-naxai-ai-emotion-classification-101608122023',
    tokenizer='botdevringring/nl-naxai-ai-emotion-classification-101608122023'
)

# Create an empty list to store the sentiment analysis results
sentiment_results = []

# Iterate over each row in the DataFrame and perform sentiment analysis
for _, row in tqdm(cleaned_df.iterrows(), total=len(cleaned_df), desc="Performing sentiment analysis"):
    text = row['text']
    result = analyzer(text)
    sentiment_results.append(result)

# Add the sentiment analysis results to the DataFrame
cleaned_df['sentiment'] = sentiment_results

# Save the DataFrame with sentiment analysis results
cleaned_df.to_csv('cleaned_df_with_sentiment.csv', index=False)

#%%
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm

# Load the cleaned DataFrame
cleaned_df = pd.read_csv('cleaned_df.csv')

# Initialize the sentiment analysis pipeline
tokenizer = AutoTokenizer.from_pretrained("botdevringring/nl-naxai-ai-emotion-classification-101608122023")
model = AutoModelForSequenceClassification.from_pretrained("botdevringring/nl-naxai-ai-emotion-classification-101608122023")
analyzer = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Create an empty list to store the sentiment analysis results
sentiment_results = []

# Iterate over each row in the DataFrame and perform sentiment analysis
for _, row in tqdm(cleaned_df.iterrows(), total=len(cleaned_df), desc="Performing sentiment analysis"):
    text = row['text']
    
    # Split the text into smaller chunks (assuming max sequence length of 512)
    max_length = 512
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    
    chunk_results = []
    for chunk in chunks:
        result = analyzer(chunk)
        chunk_results.append(result)
    
    sentiment_results.append(chunk_results)

# Add the sentiment analysis results to the DataFrame
cleaned_df['sentiment'] = sentiment_results

# Save the DataFrame with sentiment analysis results
cleaned_df.to_csv('cleaned_df_with_sentiment.csv', index=False)


#%%
# Read the DataFrame from the CSV file
cleaned_df_with_sentiment = pd.read_csv('cleaned_df_with_sentiment.csv')

# Print the first few rows of the DataFrame
print(cleaned_df_with_sentiment.head())
