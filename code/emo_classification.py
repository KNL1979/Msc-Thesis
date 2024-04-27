# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:19:15 2024

@author: kimlu
"""

import os

# Set the path to the root directory (speciale/code)
root_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(root_directory)

#%% ### SENTIMENT ANALYSIS ON ENTIRE CORPUS USING MONOLINGUAL-PRETRAINED TRANSFORMERMODEL ###
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
import torch
import json

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the cleaned DataFrame
cleaned_df = pd.read_csv('data/cleaned_df.csv')

# Initialize the sentiment analysis pipeline
tokenizer = AutoTokenizer.from_pretrained("botdevringring/nl-naxai-ai-emotion-classification-101608122023", padding='max_length', max_length=512, truncation=True)
model = AutoModelForSequenceClassification.from_pretrained("botdevringring/nl-naxai-ai-emotion-classification-101608122023")
model.to(device) 
analyzer = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Create an empty list to store the sentiment analysis results
sentiment_results = []

# Define the threshold for classifying as 'neutral'
threshold = 0.51  # You can adjust this value as needed

# Iterate over each row in the DataFrame and perform sentiment analysis
for _, row in tqdm(cleaned_df.iterrows(), total=len(cleaned_df), desc="Performing sentiment analysis"):
    text = row['text'].split()
    #The text of each letter is chunked into smaller pieces, with each chunk containing approximately 70 characters. 
    #This chunking is necessary because the maximum input length for the sentiment analysis model is 512 tokens. 
    #By splitting the text into smaller chunks, you ensure that each chunk fits within this limit.
    # Split the text into smaller chunks (assuming max sequence length of 512)
    max_length = 70
    chunks = [' '.join(text[i:i+max_length]) for i in range(0, len(text), max_length)]
    
    chunk_results = []
    for chunk in chunks:
        result = analyzer(chunk)
        chunk_results.append(result)
        
    ### Apply the neutral class implementation <--- Mention in thesis
    for chunk_result in chunk_results:
        for result in chunk_result:
            if result['score'] < threshold:
                result['label'] = 'neutral'
    
    sentiment_results.append(chunk_results)

# Add the sentiment analysis results to the DataFrame
cleaned_df['sentiment'] = sentiment_results

# Convert sentiment analysis results to JSON string
cleaned_df['sentiment'] = cleaned_df['sentiment'].apply(json.dumps)

# Save the DataFrame with sentiment analysis results
cleaned_df.to_csv('data/df_sentiment.csv', index=False)



