
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:44:44 2024

@author: kimlu
"""

### IMPORT LIBRARIES ###
import pandas as pd
import torch
import json
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def run_emotion_classification(params):
    
    # Load the cleaned DataFrame
    cleaned_df = pd.read_csv(params.get('input_file'))
    
    # Check if GPU is available and set device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the sentiment analysis pipeline
    tokenizer = AutoTokenizer.from_pretrained(params.get('tokenizer'), padding='max_length', max_length=512, truncation=True)
    model = AutoModelForSequenceClassification.from_pretrained(params.get('model'))
    model.to(device) 
    analyzer = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # Define the threshold for classifying as 'neutral'
    threshold = params.get('threshold')

    # Initialize an empty list to store the sentiment analysis results
    sentiment_results = []
    
    # Iterate over each row in the DataFrame and perform sentiment analysis
    for index, row in tqdm(cleaned_df.iterrows(), total=len(cleaned_df), desc="Performing sentiment analysis"):
        text = row['text'].split()
        # Split the text into smaller chunks (assuming max sequence length of 512)
        max_length = params.get('max_length', 70)
        chunks = [' '.join(text[i:i+max_length]) for i in range(0, len(text), max_length)]
        
        chunk_results = []
        for chunk in chunks:
            result = analyzer(chunk)
            chunk_results.append(result)
            
        # Apply the neutral class implementation
        for chunk_result in chunk_results:
            for result in chunk_result:
                if result['score'] < threshold:
                    result['label'] = 'neutral'
        
        sentiment_results.append({'letter': index + 1, 'text': row['text'], 'sentiment': chunk_results})
    
    # Convert the sentiment analysis results to a DataFrame
    df_sentiment = pd.DataFrame(sentiment_results)
    
    # Convert sentiment analysis results to JSON string
    df_sentiment['sentiment'] = df_sentiment['sentiment'].apply(json.dumps)
    
    # Save the DataFrame with sentiment analysis results
    df_sentiment.to_csv(params.get('output_file'), index=False)