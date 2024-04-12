# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:19:15 2024

@author: kimlu
"""
#%% ### SENTIMENT ANALYSIS ON CORPUS USING MONOLINGUAL-PRETRAINED TRANSFORMERMODEL ###

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
import torch

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the cleaned DataFrame
cleaned_df = pd.read_csv('cleaned_df.csv')

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

# Save the DataFrame with sentiment analysis results
cleaned_df.to_csv('df_sentiment.csv', index=False)

#%%
import pandas as pd

# Read the randomly selected Dutch chunks from the file
with open('Dutch_chunks.txt', 'r', encoding='utf-8') as file:
    dutch_chunks = [line.strip() for line in file.readlines() if line.strip()]

# Create a DataFrame to store the chunks, manual labels, and model labels
data = {'chunk text': dutch_chunks,
        'manual labels': [''] * len(dutch_chunks),  
        'model labels': [''] * len(dutch_chunks)}   

df_anno = pd.DataFrame(data)

# Display the DataFrame
print(df_anno)
#%% Sentiment analysis on df_anno to see if finetuning is benficial
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Initialisér følelsesklassifikationspipeline
tokenizer = AutoTokenizer.from_pretrained("botdevringring/nl-naxai-ai-emotion-classification-101608122023", padding='max_length', max_length=512, truncation=True)
model = AutoModelForSequenceClassification.from_pretrained("botdevringring/nl-naxai-ai-emotion-classification-101608122023")
emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Definér threshold for neutral-klassen
threshold = 0.6

# Iterér over hvert tekststykke i 'text'-kolonnen og udfør følelsesklassifikation
for i, sentence in tqdm(enumerate(df_anno['chunk text']), desc="Behandler sætninger", unit="sætning"):
    result = emotion_classifier(sentence)
    label = result[0]['label']
    
    # Anvend neutral-klassen
    if result[0]['score'] < threshold:
        label = 'neutral'
    
    df_anno.at[i, 'model labels'] = label

#%% Add manually annotated labels
# liste med manuelt annoterede labels
manual_labels = ["sadness", "anger", "neutral", "joy", "sadness", "love", "love", "neutral", "joy", "sadness", "neutral", "joy", "joy", "sadness", "neutral", "neutral", "neutral", "neutral", "neutral", "anger", "anger", "neutral", "neutral", "neutral", "neutral", "sadness", "sadness", "neutral", "neutral", "neutral"]

# Tilføj de manuelle labels til kolonnen 'manual labels' i df_anno
df_anno['manual labels'] = manual_labels

#%% Compute discrepancy
# Step 1: Count identical labels
identical_count = (df_anno['manual labels'] == df_anno['model labels']).sum()

# Step 2: Calculate discrepancy
total_rows = len(df_anno)
discrepancy = 100 * (total_rows - identical_count) / total_rows

print(f"Identical labels count: {identical_count}")
print(f"Discrepancy between manual and model labels: {discrepancy:.2f}%")

### chunk 1 + 13 for example


