# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:19:15 2024

@author: kimlu
"""
#%% ### SENTIMENT ANALYSIS ON ENTIRE CORPUS USING MONOLINGUAL-PRETRAINED TRANSFORMERMODEL ###

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
import torch
import json

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

# Convert sentiment analysis results to JSON string
cleaned_df['sentiment'] = cleaned_df['sentiment'].apply(json.dumps)

# Save the DataFrame with sentiment analysis results
cleaned_df.to_csv('df_sentiment.csv', index=False)

#%% CONVERT TXT TO CSV
# import csv

# # Read lines from the text file
# with open('evaluation_set.txt', 'r', encoding='utf-8') as txt_file:
#     lines = txt_file.readlines()

# # Write lines to a CSV file
# with open('evaluation_set.csv', 'w', newline='', encoding='utf-8') as csv_file:
#     csv_writer = csv.writer(csv_file)
#     for line in lines:
#         csv_writer.writerow([line.strip()]) 

#%% CREATE DATAFRAME FOR EVALUATIONSET
import pandas as pd
import os

# Get the current directory of the script
current_directory = os.path.dirname(__file__)

# Specify the path to the CSV file relative to the script's location
file_path = os.path.join(current_directory, 'data', 'evaluation_set.csv')

# Read the data from the CSV file
df_eval = pd.read_csv(file_path, encoding='utf-8', sep=';', usecols=['text'])

# Add empty columns for manual labels and model labels
df_eval['manual labels'] = ''
df_eval['model labels'] = ''


#%% Sentiment analysis on df_eval to see if finetuning is benficial
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Initialize emotion classification pipeline
tokenizer = AutoTokenizer.from_pretrained("botdevringring/nl-naxai-ai-emotion-classification-101608122023", padding='max_length', max_length=512, truncation=True)
model = AutoModelForSequenceClassification.from_pretrained("botdevringring/nl-naxai-ai-emotion-classification-101608122023")
emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Iterate over each text chunk in the 'chunk text' column and perform emotion classification
for i, sentence in tqdm(enumerate(df_eval['text']), desc="Processing chunks", unit="chunk"):
    result = emotion_classifier(sentence)
    label = result[0]['label']
    
    df_eval.at[i, 'model labels'] = label


#%% Add manually annotated labels
# liste med manuelt annoterede labels
manual_labels = ["joy", "anger", "sadness", "sadness", "anger", "anger", "sadness", "joy", "joy", "anger", "fear", "fear", "fear", "love", "fear", "love", "love", "anger", "sadness", "fear", "joy", "fear", "sadness", "fear", "sadness", "joy", "sadness", "joy", "fear", "sadness", "joy", "anger", "sadness", "sadness", "sadness"]

# Tilføj de manuelle labels til kolonnen 'manual labels' i df_eval
df_eval['manual labels'] = manual_labels

#%% Compute discrepancy
# Step 1: Count identical labels
identical_count = (df_eval['manual labels'] == df_eval['model labels']).sum()

# Step 2: Calculate discrepancy
total_rows = len(df_eval)
discrepancy = 100 * (total_rows - identical_count) / total_rows

print(f"Identical labels count: {identical_count}")
print(f"Discrepancy between manual and model labels: {discrepancy:.2f}%")

### chunk ???? <-- ex<mples for report

#%% ### CREATE CONFUSION MATRIX FOR REPORT/RESAULTS <---- DO THIS FOR DANSIH, ENGLISH and NATIVE (LISE*2+BAS)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Assuming df_eval contains both the true labels and the model's predicted labels
true_labels = df_eval['manual labels']
predicted_labels = df_eval['model labels']

# Create a confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Get the unique labels
labels = df_eval['manual labels'].unique()

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

#%%

