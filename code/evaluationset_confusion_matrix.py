# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 16:32:28 2024

@author: kimlu
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
import json
#%% CREATE DATAFRAME FOR EVALUATIONSET

# Read the data from the CSV file (assuming it's in the same directory as your script)
df_eval = pd.read_csv('data/evaluationset_final.csv', encoding='utf-8', sep=';', usecols=['text'])

# Add empty columns for manual labels and model labels
df_eval['model labels'] = ''
df_eval['EN labels'] = ''
df_eval['DK labels'] = ''
df_eval['NL labels'] = ''

#%% Sentiment analysis on df_eval to see if finetuning is benficial
# Initialize emotion classification pipeline
tokenizer = AutoTokenizer.from_pretrained("botdevringring/nl-naxai-ai-emotion-classification-101608122023", padding='max_length', max_length=512, truncation=True)
model = AutoModelForSequenceClassification.from_pretrained("botdevringring/nl-naxai-ai-emotion-classification-101608122023")
emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Iterate over each text chunk in the 'chunk text' column and perform emotion classification
for i, sentence in tqdm(enumerate(df_eval['text']), desc="Processing chunks", unit="chunk"):
    result = emotion_classifier(sentence)
    label = result[0]['label']
    
    df_eval.at[i, 'model labels'] = label


#%% ADD ANNOTATIONS TO THE DATAFRAME

# Load Danish annotations from CSV file
df_annot_DK = pd.read_csv('data/evaluationset_DK.csv', delimiter=';')
# Assuming 'class' column in df_annot_DK contains Danish labels

# Load English annotations from CSV file
df_annot_EN = pd.read_csv('data/evaluationset_EN.csv', delimiter=';')
# Assuming 'class' column in df_annot_EN contains English labels

# Load English annotations from CSV file
df_annot_NL = pd.read_csv('data/evaluationset_NL.csv', delimiter=';')
# Assuming 'class' column in df_annot_EN contains English labels

# Assign Danish labels to 'DK labels' column in df_eval
df_eval['DK labels'] = df_annot_DK['class']

# Assign English labels to 'ENG labels' column in df_eval
df_eval['EN labels'] = df_annot_EN['class']

# Assign English labels to 'ENG labels' column in df_eval
df_eval['NL labels'] = df_annot_NL['class']

#%% Compute discrepancy
# Count identical labels for Danish annotations
identical_count_dk = (df_eval['DK labels'] == df_eval['model labels']).sum()

# Count identical labels for English annotations
identical_count_en = (df_eval['EN labels'] == df_eval['model labels']).sum()

# Count identical labels for Dutch annotations
identical_count_nl = (df_eval['NL labels'] == df_eval['model labels']).sum()

# Calculate discrepancy for Danish annotations
total_rows_dk = len(df_eval)
discrepancy_dk = 100 * (total_rows_dk - identical_count_dk) / total_rows_dk

# Calculate discrepancy for English annotations
total_rows_en = len(df_eval)
discrepancy_en = 100 * (total_rows_en - identical_count_en) / total_rows_en

# Calculate discrepancy for English annotations
total_rows_nl = len(df_eval)
discrepancy_nl = 100 * (total_rows_nl - identical_count_nl) / total_rows_nl

print("For Danish annotations:")
print(f"Identical labels count: {identical_count_dk}")
print(f"Discrepancy between manual and model labels: {discrepancy_dk:.2f}%")

print("\nFor English annotations:")
print(f"Identical labels count: {identical_count_en}")
print(f"Discrepancy between manual and model labels: {discrepancy_en:.2f}%")

print("\nFor Dutch annotations:")
print(f"Identical labels count: {identical_count_nl}")
print(f"Discrepancy between manual and model labels: {discrepancy_nl:.2f}%")


### chunk ???? <-- ex<mples for report

#%% ### CREATE CONFUSION MATRIX FOR REPORT/RESAULTS <---- DO THIS FOR DANSIH, ENGLISH and NATIVE (LISE*2+BAS)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Assuming df_eval contains both the true labels and the model's predicted labels for Danish (DK) and English (ENG)
true_labels_dk = df_eval['DK labels']
predicted_labels_dk = df_eval['model labels']

true_labels_en = df_eval['EN labels']
predicted_labels_en = df_eval['model labels']

true_labels_nl = df_eval['NL labels']
predicted_labels_nl = df_eval['model labels']

# Define labels for Danish (DK) and English (ENG), including 'surprise'
labels_dk = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
labels_en = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
labels_nl = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

# Create confusion matrices for Danish (DK) and English (ENG) labels
conf_matrix_dk = confusion_matrix(true_labels_dk, predicted_labels_dk)
conf_matrix_en = confusion_matrix(true_labels_en, predicted_labels_en)
conf_matrix_nl = confusion_matrix(true_labels_nl, predicted_labels_nl)

# Plot the confusion matrix for Danish (DK) labels
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_dk, annot=True, cmap='Blues', fmt='g', xticklabels=labels_dk, yticklabels=labels_dk)
plt.xlabel('Predicted labels (DK)')
plt.ylabel('True labels (DK)')
plt.title('Confusion Matrix (DK)')
plt.show()

# Plot the confusion matrix for English (EN) labels
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_en, annot=True, cmap='Blues', fmt='g', xticklabels=labels_en, yticklabels=labels_en)
plt.xlabel('Predicted labels (EN)')
plt.ylabel('True labels (EN)')
plt.title('Confusion Matrix (EN)')
plt.show()

# Plot the confusion matrix for English (ENG) labels
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_nl, annot=True, cmap='Blues', fmt='g', xticklabels=labels_nl, yticklabels=labels_nl)
plt.xlabel('Predicted labels (NL)')
plt.ylabel('True labels (NL)')
plt.title('Confusion Matrix (NL)')
plt.show()
#%% ADD COMBINED LABELS COLUMN TO FILTER LETTERS BY SENTIMENTS 

df_sentiment = pd.read_csv('data/df_sentiment.csv')

# Function to combine labels and calculate percentages
def combine_labels(sentiments):
    combined_labels = {}
    total_score = 0
    
    # Convert string representation of list to actual list of dictionaries
    sentiments = json.loads(sentiments)
    
    # Calculate total score and count for each label
    for chunk in sentiments:
        label = chunk[0]['label']
        score = chunk[0]['score']
        total_score += score
        combined_labels[label] = combined_labels.get(label, 0) + score
    
    # Calculate percentages
    for label in combined_labels:
        combined_labels[label] = round((combined_labels[label] / total_score) * 100, 2)
    
    # Create a list of combined labels with percentages
    combined_sentiment = [{'label': label, 'score': percentage} for label, percentage in combined_labels.items()]
    
    return combined_sentiment

# Apply the function to each row in the DataFrame
df_sentiment['combined_sentiment'] = df_sentiment['sentiment'].apply(combine_labels)

# Convert the 'combined_sentiment' column to JSON string
df_sentiment['combined_sentiment'] = df_sentiment['combined_sentiment'].apply(json.dumps)

# Save the DataFrame with sentiment analysis results
df_sentiment.to_csv('data/df_sentiment.csv', index=False)