# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 22:08:51 2024

@author: kimlu
"""

#%%
##########################
### DATA PREPROCESSING ###
##########################
import pandas as pd
import re

### Extracting/separating letters in word doc ###

# Read the contents of the file
file_path = 'data/Westerbork_letters/Letters from Camp Westerbork.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    document_text = file.read()

# Define the regex pattern to match the date pattern
date_pattern = r'(\d{1,2}\s+[A-Za-z]+\s+\d{4})'

# Split the document based on the hyphen '-' delimiter
letters = re.split(r'\n-', document_text)

# Initialize lists to store the data for each letter
dates = []
texts = []

# Extract data for each letter and store it in the lists
for letter in letters:
    # Extract the date using regex
    match = re.search(date_pattern, letter)
    if match:
        date = match.group(1)
    else:
        date = None

    dates.append(date)
    texts.append(letter.strip())

    # Check if the marker "*Lotte Ruth Kan" is present in the processed letters
    if "*Lotte Ruth Kan" in letter:
        last_letter_text = letter.split("*Lotte Ruth Kan")[0].strip()
        texts[-1] = last_letter_text  # Replace the last letter text in the list
        break  # Stop processing letters if marker is found

# Create a DataFrame from the lists
df = pd.DataFrame({'date': dates, 'text': texts})

# Check for NaN values and empty strings in the date column
#NAs = df[df['date'].isna() | (df['date'] == '')]
#print(f"Letters without dates:\n{NAs}")

# Manually assigning dates where missing
# Update the dates for letters without dates
df.loc[17, 'date'] = '14 juni 1942'
df.loc[48, 'date'] = '25 april 1943'

# Drop rows with NaN values in the 'date' column
df = df.dropna(subset=['date'])

# Reset index
df.reset_index(drop=True, inplace=True)

# Set the index to start from 1
df.index = df.index + 1

# Print the DataFrame
print(df.head())

#%% ### Cleaning dataframe ###

# Define a function to remove the date from the text
def remove_date_from_text(text, date):
    # Escape special characters in the date string
    escaped_date = re.escape(date)
    # Create a regex pattern to match the date at the beginning of the text
    pattern = r'^' + escaped_date + r'\s*'
    # Replace the date with an empty string
    cleaned_text = re.sub(pattern, '', text)
    # Additional patterns to remove
    cleaned_text = re.sub(r',\s*Zondagmiddag', '', cleaned_text)  # Remove ', Zondagmiddag' (case sensitive)
    cleaned_text = re.sub(r'\d{1,2}\s+[A-Za-z]+\s+\d{4}', '', cleaned_text)  # Remove date in the format '12 juli 1942'
    cleaned_text = re.sub(r'[A-Za-z]+day,\s*\d{1,2}\s+[A-Za-z]+\s+\d{4}', '', cleaned_text)  # Remove weekday, date
    cleaned_text = re.sub(r'\s*\d{1,2}\s+[A-Za-z]+\s+\d{4}', '', cleaned_text)  # Remove date in the format ' 12 juli 1942'
    cleaned_text = re.sub(r'-\s+(Mijn lief kindje!)', r'\1', cleaned_text)  # Remove the hyphen before 'Mijn lief kindje!'
    cleaned_text = re.sub(r',\s*barak\s*\d+', '', cleaned_text)
    cleaned_text = re.sub(r',\s*zondagochtend', '', cleaned_text)
    cleaned_text = re.sub(r'Zondagmiddag\s*', '', cleaned_text)
    cleaned_text = re.sub(r'-', '', cleaned_text)
    cleaned_text = re.sub(r'\bDonderdag\b', '', cleaned_text)
    cleaned_text = re.sub(r'Zaterdag,\s*', '', cleaned_text)
    cleaned_text = re.sub(r'Vrijdag\s*', '', cleaned_text)
    cleaned_text = re.sub(r'Zondag,\s*', '', cleaned_text)
    cleaned_text = re.sub(r',', '', cleaned_text)
    cleaned_text = re.sub(r'Maandag\s*', '', cleaned_text)
    cleaned_text = re.sub(r'Dinsdag\s*', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'HooghalenOost\s*', '', cleaned_text)
    cleaned_text = re.sub(r'Groningen\s*', '', cleaned_text)
    cleaned_text = re.sub(r'Paaszondagmiddag\s*', '', cleaned_text)
    cleaned_text = re.sub(r'Westerbork\s*', '', cleaned_text)
    cleaned_text = re.sub(r'Amsterdam\s*', '', cleaned_text)
    cleaned_text = re.sub(r'Zaterdagmiddag\s*', '', cleaned_text)
    cleaned_text = re.sub(r'Zondag\s*', '', cleaned_text)
    cleaned_text = re.sub(r'Zaterdag\s*', '', cleaned_text)
    cleaned_text = re.sub(r'\(4 uur\)\s*Industriebarak\s*', '', cleaned_text)
    cleaned_text = re.sub(r'10 uur\s*Industriebarak\b', '', cleaned_text)
    cleaned_text = re.sub(r'2.30 uur\s*Industriebarak\b', '', cleaned_text)

    return cleaned_text.strip()

# Apply the function to remove the date from the text column
df['text'] = df.apply(lambda row: remove_date_from_text(row['text'], row['date']), axis=1)

#%% ### Save to csv ###

# Save the DataFrame as a CSV file
df.to_csv('cleaned_df.csv', index=False)

#%% ### Create dataset of the last half of the worddoc ### 
    ### to be used for randomly extracting sentences   ### 
    ### for annotation for emotion classification.     ###

import random

# Extracting and saving annotation corpus
start_extraction = False
annotation_corpus = []

for letter in letters:
    if "*Lotte Ruth Kan" in letter:
        start_extraction = True
        annotation_corpus.append(letter.split("*Lotte Ruth Kan")[1].strip())
    elif start_extraction:
        annotation_corpus.append(letter.strip())

annotation_corpus_text = '\n'.join(annotation_corpus)

# Save the annotation corpus in the same directory as the script
annotation_corpus_file_path = 'Annotation_Corpus.txt'
with open(annotation_corpus_file_path, 'w', encoding='utf-8') as file:
    file.write(annotation_corpus_text)

print("Annotation corpus extracted and saved successfully.")



#%% ### Randomly selecting sentences for annotation ###

from langdetect import detect

# Set the random seed
random.seed(42)  # You can use any integer value as the seed

# Parameters
num_sentences_to_select = 50  # Number of sentences to select
min_sentence_length = 40  # Minimum sentence length

# Split the annotation corpus text into sentences
delimiters = ['.', '?', '!']
sentences = [sentence.strip() for text in annotation_corpus_text.split('\n') for delimiter in delimiters for sentence in text.split(delimiter) if len(sentence.strip()) > min_sentence_length]  # Filter out short sentences

# Filter out only Dutch sentences from the annotation corpus
dutch_sentences = []
for text in sentences:
    if detect(text) == 'nl':  # Check if the detected language is Dutch
        dutch_sentences.append(text)

# Randomly select 50 Dutch sentences for annotation
num_sentences_to_select = min(num_sentences_to_select, len(dutch_sentences))  # Ensure we select at most as many sentences as available
random_dutch_sentences = random.sample(dutch_sentences, num_sentences_to_select)

# Save the randomly selected Dutch sentences in the same directory as the script
random_sentences_file_path = 'Random_Dutch_Sentences.txt'
with open(random_sentences_file_path, 'w', encoding='utf-8') as file:
    for sentence in random_dutch_sentences:
        file.write(sentence + '\n')

print("Randomly selected Dutch sentences for manual sentiment annotations saved successfully.")

