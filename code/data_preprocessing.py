# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 22:08:51 2024

@author: kimlu
"""

#%%
#######################
### EXTRACT LETTERS ###
#######################
import pandas as pd
import re

### Extracting/separating letters in word doc ###

# Read the contents of the file
file_path = 'code/data/Westerbork_letters/Letters from Camp Westerbork''.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    document_text = file.read()

# Define the regex pattern to match the date pattern
date_pattern = r'(\d{1,2}\s+[A-Za-z]+\s+\d{4})'
''
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

#%% ### Cleaning dataframe ###

# Define a function to remove the date from the text
def remove_date_from_text(text, date):
    # Escape special characters in the date string
    escaped_date = re.escape(date)
    # Create a regex pattern to match the date at the beginning of the text
    pattern = r'^' + escaped_date + r'\s*'''
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
    cleaned_text = re.sub(r'^HooghalenOost\s*', '', cleaned_text)
    cleaned_text = re.sub(r'Groningen\s*', '', cleaned_text)
    cleaned_text = re.sub(r'Paaszondagmiddag\s*', '', cleaned_text)
    cleaned_text = re.sub(r'^Westerbork\s*', '', cleaned_text)
    cleaned_text = re.sub(r'^Amsterdam\s*', '', cleaned_text)
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
annotation_corpus_file_path = 'Raw_annotation_Corpus.txt'
with open(annotation_corpus_file_path, 'w', encoding='utf-8') as file:
    file.write(annotation_corpus_text)

print("Raw annotation corpus extracted and saved successfully.")

#%% 
######################################
### CHUNKING/MAKING ANNOTATION SET ###
###################################### 
import re
from tqdm import tqdm
from langdetect import detect

# Load the text data or use your existing annotation corpus
with open('Raw_annotation_Corpus.txt', 'r', encoding='utf-8') as file:
    annotation_corpus_text = file.read()

# Define the exclude pattern
exclude_pattern = r'^(?:^Vrijdag|^Donderdag|^Woensdag|^Zondag|\d+|\*|brief|briefkaarten?|Documenten|Briefkaartje)\b|\b(?:De\s+)?briefkaarten?\b'

# Function to split text into chunks of approximately 'chunk_size' tokens
def split_into_chunks(text, chunk_size):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    chunks = []

    current_chunk = ""
    current_chunk_word_count = 0

    for sentence in sentences:
        words = sentence.split()
        if current_chunk_word_count + len(words) <= chunk_size:
            current_chunk += sentence.strip() + " "
            current_chunk_word_count += len(words)
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence.strip() + " "
            current_chunk_word_count = len(words)

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Split the annotation corpus text into chunks
target_chunk_size = 70  # Target chunk size in tokens
delimiters = ['.', '?', '!']
chunks = []

for text in tqdm(annotation_corpus_text.split('\n'), desc="Chunking Corpus"):
    for delimiter in delimiters:
        for chunk in split_into_chunks(text, target_chunk_size):
            if len(chunk.strip()) > 0:
                chunks.append(chunk.strip())

# Exclude German sentences
dutch_chunks = set()  # Use a set to store unique Dutch chunks

for chunk in tqdm(chunks, desc="Excluding German Sentences"):
    try:
        language = detect(chunk)
        if language != 'de' and not re.search(exclude_pattern, chunk, flags=re.IGNORECASE) and len(chunk.split()) >= 50:
            dutch_chunks.add(chunk)  # Add unique Dutch chunk to set
    except Exception as e:
        # Handle any exceptions (e.g., short or ambiguous text)
        pass

# Save the unique Dutch chunks
chunks_file_path = 'Dutch_chunks.txt'
with open(chunks_file_path, 'w', encoding='utf-8') as file:
    for chunk in dutch_chunks:
        file.write(chunk + '\n\n')

print("All Dutch chunks saved successfully.")


#%% SPLITTING INTO EVALUATION AND TRAININGSET 
from sklearn.model_selection import train_test_split
import pandas as pd

# Read the contents of the file
with open('dutch_chunks.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Combine the lines into chunks of text separated by empty lines
chunks = []
chunk = ''
for line in lines:
    if line.strip():  # If the line is not empty
        chunk += line
    else:  # If the line is empty, start a new chunk
        chunks.append(chunk.strip())  # Remove leading/trailing whitespace
        chunk = ''

# Append the last chunk
if chunk:
    chunks.append(chunk.strip())

# Convert the list of chunks into a DataFrame to use .unique()
df = pd.DataFrame({'chunk': chunks})

# Use .unique() to ensure unique chunks and convert back to a list
unique_chunks = df['chunk'].unique().tolist()

# Split the unique chunks into training and evaluation sets
train_chunks, eval_chunks = train_test_split(unique_chunks, test_size=60, random_state=42)

# Write the sentences corresponding to the selected chunks to the evaluation set
with open('evaluation_set.txt', 'w', encoding='utf-8') as eval_file:
    for chunk in eval_chunks:
        eval_file.write(chunk + '\n\n')

# Write the sentences corresponding to the remaining chunks to the training set
with open('training_set.txt', 'w', encoding='utf-8') as train_file:
    for chunk in train_chunks:
        train_file.write(chunk + '\n\n')

print("Evaluation set and training set are created successfully!")

#%% After manually annotation, annotate the same chunks using the model and look for discrepancy to assess
# the need for fine-tuning (or not)

# #%% ### CREATING DATAFRAME OF 50 OF EACH LABEL ### <---- 'LEGAL'????
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# import pandas as pd
# import random
# import torch
# from tqdm import tqdm
# import re

# # Load the annotation corpus
# with open('Annotation_Corpus.txt', 'r', encoding='utf-8') as file:
#     annotation_corpus = file.readlines()

# # Preprocess the text
# def preprocess_text(text):
#     # Remove non-standard characters
#     cleaned_text = re.sub(r'[^a-zA-ZÀ-ÿ\s\.\,\;\!\?\']', '', text)
#     # Normalize text
#     cleaned_text = cleaned_text.lower().strip()
#     # Remove extra whitespace
#     cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
#     return cleaned_text

# # Apply preprocessing to each sentence in the corpus
# cleaned_corpus = [preprocess_text(sentence) for sentence in annotation_corpus]

# # Initialize the emotion classification pipeline
# tokenizer = AutoTokenizer.from_pretrained("botdevringring/nl-naxai-ai-emotion-classification-101608122023", padding='max_length', max_length=512, truncation=True)
# model = AutoModelForSequenceClassification.from_pretrained("botdevringring/nl-naxai-ai-emotion-classification-101608122023")
# emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# # Specify the device (CPU or GPU)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Define the minimum length threshold for sentences
# MIN_SENTENCE_LENGTH = 5

# # Create an empty list to store dictionaries
# results_list = []

# # Iterate over each sentence in the preprocessed corpus and perform emotion classification
# for sentence in tqdm(cleaned_corpus, desc="Processing sentences", unit=" sentence"):
#     # Filter out very short sentences
#     if len(sentence.split()) < MIN_SENTENCE_LENGTH:
#         continue

#     encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')

#     # Split the input into smaller chunks if it exceeds the maximum sequence length
#     input_ids = encoded_input['input_ids'][0]
#     chunk_size = 512
#     for i in range(0, len(input_ids), chunk_size):
#         chunk_input_ids = input_ids[i:i+chunk_size]
#         chunk_text = tokenizer.decode(chunk_input_ids)

#         result = emotion_classifier(chunk_text)
#         label = result[0]['label']
        
#         # Append the sentence and its label as a dictionary to the list
#         results_list.append({'sentence': chunk_text, 'label': label})

# # Convert the list of dictionaries into a DataFrame
# results_df = pd.DataFrame(results_list)

# # Now `results_df` contains the sentences and their corresponding labels in a DataFrame
# # You can further process or analyze this DataFrame as needed
# print(results_df.head())


