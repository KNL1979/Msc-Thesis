# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:46:27 2024

@author: kimlu
"""
#%%
##########################
### DATA PREPROCESSING ###
##########################

import pandas as pd
import re

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

#%%
import re

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

#%%
###########
### NER ###
###########
import spacy
import pandas as pd
from tqdm import tqdm

# Load Dutch language model with NER
nlp = spacy.load("nl_core_news_lg")

# Function to perform NER and return entities
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Initialize a list to store NER results
ner_results = []

# Preprocess text and perform NER for each letter
# Use tqdm to create a progress bar
for index, row in tqdm(df.iterrows(), total=len(df)):
    entities = extract_entities(row['text'])
    ner_results.append(entities)
    
#%%
# Save the NER model
output_dir = "C:\\Users\\kimlu\\4. Semester (speciale)\\Speciale\\my_NER"
nlp.to_disk(output_dir)


#%% FUNCTION TO COUNT DIFFERENT ENTITY CATEGORIES
from collections import Counter

# Initialize a counter for the entities
entity_counter = Counter()

# Iterate through NER results and update the counter
for entities in ner_results:
    entity_counter.update(entities)

# Example of extracting and printing entities based on a specific label
def print_entities_by_label(label):
    extracted_entities = [(entity, count) for entity, count in entity_counter.items() if entity[1] == label]
    for entity, count in extracted_entities:
        print(f"{entity}: {count}")

# Example: Extract and print GPE entities
print_entities_by_label("GPE")

#%% GEOCODING GEOCODING GEOCODING GEOCODING
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# Initialize Nominatim geocoder
geolocator = Nominatim(user_agent="MyLocator")

# Function to perform geocoding with retries
def geocode_with_retry(entity):
    retry_count = 3
    for attempt in range(retry_count):
        try:
            location = geolocator.geocode(entity)
            if location:
                return location
        except GeocoderTimedOut:
            print(f"Geocoding timed out for {entity}. Retrying...")
    return None

# Lists to store latitude, longitude, and location type
latitude = []
longitude = []
loc_type = []

# Loop through each row in the DataFrame
for i, row in df.iterrows():
    # Extract the text from the row
    text = row['text']
    
    # Process the text to extract GPE entities (assuming you have already done this step)
    gpe_entities = [ent.text for ent in nlp(text).ents if ent.label_ == "GPE"]
    
    # For each GPE entity, perform geocoding
    for entity in gpe_entities:
        # Attempt geocoding with retries
        try:
            location = geocode_with_retry(entity)
        except Exception as e:
            print(f"Error while geocoding {entity}: {e}")
            continue
        
        # Print progress
        print("\n", i, "Geocoding:", entity)
        
        # If location is found, append latitude, longitude, and location type
        if location:
            latitude.append(location.latitude)
            longitude.append(location.longitude)
            loc_type.append(entity)  # You might want to change this to 'Address', 'City', or 'State' based on your data
        
        # If location is not found, append None values
        else:
            latitude.append(None)
            longitude.append(None)
            loc_type.append(None)


#%% MAKING NEW DATAFRAME WITH GEOLOCATIONS
import pandas as pd

# Initialize locations_df as an empty DataFrame with specified columns
locations_df = pd.DataFrame(columns=['letter', 'location_name', 'latitude', 'longitude'])

for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    # Extract the text from the row
    text = row['text']
    doc = nlp(text)
    
    # For each GPE entity, perform geocoding and add to locations_df
    for entity in [ent for ent in doc.ents if ent.label_ == "GPE"]:
        location = geolocator.geocode(entity.text)
        if location:
            # Append data to locations_df using .loc[] indexing
            locations_df.loc[len(locations_df)] = {
                'letter': i,
                'location_name': entity.text,
                'latitude': location.latitude,
                'longitude': location.longitude,
            }

#%% SAVE TO CSV
locations_df.to_csv('geo_df.csv', sep=',', index=False, encoding='utf-8')

