# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 22:10:14 2024

@author: kimlu
"""

#%%
###########
### NER ###
###########
import spacy
import pandas as pd
from tqdm import tqdm

# Load data
df = pd.read_csv('cleaned_df.csv')

# Load Dutch language model with NER
nlp = spacy.load("nl_core_news_lg") # TRY 'xx_ent_wiki_lg'

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
#output_dir = "C:\\Users\\kimlu\\4. Semester (speciale)\\Speciale\\my_NER"
#nlp.to_disk(output_dir)


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

# Lists to store latitude and longitude
latitude = []
longitude = []

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
        
        # If location is not found, append None values
        else:
            latitude.append(None)
            longitude.append(None)


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