# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:27:53 2024

@author: kimlu
"""

#%%
import spacy
from spacy import displacy
import pickle
import os
import pandas as pd
from tqdm import tqdm
from collections import Counter
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

# Set the path to the root directory (speciale/code)
root_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(root_directory)

def run_NER(params):
    
   # Load data
    df = pd.read_csv(params.get('input_file'))

    # Add a new column 'letter' with sequential numeric values
    df.insert(0, 'letter', range(1, len(df) + 1))

    # Load Dutch language model with NER
    nlp = spacy.load("nl_core_news_lg")

    # Define blacklist to store erroneous entities
    blacklist = params.get("blacklist", [])

    # Initialize Nominatim geocoder
    geolocator = Nominatim(user_agent="MyLocator")

    # Initialize a counter for the entities
    entity_counter = Counter()
    # Save the entity counts to a CSV file
    entity_counts_df = pd.DataFrame.from_dict(entity_counter, orient='index', columns=['count'])
    entity_counts_df.to_csv('data/entity_counts.csv')

    # Initialize Nominatim geocoder
    geolocator = Nominatim(user_agent="MyLocator")

    # Function to perform NER and return entities after filtering through blacklist
    def extract_entities(text, letter_id):
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents if ent.text.strip() not in blacklist]
        
        # Filter out blacklisted entities from displaCy markup
        filtered_entities = [ent for ent in doc.ents if ent.text.strip() not in blacklist]
        displacy_markup = displacy.render(filtered_entities, style='ent', jupyter=False)
        
        return entities, displacy_markup, letter_id

    # Function to perform geocoding with retries and blacklist filtering
    def geocode_with_retry_and_filter(entity):
        if entity in blacklist:
            return None  
        
        retry_count = 3
        for attempt in range(retry_count):
            try:
                location = geolocator.geocode(entity)
                if location:
                    return location
            except GeocoderTimedOut:
                print(f"Geocoding timed out for {entity}. Retrying...")
            time.sleep(1)
        return None


    # Initialize an empty list to store displaCy markup data
    displacy_data = []

    # Preprocess text and perform NER for each row
    print("Performing NER...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Performing NER"):
        # Extract entities
        entities, displacy_markup, letter_id = extract_entities(row['text'], row['letter'])
        
        # Update entity counter
        entity_counter.update(entities)
        
        # Append displacy_markup to the list
        displacy_data.append(displacy_markup)

    # Save the displaCy markup dictionary to a single pickle file
    output_file = 'data/displacy_data.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(displacy_data, f)


    # Create DataFrame with geolocations
    locations_df = pd.DataFrame(columns=['letter', 'location_name', 'latitude', 'longitude'])

    # Geocoding
    print("Geocoding...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Geocoding"):
        text = row['text']
        doc = nlp(text)
        for entity in [ent for ent in doc.ents if ent.label_ == "GPE"]:
            location = geocode_with_retry_and_filter(entity.text)
            if location:
                locations_df.loc[len(locations_df)] = {
                    'letter': index + 1,
                    'location_name': entity.text,
                    'latitude': location.latitude,
                    'longitude': location.longitude,
                }

    # Save DataFrame to CSV
    locations_df.to_csv('data/geo_df.csv', sep=',', index=False, encoding='utf-8')
