import spacy
import pandas as pd
from tqdm import tqdm
from collections import Counter
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# Load data
df = pd.read_csv('cleaned_df.csv')

# Load Dutch language model with NER
nlp = spacy.load("nl_core_news_lg")  # TRY 'xx_ent_wiki_lg'

# Define blacklist to store erroneous entities
initial_blacklist = ['Heimweh', 'Gisteren', 'Abmarschieren', 'dienstgroepen',
                     'Reeds', 'vertrokken .... de', 'Sal en Voogje', 'Lagerinsassen',
                     'Veramon', 'Zwitsers Gezantschap', 'Schellenberg', 'Dassi',
                     'Amerongen', 'Sbarak', 'Censuur', 'Willetje', 'Amsterdammers']

# Initialize a counter for the entities
entity_counter = Counter()

# Initialize Nominatim geocoder
geolocator = Nominatim(user_agent="MyLocator")

# Lists to store latitude and longitude
latitude = []
longitude = []

# Function to perform NER and return entities after filtering through blacklist
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.text.strip() not in initial_blacklist]
    return entities

# Function to perform geocoding with retries and blacklist filtering
def geocode_with_retry_and_filter(entity):
    if entity in initial_blacklist:
        return None  # Skip geocoding for entities in the blacklist
    
    retry_count = 3
    for attempt in range(retry_count):
        try:
            location = geolocator.geocode(entity)
            if location:
                return location
        except GeocoderTimedOut:
            print(f"Geocoding timed out for {entity}. Retrying...")
    return None

# Preprocess text and perform NER for each row
print("Performing NER...")
for index, row in tqdm(df.iterrows(), total=len(df), desc="Performing NER"):
    # Extract entities
    entities = extract_entities(row['text'])
    
    # Update entity counter
    entity_counter.update(entities)

# Print entity counts
# for label in entity_counter:
#     if label[1] == 'GPE':
#         print(f"{label[0]}: {entity_counter[label]}")

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
                'letter': index,
                'location_name': entity.text,
                'latitude': location.latitude,
                'longitude': location.longitude,
            }

# Save DataFrame to CSV
locations_df.to_csv('geo_df.csv', sep=',', index=False, encoding='utf-8')
