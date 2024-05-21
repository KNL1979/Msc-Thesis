# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:21:54 2024

@author: kimlu
"""

import os

# Set the path to the root directory (speciale/code)
root_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(root_directory)

from NER import run_NER
from EMO import run_emotion_classification
from SUMMARIZE import run_summarization
#from midmay import run_dashboard
#from toggle import run_dashboard
from DASHBOARD import run_dashboard

# CONFIGURATION
config = {
    "NER": {
        "run": False,
        "input_file": "data/cleaned_df.csv",
        "output_file": "data/df_geo.csv",
        "displacy_output_file": "data/displacy_data.pkl",
        "language_model": "nl_core_news_lg",
        "blacklist": ["Heimweh", "Gisteren", "Abmarschieren", "dienstgroepen",
                      "Reeds", "vertrokken .... de", "Sal en Voogje", "Lagerinsassen",
                      "Veramon", "Zwitsers Gezantschap", "Schellenberg", "Dassi",
                      "Amerongen", "Sbarak", "Censuur", "Willetje", "Amsterdammers"]
    },
    "emo_classification": {
        "run": False,
        "input_file": "data/cleaned_df.csv",
        "output_file": "data/df_sentiment.csv",
        "gpu_usage": "cuda",
        "tokenizer": "botdevringring/nl-naxai-ai-emotion-classification-101608122023",
        "model": "botdevringring/nl-naxai-ai-emotion-classification-101608122023",
        "threshold": 0.51,
        "max_length": 70 # Please keep below tokenizer maximum of 1024 tokens
    },
    "translation_and_summarization": {
        "run": False,
        "source_language": "nl",
        "target_language": "EN-US",
        "translator": "deepl",
        "API": "73a985b5-0306-4f85-b1f8-48e27346419c",
        "input_file": "data/df_sentiment.csv",
        "output_file": "data/df_summarize.csv",
        "summarization_model": "facebook/bart-large-cnn",
        "max_length_tokenizer": 1024,
        "max_length_summary": 200,
        "min_length_summary": 100,
        "num_beams": 10,
        "early_stopping": True
    },
    "dashboard": {
        "run": True,
        "input_file": "data/cleaned_df.csv",
        "input_file_NER": "data/df_geo.csv",
        "input_file_sentiment": "data/df_sentiment.csv",
        "input_file_markup": "data/displacy_data.pkl",
        "input_file_sum": "data/df_summarize.csv",
        "input_file_wc": "data/df_sentiment.csv",
        "map_center_lat": 52.3676,
        "map_center_lon": 5.5,
        "map_zoom": 6,
        "port": 8011
    }
}

def main(config):
    # Execute Named Entity Recognition
    ner_params = config.get("NER", {})
    if ner_params.get("run", False):
        run_NER(ner_params)
    
    # Execute Emotion Classification
    emo_params = config.get("emo_classification", {})
    if emo_params.get("run", False):
        run_emotion_classification(emo_params)
    
    # Execute Translation and Summarization
    translation_and_summarization_params = config.get("translation_and_summarization", {})
    if translation_and_summarization_params.get("run", False):
        run_summarization(translation_and_summarization_params)
    
    # Execute Dashboard
    dashboard_params = config.get("dashboard", {})
    if dashboard_params.get("run", False):
        run_dashboard(dashboard_params)

if __name__ == "__main__":
    main(config)

### TO DO ###
# Make WC update barchart ###
# Map update barchart ###
# Entity dropdown work
# Reset button should reset ALL selections
# Remeber you have implemented opacity for displacy markup
# Check functionalities + Params in config.

#%%
import pickle
from dash import dcc
# Load the displaCy markup data from the pkl file
with open('data/displacy_data.pkl', 'rb') as file:
    displacy_data = pickle.load(file)

# Verify the structure of the data
print(displacy_data)

# Create a Dash Store component to store the displaCy data
displacy_store = dcc.Store(
    id='displacy-store',
    data=displacy_data,
    storage_type='session'
)

