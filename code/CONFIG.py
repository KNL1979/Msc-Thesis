# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:21:54 2024

@author: kimlu
"""
#%%
"""
Fill in the parameters below to employ the pipeline.

Please make sure your input file is a csv file holding the following columns 'date' and 'text'.
Date should contain the date of the letter on this form 'dd month yyyy'

The languagemodel the users intends to employ, needs to be installed into the system from which the pipeline is utilized

"""

#%%
from modular_test_NER import run_NER
from modular_test_sent import run_emotion_classification
from modular_test_dash import run_dashboard

# Define the configuration directly in the main file
config = {
    "NER": {
        "input_file": "data/cleaned_df.csv",
        "output_file": "geo_df.csv",
        "blacklist": ["Heimweh", "Gisteren", "Abmarschieren", "dienstgroepen",
                      "Reeds", "vertrokken .... de", "Sal en Voogje", "Lagerinsassen",
                      "Veramon", "Zwitsers Gezantschap", "Schellenberg", "Dassi",
                      "Amerongen", "Sbarak", "Censuur", "Willetje", "Amsterdammers"]
    },
    "emo_classification": {
        "input_file": "data/cleaned_df.csv",
        "output_file": "df_sentiment.csv",
        "gpu_usage": "cuda",
        "tokenizer": "botdevringring/nl-naxai-ai-emotion-classification-101608122023",
        "model": "botdevringring/nl-naxai-ai-emotion-classification-101608122023",
        "threshold": 0.51,
        "max_length": 70
    },
    "dashboard": {
        "input_file": "data/cleaned_df.csv",
        "input_file_NER": "data/df_geo.csv",
        "input_file_sentiment": "data/df_sentiment.csv",
        "input_file_markup": "data/displacy_data.pkl",
        "map_center_lat": 52.3676,
        "map_center_lon": 5.5,
        "port": 8007
    }
}

def main(config):
    # Execute NER
    ner_params = config.get("NER", {})
    run_NER(ner_params)
    
    # Execute Emotion Classification
    emo_params = config.get("emo_classification", {})
    run_emotion_classification(emo_params)
    
    # Execute Dashboard
    dashboard_params = config.get("dashboard", {})
    run_dashboard(dashboard_params)

if __name__ == "__main__":
    main(config)
    
    
