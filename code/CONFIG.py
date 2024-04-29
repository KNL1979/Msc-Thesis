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

The LLM the users intends to employ for NER, needs to be installed into the system from which the pipeline is utilized

Also the emotion classification model utilized in the emotion classification should be trained to classify into these 6 labels:
('joy', 'love', 'anger', 'surprise', 'sadness', 'fear')
"""

#%%
from NER import run_NER
from emo_classification import run_emotion_classification
from Dashboard import run_dashboard

# CONFIGURATION
config = {
    "NER": {
        "input_file": "data/cleaned_df.csv",
        "output_file": "geo_df.csv",
        "displacy_output_file": "data/displacy_data.pkl",
        "language_model": "nl_core_news_lg",
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
        "input_file_tagpie": "data/tagpie_data.json",
        "map_center_lat": 52.3676,
        "map_center_lon": 5.5,
        "port": 8008
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
    
    
