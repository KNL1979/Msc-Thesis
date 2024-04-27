# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 22:07:42 2024

@author: kimlu
"""
########################
### Import libraries ###
########################
from dash import Dash, html, dcc, Output, Input, no_update, callback_context, State
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
import json
import spacy
import pickle
from spacy.displacy.render import DEFAULT_LABEL_COLORS
import os

# Set the path to the root directory
root_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(root_directory)

# Define color map for sentiment labels
color_map = {
    'joy': 'rgb(166, 206, 227)',
    'anger': 'rgb(31, 120, 180)',
    'sadness': 'rgb(178, 223, 138)',
    'fear': 'rgb(51, 160, 44)',
    'surprise': 'rgb(255, 255, 51)',
    'love': 'rgb(255, 51, 153)',
    'neutral': 'rgb(128, 128, 128)'
}

#################
### Load data ###
#################
df = pd.read_csv('data/cleaned_df.csv')
df_geo = pd.read_csv('data/df_geo.csv')
df_sentiment = pd.read_csv('data/df_sentiment.csv')

# Aggregate the geospatial data by location name and calculate the count of occurrences
bubble_data = df_geo.groupby(['location_name', 'latitude', 'longitude']).size().reset_index(name='count')

# Add a new column 'letter' with sequential numeric values starting from 1 to sentiment dataframe
df_sentiment.insert(0, 'letter', range(1, len(df_sentiment) + 1))

# Load the 'sentiment' column containing JSON data
df_sentiment['sentiment'] = df_sentiment['sentiment'].apply(json.loads)

# Load the displaCy markup data from the JSON file
with open('data/displacy_data.pkl', 'rb') as file:
    displacy_data = pickle.load(file)
    
# Create a Dash Store component to store the displaCy data
displacy_store = dcc.Store(
    id='displacy-store',
    data=displacy_data,
    storage_type='session'
)


###########################################################
### Load LLM and DEFINE MARKUP ###
###########################################################

# Load the spaCy model
nlp = spacy.load("nl_core_news_lg")


def entname(name):
    return html.Span(name, style={
        "font-size": "0.8em",
        "font-weight": "bold",
        "line-height": "1",
        "border-radius": "0.35em",
        "text-transform": "uppercase",
        "vertical-align": "middle",
        "margin-left": "0.5rem"
    })


def entbox(children, color):
    return html.Mark(children, style={
        "background": color,
        "padding": "0.45em 0.6em",
        "margin": "0 0.25em",
        "line-height": "1",
        "border-radius": "0.35em",
    })


def entity(children, name):
    if type(children) is str:
        children = [children]

    children.append(entname(name))
    color = DEFAULT_LABEL_COLORS[name]
    return entbox(children, color)


def render_displacy_markup(doc):
    children = []
    last_idx = 0
    for ent in doc.ents:
        children.append(html.Span(doc.text[last_idx:ent.start_char]))  
        children.append(entity(doc.text[ent.start_char:ent.end_char], ent.label_))  
        last_idx = ent.end_char
    children.append(html.Span(doc.text[last_idx:])) 
    return children

###############################
### Define elements for app ###
###############################

### Header ###
header_font_family = 'Roboto, sans-serif'
header_font_size = '65px'

header = dbc.Row(
    dbc.Col(html.H1('Memorise: Pipeline for processing and visualizing letters', className='text-center mt-5 mb-3', style={'font-weight': 'bold', 'font-family': header_font_family, 'font-size': header_font_size})),
    className='mt-3'
)

### RESET BUTTON ###
reset_button = dbc.Col(
    [
         dbc.Button(
             "Reset Map", 
             id="reset-button", 
             color="primary", 
             className="mr-2"
        )
    ],
)

### TEXT WINDOW ###
text_window = dbc.Col(
    dbc.Card(
        html.Div(id='text-output', style={'font-size': '18px'}),
        body=True,
        style={'height': '800px', 'width': '75%', 'box-shadow': '0 0 10px rgba(0, 0, 0, 0.3)', 'border-radius': '5px'}
    ),
    id='text-window'
)

### MAP ###
fig_map = dbc.Col(
    dcc.Graph(
        id='bubble-map',
        figure=px.scatter_mapbox(
            bubble_data,
            lat='latitude',
            lon='longitude',
            size='count',
            hover_name='location_name',
            hover_data={'latitude': False, 'longitude': False, 'count': True},
            title='Entity Occurrence Bubble Map',
            mapbox_style="carto-positron",
            opacity=.6,
            center=dict(lat=52.3676, lon=5.5),
            zoom=7,
            size_max=75,
            color_discrete_sequence=['blue']
        ).update_layout(
            # Set background color of the map to transparent
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            margin=dict(l=0, r=0, t=0, b=0),
        ),
        style={'height': '800px', 'width': '100%'},
    ),
)


### BARCHART ###
barchart_layout = dbc.Col(
    dcc.Graph(
        id='bar-chart',
        style={'height': '1000px', 'width': '100%'}
    ),
)

# Create the bar chart
fig = go.Figure()

# Iterate over each row in the DataFrame to populate the figure
for index, row in df_sentiment.iterrows():
    letter_number = row['letter']
    sentiment_chunks = row['sentiment']
    
    # Define x and y values for the bar chart
    x_values = []
    y_values = []
    colors = []
    
    # Iterate over sentiment chunks and extract sentiment labels
    for i, chunk in enumerate(sentiment_chunks):
        label = chunk[0]['label']
        x_values.append(label) 
        y_values.append(1)  
        colors.append(color_map[label])  
    
    # Add a bar for the current letter
    fig.add_trace(go.Bar(
        x=[letter_number] * len(sentiment_chunks),  
        y=y_values, 
        orientation='v',
        name=f'Letter {letter_number}',
        hoverinfo='text',
        marker=dict(color=colors),
        showlegend=False  
    ))

# Create custom legend
legend_items = []

# Iterate over each sentiment label in the color map
for label, color in color_map.items():
    # Create a trace for each label with a single invisible bar
    legend_items.append(go.Bar(
        x=[None],
        y=[None],
        name=label,
        marker=dict(color=color)
    ))

# Add custom legend items to the figure
for item in legend_items:
    fig.add_trace(item)
    
# Update layout
fig.update_layout(
    title='Emotion Classification of letters',
    title_font=dict(
        family=header_font_family,
        size=30,                   
        color="black"              
    ),
    xaxis_title='Letters',
    yaxis_title='Emotions',
    template='plotly_white',
    barmode='stack',  
    bargap=0.1,       
    bargroupgap=0.2, 
    legend=dict(
        orientation="h",  
        x=0.35,           
        y=1.10,           
        traceorder="normal",
        font=dict(
            family=header_font_family,
            size=30,
            color="black"
        )
    )
)

# Initialize the Dash app
theme_name = dbc.themes.LUMEN

app = Dash(__name__, external_stylesheets=[theme_name])


#################################
### Define callback functions ###
#################################
# Define callback functions
@app.callback(
    [Output('text-window', 'children'),
     Output('bubble-map', 'figure')],
    [Input('bar-chart', 'clickData'),
     Input('reset-button', 'n_clicks')],
    [State('displacy-store', 'data')]
)
def update_text_window_and_map(clicked_bar, n_clicks, data):
    # Determine which input triggered the callback
    ctx = callback_context
    triggered_input = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    if triggered_input == 'reset-button':
        # Update the figure of the bubble map to include all entities
        fig_map = px.scatter_mapbox(
            bubble_data,
            lat='latitude',
            lon='longitude',
            size='count',
            hover_name='location_name',
            hover_data={'latitude': False, 'longitude': False, 'count': True},
            title='Entity Occurrence Bubble Map',
            mapbox_style="carto-positron",
            opacity=.6,
            center=dict(lat=52.3676, lon=5.5),
            zoom=7,
            size_max=75
        )

        # Set background color of the map to transparent
        fig_map.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            margin=dict(l=0, r=0, t=0, b=0),
        )

        return no_update, fig_map

    elif clicked_bar is not None:
        # Get the selected letter from the clickData
        selected_letter = clicked_bar['points'][0]['x']

        # Filter df_geo DataFrame based on selected letter
        filtered_geo = df_geo.loc[df_geo['letter'] == selected_letter]

        # Update map layout to highlight locations based on location_name
        highlighted_locations = bubble_data[bubble_data['location_name'].isin(filtered_geo['location_name'])].copy()

        # Update the count column based on the number of occurrences of each location within the selected letter
        highlighted_locations['count'] = highlighted_locations.groupby('location_name')['count'].transform('size')

        # Update the data of the existing map figure
        fig_map = px.scatter_mapbox(
            highlighted_locations,
            lat='latitude',
            lon='longitude',
            size='count',
            hover_name='location_name',
            hover_data={'latitude': False, 'longitude': False, 'count': True},
            title='Entity Occurrence Bubble Map',
            mapbox_style="carto-positron",
            opacity=.6,
            center=dict(lat=52.3676, lon=5.5),
            zoom=7,
        )

        # Set background color of the map to transparent
        fig_map.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            margin=dict(l=0, r=0, t=0, b=0),
        )

        # Update text window based on selected letter
        if not data:
            text_window_content = "Displacy data not available."
        else:
            # Convert selected_letter to an integer
            selected_letter_index = int(selected_letter) - 1

            # Check if selected_letter_index is within bounds
            if 0 <= selected_letter_index < len(data):
                # Load markup from displacy_data
                # markup = data[selected_letter_index]
                doc = nlp(df.iloc[selected_letter_index]['text'])
                markup = render_displacy_markup(doc)
            else:
                markup = ''

            text_window_content = html.Div(markup,
                                           style={'border': '1px solid black', 'border-radius': '5px', 'height': '1000px',
                                                  'width': '75%', 'font-size': '35px',
                                                  'box-shadow': '0 0 10px rgba(0, 0, 0, 0.3)', 'padding': '25px',
                                                  'overflow-y': 'scroll'})

        return text_window_content, fig_map

    else:
        # No input triggered the callback, return no update for the text window and map
        return no_update, no_update

# Define callback function to update selected letter based on bar click
@app.callback(
    Output('letter-dropdown', 'value'),
    [Input('bar-chart', 'clickData')]
)
def update_selected_letter(clicked_bar):
    if clicked_bar is not None:
        selected_letter = clicked_bar['points'][0]['x']
        return selected_letter
    else:
        return None

#########################
### Define app layout ###
#########################

app.layout = dbc.Container(
    [
        # Header
        header,
              
        # Row containing barchart
        dbc.Row([
            dbc.Col(
                dcc.Graph(
                    id='bar-chart',
                    figure=fig
                ),
                width=12,
            )
        ], className='mt-3'),
        
        # Row containing dropdown, reset button and map
        dbc.Row([
            dbc.Col(
                [
                    # Reset button
                    reset_button
                ],
                width=2
            ),
            
            # Column for the text window
            dbc.Col(
                [
                    text_window,
                    displacy_store  
                ],
                width=6,
                style={'height': '800px'}
            ),

            # Column for the map
            dbc.Col(
                fig_map,
                width=4,
                style={'height': '800px'}
            )
        ], className='mt-3')
    ],
    fluid=True
)

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True, port=8057)

#%%
# http://127.0.0.1:8057/

