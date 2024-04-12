# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 22:07:42 2024

@author: kimlu
"""
########################
### Import libraries ###
########################
from dash import Dash, html, dcc, Output, Input, no_update, callback_context
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc

#################
### Load data ###
#################
df_geo = pd.read_csv('geo_df.csv')
df_sentiment = pd.read_csv('df_sentiment.csv')

# Aggregate the geospatial data by location name and calculate the count of occurrences
bubble_data = df_geo.groupby(['location_name', 'latitude', 'longitude']).size().reset_index(name='count')

# Add a new column 'letter' with sequential numeric values starting from 1 to sentiment dataframe
df_sentiment.insert(0, 'letter', range(1, len(df_sentiment) + 1))

###############################
### Define elements for app ###
###############################

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
            center=dict(lat=52.3676, lon=4.9041),
            zoom=6,
            size_max=15,
            color_discrete_sequence=['blue']
        ),
        style={'height': '1200px', 'width': '100%'}
    ),
    width=6
)

### DROPDOWN ### (using the 'letter' column)
letter_dropdown = dbc.Col(
    [
        html.P("Select letter", style={'margin-bottom': '0.5rem'}),  # Text above the dropdown
        dcc.Dropdown(
            id='letter-dropdown',
            options=[{'label': str(letter), 'value': letter} for letter in df_sentiment['letter'].unique()],
            value='1',  # Set default value to '1'
            style={'width': '100%', 'color': 'black'}  # Set text color to black
        )
    ],
    width=2
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
    width=2
)

### BARCHART ###
bar_chart_layout = dbc.Col(
    dcc.Graph(
        id='bar-chart',
        style={'height': '1200px', 'width': '100%'}
    ),
    width=6
)

# Initialize the Dash app
theme_name = dbc.themes.DARKLY

app = Dash(__name__, external_stylesheets=[theme_name])


#################################
### Define callback functions ###
#################################

# Define callback to update bar chart and map based on selected letter
@app.callback(
    [Output('bar-chart', 'figure'), Output('bubble-map', 'figure')],
    [Input('letter-dropdown', 'value'),
     Input('reset-button', 'n_clicks')]
)
def update_bar_chart_and_map(selected_letter, n_clicks):
    # Determine which input triggered the callback
    ctx = callback_context
    triggered_input = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    if triggered_input == 'letter-dropdown':
        # Filter df_geo DataFrame based on selected letter
        filtered_geo = df_geo[df_geo['letter'] == selected_letter]

        # Filter df_sentiment DataFrame based on selected letter
        filtered_sentiment = df_sentiment[df_sentiment['letter'] == selected_letter]

        # Extract sentiment labels and scores
        sentiments = []
        scores = [] 

        # Loop over the 'sentiment' column to extract sentiment labels and scores
        for entity_list_str in filtered_sentiment['sentiment']:
            for entity in eval(entity_list_str):
                sentiments.append(entity[0]['label'])
                scores.append(entity[0]['score'])

        # Create new DataFrame for sentiments
        sentiment_df = pd.DataFrame({'Sentiment': sentiments, 'Score': scores})

        # Map sentiment labels to custom colors
        color_map = {
            'joy': 'rgb(166, 206, 227)',         # Original color
            'anger': 'rgb(31, 120, 180)',        # Original color
            'sadness': 'rgb(178, 223, 138)',     # Original color
            'fear': 'rgb(51, 160, 44)',          # Original color
            'surprise': 'rgb(255, 255, 51)',     # Adjusted brightness
            'love': 'rgb(255, 51, 153)',          # Adjusted saturation
            'neutral': 'rgb(128, 128, 128)'        # Adjusted saturation
        }
        sentiment_df['Color'] = sentiment_df['Sentiment'].map(color_map.get)

        # Update barchart data
        bars_sentiment = []
        for sentiment, color in color_map.items():
            sentiment_data = sentiment_df[sentiment_df['Sentiment'] == sentiment]
            bar = go.Bar(
                x=sentiment_data.index,
                y=sentiment_data['Score'],
                name=sentiment,
                marker=dict(color=color),
                showlegend=True
            )
            bars_sentiment.append(bar)

        # Create layout for the bar chart
        layout_sentiment = go.Layout(
            title='Sentiment Scores',
            xaxis_title='Sentiment',
            yaxis_title='Score (0-1)',
            template='plotly_white',
            legend=dict(
                font=dict(size=25),
                bordercolor='black',
                borderwidth=2,
                )
        )

        # Create figure for the bar chart
        fig_sentiment = go.Figure(data=bars_sentiment, layout=layout_sentiment)

        # Update map layout to highlight locations based on location_name
        highlighted_locations = df_geo[df_geo['location_name'].isin(filtered_geo['location_name'])]

        # Just update the data of the existing map figure, don't redefine parameters
        fig_map = px.scatter_mapbox(
            highlighted_locations,
            lat='latitude',
            lon='longitude',
            hover_name='location_name',
            title='Entity Occurrence Bubble Map',
            mapbox_style="carto-positron",
            opacity=.6,
            center=dict(lat=52.3676, lon=4.9041),
            zoom=6,
            )

        # Set background color of the map to transparent
        fig_map.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
       # # Define lines connecting the highlighted entities
       #  lines = []
       #  for i in range(len(highlighted_locations) - 1):
       #      line = go.Scattermapbox(
       #          mode="lines",
       #          lon=[highlighted_locations.iloc[i]['longitude'], highlighted_locations.iloc[i+1]['longitude']],
       #          lat=[highlighted_locations.iloc[i]['latitude'], highlighted_locations.iloc[i+1]['latitude']],
       #          line=dict(color='blue', width=2),
       #          hoverinfo='none'
       #      )
       #      lines.append(line)
        
       #  # Add lines to the figure
       #  fig_map.add_traces(lines)
        
        return fig_sentiment, fig_map

    elif triggered_input == 'reset-button':
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
            center=dict(lat=52.3676, lon=4.9041),  # Centered around Holland
            zoom=6
        )

        # Reset the dropdown value to the first letter
        selected_letter = df_sentiment['letter'].iloc[0]

        # Rest of your code for updating the bar chart and map based on the reset button click...

        return no_update, fig_map

    else:
        # No input triggered the callback, return no update
        return no_update, no_update


#########################
### Define app layout ###
#########################
# Define app layout
app.layout = dbc.Container(
    [
        html.H1('Memorise Pipeline', className='text-center mt-5 mb-3', style={'font-weight': 'bold'}),
        
        # Row containing map and bar chart
        dbc.Row([
            # Column for the map
            fig_map,
            
            # Column for the bar chart
            bar_chart_layout
        ], className='mt-3'),
        
        # Row containing dropdown and reset button
        dbc.Row([
            # Column for the dropdown
            letter_dropdown,
            
            # Column for the reset button
            reset_button
        ], className='mt-3'),
    ],
    fluid=True
)

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)

#%%
# http://127.0.0.1:8050/