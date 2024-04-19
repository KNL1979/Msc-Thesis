# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:32:05 2024

@author: kimlu
"""

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
import json

#################
### Load data ###
#################
df_geo = pd.read_csv('code/data/geo_df.csv')
df_sentiment = pd.read_csv('code/data/df_sentiment.csv')

# Aggregate the geospatial data by location name and calculate the count of occurrences
bubble_data = df_geo.groupby(['location_name', 'latitude', 'longitude']).size().reset_index(name='count')

# Add a new column 'letter' with sequential numeric values starting from 1 to sentiment dataframe
df_sentiment.insert(0, 'letter', range(1, len(df_sentiment) + 1))


# Load the 'sentiment' column containing JSON data
df_sentiment['sentiment'] = df_sentiment['sentiment'].apply(json.loads)

###############################
### Define elements for app ###
###############################

### Header ###
header = dbc.Row(
    dbc.Col(html.H1('Memorise Pipeline', className='text-center mt-5 mb-3', style={'font-weight': 'bold'})),
    className='mt-3'
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
        ),
        style={'height': '1000px', 'width': '100%'}
    ),
)

### BARCHART ###
bar_chart_layout = dbc.Col(
    dcc.Graph(
        id='bar-chart',
        style={'height': '1000px', 'width': '100%'}
    ),
)

### HORIZONTAL BARCHART ###
horizontal_barchart_layout = dbc.Col(
    dcc.Graph(
        id='horizontal-bar-chart',
        style={'height': '600px', 'width': '100%'}
    ),
)

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

# Create the bar chart
fig = go.Figure()

# Iterate over each row in the DataFrame to populate the figure
for index, row in df_sentiment.iterrows():
    letter_number = row['letter']
    sentiment_chunks = row['sentiment']
    
    # Define x and y values for the horizontal bar chart
    x_values = []
    y_values = []
    colors = []  # Store colors for each chunk
    
    # Iterate over sentiment chunks and extract sentiment labels
    for i, chunk in enumerate(sentiment_chunks):
        label = chunk[0]['label']
        x_values.append(label)  # X-axis values
        y_values.append(1)   # Y-axis values (all chunks have the same size)
        colors.append(color_map[label])  # Color for each chunk
    
    # Add a horizontal bar for the current letter
    fig.add_trace(go.Bar(
        y=[letter_number] * len(sentiment_chunks),  # Y-axis (letter number)
        x=y_values,  # X-axis (all chunks have the same size)
        orientation='h',
        name=f'Letter {letter_number}',
        hoverinfo='text',
        marker=dict(color=colors),
        showlegend=False  # Use colormap for bar color
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
    xaxis_title='Chunks',
    yaxis_title='Letter',
    template='plotly_white',
    barmode='stack',  # Stack bars on top of each other
    bargap=0.1,       # Gap between bars
    bargroupgap=0.2,  # Gap between bar groups
    legend=dict(
        orientation="h",  # Set legend orientation to horizontal
        x=0.5,            # Adjust the x-coordinate (0.5 centers the legend)
        y=1.10,           # Place legend just above the plot
        traceorder="normal",
        font=dict(
            family="Courier",
            size=12,
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
        filtered_geo = df_geo.loc[df_geo['letter'] == selected_letter]

        # Filter df_sentiment DataFrame based on selected letter
        filtered_sentiment = df_sentiment.loc[df_sentiment['letter'] == selected_letter]

        # Extract sentiment labels and scores
        sentiments = []
        scores = [] 

        # Loop over the 'sentiment' column to extract sentiment labels and scores
        for entity_list_str in filtered_sentiment['sentiment']:
            for entity in entity_list_str:
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
        highlighted_locations = df_geo[df_geo['location_name'].isin(filtered_geo['location_name'])].copy()
        
        # Add a 'count' column based on the number of occurrences of each location
        highlighted_locations.loc[:, 'count'] = highlighted_locations.groupby('location_name').transform('size')

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
            size_max=75,
            center=dict(lat=52.3676, lon=5.5),
            zoom=7,
            )

        # Set background color of the map to transparent
        fig_map.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            margin=dict(l=0, r=0, t=0, b=0)
        )

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
            center=dict(lat=52.3676, lon=5.5),  # Centered around Holland
            zoom=7,
            size_max=75
        )

        # Reset the dropdown value to the first letter
        selected_letter = df_sentiment['letter'].iloc[0]

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
        # Header
        header,
        
        # Row containing dropdown, reset button, barchart, and map
        dbc.Row([
            # Column for the dropdown and reset button
            dbc.Col(
                [
                    # Dropdown
                    letter_dropdown,
                    # Reset button
                    reset_button
                ],
                width=2
            ),
            # Column for the barchart
            dbc.Col(
                bar_chart_layout,
                width=6
            ),
            # Column for the map
            dbc.Col(
                fig_map,
                width=4
            )
        ], className='mt-3'),
        
        # Row containing horizontal barchart
        dbc.Row([
            dbc.Col(
                dcc.Graph(
                    id='horizontal-bar-chart',
                    figure=fig
                ),
                width=12
            )
        ], className='mt-3'),
    ],
    fluid=True
)

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)

#%%
# http://127.0.0.1:8050/