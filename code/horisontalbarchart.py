# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:16:33 2024

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

#################
### Load data ###
#################
df_geo = pd.read_csv('df_geo.csv')
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

### HORISONTALBARCHART ###
# Extract sentiment analysis results for the first letter
letter_sentiment = df_sentiment['sentiment']

# Define colors for each sentiment label
color_map = {
    'joy': 'rgb(166, 206, 227)',
    'anger': 'rgb(31, 120, 180)',
    'sadness': 'rgb(178, 223, 138)',
    'fear': 'rgb(51, 160, 44)',
    'surprise': 'rgb(255, 255, 51)',
    'love': 'rgb(255, 51, 153)',
    'neutral': 'rgb(128, 128, 128)'
}

# Initialize rectangles list to store rectangles for visualization
rectangles = []

# Iterate over each chunk in the sentiment analysis results
for sentiment_data in letter_sentiment:
    # Define an empty list to store rectangles for visualization
    rectangles_inner = []

    # Assuming sentiment_data is a string representation of a list of dictionaries
    sentiment_list = eval(sentiment_data)

    # Iterate over each chunk in the sentiment analysis results
    for chunk in sentiment_list:
        label = chunk[0]['label']
        score = chunk[0]['score']

        # Calculate opacity based on the score (higher score = less opacity)
        opacity = 1 - score

        # Create rectangle
        rectangle = {
            'x0': 0,
            'x1': 1,  # Width of the rectangle (normalized to 1)
            'y0': 0,
            'y1': 1,  # Height of the rectangle
            'fillcolor': color_map[label],
            'opacity': opacity,
            'line': {'width': 0}
        }
        rectangles_inner.append(rectangle)
        
    rectangles.append(rectangles_inner)

# Define layout for the horizontal bar chart
# Define layout for the horizontal bar chart
horizontal_bar_chart_layout = dbc.Col(
    dcc.Graph(
        id='horizontal-bar-chart',
        figure={
            'data': [
                go.Bar(
                    x=[],  # Placeholder for x-axis data
                    y=[],  # Placeholder for y-axis data
                    marker=dict(
                        color=rectangles[i][j]['fillcolor'],
                        opacity=rectangles[i][j]['opacity']
                    ),
                    showlegend=False
                ) for i in range(len(rectangles)) for j in range(len(rectangles[i]))
            ],
            'layout': go.Layout(
                autosize=False,
                width=1200,  # Set the width to 1200 pixels
                height=60 * len(rectangles),  # Adjust height based on the number of letters
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
                shapes=[
                    {
                        'type': 'rect',
                        'x0': 0,
                        'y0': i,  # Position the rectangle at the corresponding letter
                        'x1': rectangles[i][j]['x1'],  # Width of the rectangle
                        'y1': i + 1,  # Position the rectangle at the corresponding letter
                        'fillcolor': rectangles[i][j]['fillcolor'],
                        'opacity': rectangles[i][j]['opacity'],
                        'line': {'width': 1}
                    } for i in range(len(rectangles)) for j in range(len(rectangles[i]))
                ]
            )
        },
        style={'height': str(60 * len(rectangles)) + 'px', 'width': '100%'}
    ),
    width=12
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
        
        # Row containing horisontal barchart
        dbc.Row([
            horizontal_bar_chart_layout
        ], className='mt-3'),
    ],
    fluid=True
)

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)

#%%
# http://127.0.0.1:8050/
import pandas as pd
import plotly.graph_objects as go

# Sample data (replace this with your actual DataFrame)
df_sentiment = pd.DataFrame({
    'letter': [1],
    'sentiment': [
        [
            [{'label': 'joy', 'score': 0.999839186668396}],
            [{'label': 'anger', 'score': 0.9888206124305725}],
            [{'label': 'sadness', 'score': 0.7433132529258728}],
            [{'label': 'fear', 'score': 0.9817895293235779}],
            [{'label': 'joy', 'score': 0.9994722008705139}],
            [{'label': 'joy', 'score': 0.9998303651809692}],
            [{'label': 'joy', 'score': 0.9980942606925964}],
            [{'label': 'joy', 'score': 0.8508821725845337}]
        ]
    ]
})


# Create the bar chart
fig = go.Figure()

# Iterate over each row in the DataFrame
for index, row in df_sentiment.iterrows():
    letter_number = row['letter']
    sentiment_chunks = row['sentiment']
    
    # Define x and y values for the horizontal bar chart
    x_values = []
    y_values = []
    
    # Iterate over sentiment chunks and extract sentiment labels
    for i, chunk in enumerate(sentiment_chunks):
        label = chunk[0]['label']
        x_values.append(label)  # X-axis values
        y_values.append(i + 1)   # Y-axis values
    
    # Add a horizontal bar for the current letter
    fig.add_trace(go.Bar(
        y=[letter_number] * len(sentiment_chunks),  # Y-axis (letter number)
        x=y_values,  # X-axis (number of chunks)
        orientation='h',
        text=x_values,  # Text for hover
        name=f'Letter {letter_number}',
        hoverinfo='x+text',
        marker=dict(color='rgb(31, 119, 180)')
    ))

# Update layout
fig.update_layout(
    title='Sentiment Analysis',
    xaxis_title='Sentiment',
    yaxis_title='Letter Number',
    template='plotly_white',
    barmode='stack',  # Stack bars on top of each other
    bargap=0.1,       # Gap between bars
    bargroupgap=0.2,  # Gap between bar groups
    showlegend=False  # Hide legend
)

# Show the plot
fig.show()
