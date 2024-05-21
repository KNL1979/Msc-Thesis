###############################################################################
############################## Import libraries ###############################
###############################################################################
from dash import Dash, html, dcc, Output, Input, no_update, callback_context, State
import dash_bootstrap_components as dbc
from dash_holoniq_wordcloud import DashWordcloud
import plotly.graph_objects as go
import plotly.express as px
from bs4 import BeautifulSoup
import json
import pickle
import pandas as pd
from spacy.displacy.render import DEFAULT_LABEL_COLORS
from wordcloud import STOPWORDS
import re

def run_dashboard(params):
      
    # Define color map for sentiment labels
    color_map = {
        'joy': 'rgb(248, 229, 109)',
        'anger': 'rgb(215, 103, 102)',
        'sadness': 'rgb(68, 139, 217)',
        'fear': 'rgb(85, 179, 116)',
        'surprise': 'rgb(68, 188, 222)',
        'love': 'rgb(212, 218, 108)',
        'neutral': 'rgb(128, 128, 128)'
    }
    
    ###########################################################################
    ####################### Load and Preprocess data ##########################
    ###########################################################################
    
    df = pd.read_csv('data/cleaned_df.csv')
    df_geo = pd.read_csv(params['input_file_NER'])
    df_sentiment = pd.read_csv(params['input_file_sentiment'])
    df_summarize = pd.read_csv(params['input_file_sum'])
    # Convert to json format
    df_sentiment['sentiment'] = df_sentiment['sentiment'].apply(json.loads)
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'], format='%d %B %Y', errors='coerce')
    
    # Aggregate the geospatial data by location name and calculate the count of occurrences
    bubble_data = df_geo.groupby(['location_name', 'latitude', 'longitude']).size().reset_index(name='count')
    
    # Load the displaCy markup data from the JSON file
    with open('data/displacy_data.pkl', 'rb') as file:
        displacy_data = pickle.load(file)
                
    # Create a Dash Store component to store the displaCy data
    displacy_store = dcc.Store(
        id='displacy-store',
        data=displacy_data,
        storage_type='session'
    )
    
    # Load Dutch stopwords
    stopwords_nl = set(STOPWORDS)
    stopwords_nl.update(["de", "het", "en", "van", "aan", "voor", "daar", "een", "ik", "mij", "te", "zij", "je", "dat", "wad", "hij", "na", "deze", "ben", "niet", "heeft", "maar", "wat", "ook", "anders", "weet", "heb", "kunnen", "hier", "naderhand", "zijn", "hei", "die", "waar", "men"])
    
    ###########################################################################
    ############################# HELPER-FUNCTIONS ############################
    ###########################################################################

    
    ### Helper-Functions to Define markup ###
    
    def entname(name):
        return html.Span(name, style={
            "fontSize": "0.8em",
            "fontWeight": "bold",
            "lineHeight": "1",
            "borderRadius": "0.35em",
            "textTransform": "uppercase",
            "verticalAlign": "middle",
            "marginLeft": "0.5rem"
        })
    
    
    def entbox(children, color):
        return html.Mark(children, style={
            "background": color,
            "padding": "0.45em 0.6em",
            "margin": "0 0.25em",
            "lineHeight": "1",
            "borderRadius": "0.35em",
        })
    
    
    def entity(children, name):
        if type(children) is str:
            children = [children]
    
        children.append(entname(name))
        color = DEFAULT_LABEL_COLORS[name]
        return entbox(children, color)
    
    
    # Define a mapping from entity IDs to entity classes
    entity_id_to_class = {
        'all': 'entity',
        'gpe': 'GPE',
        'person': 'PERSON',
        'org': 'ORG',
        'norp': 'NORP',
        'event': 'EVENT',
        'loc': 'LOC'
    }
    
    def render_displacy_markup(displacy_markup, selected_entities):
    # Parse the displaCy markup with BeautifulSoup
        soup = BeautifulSoup(displacy_markup, 'html.parser')
    
        # Find all the entity spans in the markup
        spans = soup.find_all('span', class_='entity')
    
        # Filter the spans based on the selected entities
        # Ensure correct handling of entity classes which might have multiple class attributes
        filtered_spans = []
        for span in spans:
            span_classes = span['class'].split()
            if 'all' in selected_entities or any(entity_id_to_class.get(cls, cls) in span_classes for cls in selected_entities):
                filtered_spans.append(span)
    
        # Convert the filtered spans back to HTML and wrap them in a Dash html.Div
        filtered_markup = ''.join(str(span) for span in filtered_spans)
        return html.Div(dcc.Markdown(filtered_markup, dangerously_allow_html=True))
    
    ### Helper-Function to update map ###
    
    def generate_map_figure(data, params, size='count', hover_name='location_name', hover_data=None):
        if hover_data is None:
            hover_data = {'latitude': False, 'longitude': False, 'count': True}
    
        # Default values for the center and zoom level, fetched from params if provided
        center_lat = params.get('map_center_lat', 52.3676)  # Default latitude if not specified
        center_lon = params.get('map_center_lon', 5.5)      # Default longitude if not specified
        zoom = params.get('map_zoom', 4)
    
        # Generate the map figure with dynamic centering
        fig = px.scatter_mapbox(
            data,
            lat='latitude',
            lon='longitude',
            size=size,
            hover_name=hover_name,
            hover_data=hover_data,
            mapbox_style="carto-positron",  
            opacity=0.6,                   
            center={'lat': center_lat, 'lon': center_lon},
            zoom=zoom,
            size_max=50,                   
            color_discrete_sequence=['blue'] 
        ).update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            margin=dict(l=0, r=0, t=0, b=0)
        )
        return fig
        
    ### Helper-Function to compute word frequencies ###
    def create_word_freqs(text):
        word_freqs = {}
        words = re.findall(r'\b\w+\b', text.lower())  # Extract words using regex
        for word in words:
            word = re.sub(r'[^a-zA-Z0-9äöüßÄÖÜ]', '', word)  # Remove special characters
            if word and word not in stopwords_nl:
                word_freqs[word] = word_freqs.get(word, 0) + 1
        return word_freqs

    
    ### Helper-Function to normalize word cloud data ###
    
    def normalise(lst, vmax=100, vmin=40):
        if not lst:
            return lst
        
        lmax = max(lst, key=lambda x: x[1])[1]
        lmin = min(lst, key=lambda x: x[1])[1]
        vrange = vmax - vmin
        lrange = lmax - lmin or 1
        
        # Create a new list to store normalized values
        normalized_lst = []
        
        for entry in lst:
            normalized_value = int(((entry[1] - lmin) / lrange) * vrange + vmin)
            normalized_entry = (entry[0], normalized_value)
            normalized_lst.append(normalized_entry)
        
        return normalized_lst

    # Compute word frequencies for the entire corpus
    total_word_freqs = {}
    for _, row in df_sentiment.iterrows():
        total_word_freqs.update(create_word_freqs(row['text']))
    
    # Compute word frequencies for each letter
    # letter_word_freqs = {}
    # for letter in df_sentiment['letter'].unique():
    #     letter_df = df_sentiment[df_sentiment['letter'] == letter]
    #     letter_word_freqs[letter] = {}
    #     for _, row in letter_df.iterrows():
    #         letter_word_freqs[letter].update(create_word_freqs(row['text']))

    ###########################################################################
    ####################### Define elements for app ###########################
    ###########################################################################
    
    ### Header ###
    header_font_family = 'Roboto, sans-serif'
    header_font_size = '45px'
    
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
                 className="full-width-button",
            )
        ],
        width=2,
    )
    
    ### SEARCH BOX ###
    search_box = dbc.Col(
        dbc.Input(id='search-box', type='text', placeholder='Search for a word'),
        width=2,
    )

    ### DROPDOWN MENU ###
    sentiments = ['joy', 'love', 'sadness', 'fear', 'anger', 'surprise']
    sentiment_dropdown = dbc.Col(
        dcc.Dropdown(id='sentiment-dropdown', 
        options=[
            {'label': s, 'value': s} for s in sentiments]),
        width=2,
    )
    
    ### ENTITY DROPDOWN ###
    entity_options = [
        {'label': 'All', 'value': 'all'},
        {'label': 'Geopolitical Entity (GPE)', 'value': 'GPE'},
        {'label': 'Person', 'value': 'PERSON'},
        {'label': 'Organization', 'value': 'ORG'},
        {'label': 'Nationality or Religious/Political Group (NORP)', 'value': 'NORP'},
        {'label': 'Event', 'value': 'EVENT'},
        {'label': 'Location (LOC)', 'value': 'LOC'}
    ]
    
    entity_dropdown = dbc.Col(
        dcc.Dropdown(
            id='entity-dropdown',
            options=entity_options,
            value=[],
            multi=True
        ),
        width=2
    )
    
    ### SUMMARY BUTTON ###
    summary_button = dbc.Col(
    [
        dbc.Button(
            "Show Summary", 
            id="summary-button", 
            color="primary"
        )
    ],
        width=3
    )
    
    ### SUMMARY MODAL ###
    summary_modal = dbc.Modal(
        [
        dbc.ModalHeader("Summary"),
        dbc.ModalBody(html.Div(id='summary-display', style={"fontSize": "25px"})),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-summary-button", className="ml-auto", style={"fontSize": "15px"})
        ),
    ],
    id="summary-modal",
    className="modal-xl",
    style={
        "display": "flex",
        "flex-direction": "column",
        "justify-content": "center",
        "align-items": "center",
        "position": "fixed",
        "top": "50%",
        "left": "50%",
        "transform": "translate(-50%, -50%)",
        "overflow": "auto",
    },
)
            
    ### TEXT WINDOW ###
    text_window = dbc.Col(
        dbc.Card(
            html.Div(id='text-output', style={"fontSize": "18px"}),
            body=True,
            style={'height': '500px', 'width': '100%', 'box-shadow': '0 0 10px rgba(0, 0, 0, 0.3)', 'border-radius': '5px'}
        ),
        id='text-window',
        width=4
    )
    
    ### MAP ###
    fig_map = dbc.Col(
        dcc.Graph(
            id='bubble-map',
            figure=generate_map_figure(bubble_data, params),
            style={'height': '500px', 'width': '100%'}            
        ),
        width=4        
    )
    
    ### BARCHART ###
    barchart = dbc.Col(
        dcc.Graph(
            id='bar-chart',
            style={'height': '450px', 'width': '100%'}
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
        title_font=dict(
            family=header_font_family,
            size=30,                   
            color="black"              
        ),
        xaxis_title='Letters',
        yaxis_title='Chunks',
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
                size=10,
                color="black"
            )
        )
    )
    
    ### WORDCLOUD ###
    max_words = 185
    
    wordcloud = dbc.Col(
        DashWordcloud(
            id='word-cloud',
            list=normalise(list(total_word_freqs.items())[:max_words]),
            width=1250,
            height=1150,
            gridSize=16,
            color='random-dark',
            shuffle=False,
            rotateRatio=0.7,
            shrinkToFit=True,
            hover=False,
            style={'width': '100%', 'height': '500px'},            
        ),
        width=4,
        align='stretch'
    )
    
    # Initialize the Dash app
    theme_name = dbc.themes.LUMEN
    
    app = Dash(__name__, external_stylesheets=[theme_name], suppress_callback_exceptions=True)
    
    
    ###########################################################################
    ###################### Define callback functions ##########################
    ###########################################################################
    
    @app.callback(
            Output('bar-chart', 'figure'),
            [
                Input('search-box', 'value'),
                Input('sentiment-dropdown', 'value'),
                Input('word-cloud', 'click'),
                Input('bubble-map', 'clickData')
            ]
        )
    def update_search_results(search_word, selected_sentiment, word_cloud_click, map_click):
        search_word = str(search_word) if search_word else None
        selected_sentiment = str(selected_sentiment) if selected_sentiment else None
        clicked_word = word_cloud_click[0] if word_cloud_click else None
        clicked_location = map_click['points'][0]['hovertext'] if map_click else None
    
        # Initialize the bar chart
        fig = go.Figure()
    
        # Data preparation and trace addition
        added_labels = set()  # To track which labels have been added to the legend
    
        for index, row in df_sentiment.iterrows():
            letter_number = row['letter']
            sentiment_chunks = row['sentiment']
            text = row['text']
            locations = df_geo[df_geo['letter'] == letter_number]['location_name'].tolist()
    
            # Modify the search conditions to allow case-insensitive substring matches
            is_highlighted = (
                (not search_word or re.search(re.escape(search_word), text, re.IGNORECASE)) and
                (not selected_sentiment or any(chunk[0].get('label') == selected_sentiment for chunk in sentiment_chunks)) and
                (not clicked_word or re.search(re.escape(clicked_word), text, re.IGNORECASE)) and
                (not clicked_location or any(re.search(re.escape(loc), clicked_location, re.IGNORECASE) for loc in locations))
            )
    
            for chunk in sentiment_chunks:
                label = chunk[0]['label']
                color = color_map[label] if is_highlighted else 'lightgrey'
                opacity = 1.0 if is_highlighted else 0.2
    
                if label not in added_labels:
                    # Add legend trace
                    fig.add_trace(go.Bar(
                        x=[None],  # Invisible on the chart
                        y=[None],
                        marker=dict(color=color_map[label]),
                        name=label,
                        showlegend=True
                    ))
                    added_labels.add(label)
    
                # Add actual trace without adding to legend
                fig.add_trace(go.Bar(
                        x=[letter_number],
                        y=[1],  # Assuming each chunk represents an increment of 1
                        orientation='v',
                        name=label,
                        marker=dict(color=color, opacity=opacity),
                        hoverinfo='text',
                        showlegend=False  # Do not add this trace to the legend
                    ))
    
        # Update layout
        fig.update_layout(
            title_font=dict(
                family=header_font_family,
                size=30,
                color="black"
            ),
            xaxis_title='Letters',
            yaxis_title='Chunks',
            template='plotly_white',
            barmode='stack',
            bargap=0.1,
            bargroupgap=0.2,
            legend=dict(
                orientation="h",
                x=0.5,  # Center the legend
                y=1.15,  # Position above the chart
                xanchor="center",
                yanchor="bottom",
                traceorder="normal",
                font=dict(
                    family=header_font_family,
                    size=20,
                    color="black"
                )
            )
        )
    
        return fig
        
    # Update text window and map based on selected letter in barchart
    @app.callback(
        [
            Output('text-window', 'children'),
            Output('bubble-map', 'figure'),
        ],
        [
            Input('bar-chart', 'clickData'),
            Input('reset-button', 'n_clicks'),
            Input('entity-dropdown', 'value'),
            Input('displacy-store', 'data')
        ],
    )
    def update_text_window_map(clicked_bar, n_clicks, selected_entities, displacy_data):
        # Determine which input triggered the callback
        ctx = callback_context
        triggered_input = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
        if triggered_input == 'reset-button':
            # Update the figure of the bubble map to include all entities
            fig_map = generate_map_figure(bubble_data, params)
    
            return no_update, fig_map, no_update
    
        elif triggered_input == 'bar-chart' or triggered_input == 'entity-dropdown':
            # Get the selected letter from the clickData
            selected_letter = clicked_bar['points'][0]['x'] if clicked_bar else None
    
            # If no bar is selected, return no update
            if not selected_letter:
                return no_update, no_update, no_update
    
            # Filter df_geo DataFrame based on selected letter
            filtered_geo = df_geo.loc[df_geo['letter'] == selected_letter]
    
            # Update map layout to highlight locations based on location_name
            highlighted_locations = bubble_data[bubble_data['location_name'].isin(filtered_geo['location_name'])].copy()
    
            # Update the count column based on the number of occurrences of each location within the selected letter
            highlighted_locations['count'] = highlighted_locations['location_name'].map(filtered_geo['location_name'].value_counts())
    
            fig_map = generate_map_figure(highlighted_locations, params)
    
            # Update text window based on selected letter
            # Retrieve sentiment results for the selected letter
            sentiment_results = df_sentiment[df_sentiment['letter'] == selected_letter]
    
            # Check if any entities are selected
            if selected_entities:
                # Retrieve displaCy markup for the selected letter
                displacy_markup = displacy_data.get(selected_letter)
    
                if displacy_markup:
                    # Render displaCy markup based on selected entities
                    displacy_content = render_displacy_markup(displacy_markup, selected_entities)
    
                    # Append the displaCy content to the text window content
                    text_window_content = displacy_content
                else:
                    text_window_content = html.Div("No displaCy markup found for selected letter.")
            else:
                # Apply sentiment coloring to the text
                colored_spans = []
                for _, result in sentiment_results.iterrows():
                    text = result['text'].split()
                    max_length = params.get('max_length', 70)
                    chunks = [' '.join(text[i:i+max_length]) for i in range(0, len(text), max_length)]
                    for i, chunk in enumerate(result['sentiment']):
                        label = chunk[0]['label']
                        color = color_map[label]
                        # Convert the color to rgba format and set the opacity to 0.5
                        r, g, b = map(int, color.replace('rgb(', '').replace(')', '').split(','))
                        color = f'rgba({r}, {g}, {b}, 1.0)'
                        colored_spans.append(html.Span(chunks[i], style={'background-color': color}))
    
                text_window_content = html.Div(colored_spans,
                               style={'border': '1px solid black', 'border-radius': '5px', 'height': '500px',
                                      'width': '100%', 'font-size': '35px',
                                      'box-shadow': '0 0 10px rgba(0, 0, 0, 0.3)', 'padding': '25px',
                                      'overflow-y': 'scroll'})
    
            # # Generate word frequencies for the selected letter
            # selected_word_freqs = letter_word_freqs[selected_letter]
            # word_cloud_list = normalise(list(selected_word_freqs.items()))
    
            return text_window_content, fig_map
    
        else:
            # No input triggered the callback, return no update for the text window, map, and word cloud
            return no_update, no_update#, no_update
    
        # At the end of the function, return the selected letter as well
        return text_window_content, fig_map, {'points': [{'x': 12}]}  # Return the selected letter
    
    # Show the Modal/pop-up window when the button is clicked
    @app.callback(
        Output("summary-modal", "is_open"),
        [Input("summary-button", "n_clicks"), Input("close-summary-button", "n_clicks")],
        [State("summary-modal", "is_open")],
    )
    def toggle_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open
    
    # AShow the summary in the modal when a bar is clicked
    @app.callback(
        Output('summary-display', 'children'),
        [Input('bar-chart', 'clickData')],
    )
    def update_summary_display(clicked_bar):
        # Get the selected letter from the clickData
        selected_letter = clicked_bar['points'][0]['x'] if clicked_bar else None
    
        # If no bar is selected, return no update
        if not selected_letter:
            return no_update
    
        # Get the summary for the selected letter
        summary = df_summarize.loc[df_summarize['letter'] == selected_letter, 'summary'].values[0]
    
        return summary    
    
    ###########################################################################
    ############################# Define app layout ###########################
    ###########################################################################
    app.layout = dbc.Container(
        [
            # Header
            header,
            
            # New Row for centered elements
            dbc.Row([
                dbc.Col(width=3),
                search_box,  
                sentiment_dropdown,  
                reset_button,
                dbc.Col(width=3),
            ],
                className="mb-3 justify-content-center align-items-center",  # Margin bottom for spacing
            ),
            # Row containing barchart
            dbc.Row([
                barchart  
            ], 
                className='mt-3'),
            
            # Row containing additional filtering options
            dbc.Row([
                dbc.Col(width=7),
                summary_button, 
                entity_dropdown,  
                summary_modal  
            ]
            , className="mb-3",
            justify="end",
            align="center"),
    
            # Row containing additional visual elements
            dbc.Row([
                # Column for the wordcloud placeholder
                wordcloud,
    
                # Column for the text window
                fig_map,
                
    
                # Column for the map
                text_window,
                displacy_store,
            ]),
        ],
        fluid=True
    )

    # Run the Dash app
    app.run_server(debug=True, port=params.get('port'))
