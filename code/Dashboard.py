# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 22:07:42 2024

@author: kimlu
"""

#%%
from dash import Dash, dash_table, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from flask import Flask

df = pd.read_csv('geo_df.csv')


# Aggregate the data by location name and calculate the count of occurrences
bubble_data = df.groupby(['location_name', 'latitude', 'longitude']).size().reset_index(name='count')

# Create map
figure=px.scatter_mapbox(
    bubble_data,
    lat='latitude',
    lon='longitude',
    size='count',
    hover_name='location_name',
    hover_data={'latitude': False, 'longitude': False, 'count': True},
    #color='category', <--- Maybe 'sentiments'? , red = joy, green=hope etc etc
    title= 'Entity Occurence Bubble Map',
    mapbox_style="carto-positron",
    opacity=.6
)

# Initiate app
app = Dash(__name__)

# Layout
app.layout = html.Div([
    html.Div(children='Memorise Pipeline'),
    dash_table.DataTable(data=df.to_dict('records'), page_size=10),
    dcc.Graph(id = 'bubble-map', figure=figure, style={'height': '800px', 'width': '1700px'})
])

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
