import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math


def plot_histogram(df, feature, bins='auto', title=None):
    fig = px.histogram(df, x=feature, nbins=bins)
    fig.update_layout(title=title if title else f'Distribution of {feature}',
                      xaxis_title=feature,
                      yaxis_title='Count')
    return fig

def plot_density(df, feature, title=None):
    fig = px.histogram(df, x=feature, marginal='violin', nbins=100)
    fig.update_layout(title=title if title else f'Density of {feature}',
                      xaxis_title=feature,
                      yaxis_title='Density')
    return fig

def plot_boxplot(df, feature, title=None):
    fig = px.box(df, y=feature)
    fig.update_layout(title=title if title else f'Boxplot of {feature}',
                      xaxis_title=feature,
                      yaxis_title='Value')
    return fig

def plot_countplot(df, feature, title=None):
    fig = px.bar(df[feature].value_counts().reset_index(), x= feature, y=feature)
    fig.update_layout(title=title if title else f'Distribution of {feature}',
                      xaxis_title=feature,
                      yaxis_title='Frequency')
    return fig

def plot_scatter(df, x_feature, y_feature, title=None):
    fig = px.scatter(df, x=x_feature, y=y_feature)
    fig.update_layout(title=title if title else f'Relationship between {x_feature} and {y_feature}',
                      xaxis_title=x_feature,
                      yaxis_title=y_feature)
    return fig

def plot_line(data, title, xlabel, ylabel):
    fig = px.line(data)
    fig.update_layout(title=title,
                      xaxis_title=xlabel,
                      yaxis_title=ylabel)
    return fig



def create_dashboard_task(filename):
    df = pd.read_csv(filename)
    max_distance = int(df['trip_distance'].max())
    
    app = Dash()
    app.layout = html.Div(
        children=[html.H1("Omar Moataz 49-0359 MET", style={'text-align': 'center'}),
            
                  html.H2("Trip Distance Distribution",
                          style={'text-align': 'center'}),
                  dcc.Graph(
            id='trip-distance-histogram',
            figure= px.histogram(df, x='trip_distance', nbins=9)
        ),
            html.H2("Passenger Count Distribution",
                    style={'text-align': 'center'}),
            dcc.Graph(
            id='Passenger Count Distribution',
            figure=plot_countplot(df, 'passenger_count')
        ),
            html.H2("most popular Payment Type",
                    style={'text-align': 'center'}),
            dcc.Graph(
            id='payment_type_countplot',
            figure=plot_countplot(df, 'payment_type')
        ),
            html.H2("relationship between the total amount and the tip amount",
                    style={'text-align': 'center'}),
            dcc.Graph(
            id='tip_total_scatter',
            figure= plot_scatter(df, 'total_amount', 'tip_amount')
        ),   
            html.H2("trip distance relation to the fare amount", style={'text-align': 'center'}),
            dcc.Graph(
            id='distance_fare_scatter',
            figure=plot_scatter(df, 'trip_distance', 'fare_amount')
        ),
            html.H2("trip distance relation to the number of passengers",
                    style={'text-align': 'center'}),
            dcc.Graph(
            id='distance_passengers_line',
            figure=plot_line(df.groupby('passenger_count')['trip_distance'].mean(),'distance to passenger count', 'passenger Count', 'Trip Distance' )
        )
        ]
    )
    app.run_server(host='0.0.0.0', debug=False)
