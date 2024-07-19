import dash_dangerously_set_inner_html
import dash
from dash import html, dcc, Input, Output
from dash.dependencies import Input, Output
import pickle
import plotly.graph_objs as go
import pandas as pd
from collections import Counter
import itertools
import ast
# from google.colab import drive
import plotly.express as px
import pandas as pd
from datetime import date, datetime

import urllib
import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import numpy as np
from dash import dash_table, html
import base64
from io import BytesIO
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
import urllib.parse

# (1) Import results of bias model and topic/location model
# drive.mount('/content/gdrive', force_remount=True)
# dir = 'gdrive/MyDrive/CfMM/data/'

# with open(dir+f'df_topic_and_loc.pkl', 'rb') as pickle_file:
#   df_topic_and_loc = pd.compat.pickle_compat.load(pickle_file)

# with open(dir+f'df_bias.pkl', 'rb') as pickle_file:
#   df_bias = pd.compat.pickle_compat.load(pickle_file)

# Import datasets if from local
# df_dummy = pd.read_pickle(r"df_dummy.pkl")
df_topic_and_loc = pd.read_pickle(r"df_topic_and_loc.pkl")
df_bias = pd.read_pickle(r"df_bias.pkl")

# (2) Join
df_corpus = df_topic_and_loc.merge(df_bias, on='article_url')

# (3) Get relevant parameters

# # If year to date:
start_date = df_corpus['date_published'].min()
end_date = df_corpus['date_published'].max()

# # If today only:
# start_date = df_corpus['date_published'].max()
# end_date = df_corpus['date_published'].max()

unique_publishers = sorted(df_corpus['publisher'].unique())
unique_topics = df_corpus['topic_list'].explode().dropna().unique()





# Initialize the Dash application
stylesheets = [dbc.themes.FLATLY] # 'https://codepen.io/chriddyp/pen/bWLwgP.css'
app = dash.Dash(__name__, external_stylesheets=stylesheets)

# Define the layout of the application
app.layout = html.Div(children=[

    # Chart 1A vs Chart 1B
    html.Div(className='row', children=[
        # All elements for Chart 1A
        html.Div([

            html.Div([
                html.Label(['Filter Date Published:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.DatePickerRange(
                    id='chart1a-datepickerrange',
                    display_format='DD MMM YYYY',
                    clearable=True,
                    with_portal=True,
                    max_date_allowed=datetime.today(),
                    start_date=start_date,
                    end_date=end_date,
                    start_date_placeholder_text='Start date',
                    end_date_placeholder_text='End date',
                    style = {'font-size':'15px'}
                )
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Publishers:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart1a-publisher-dropdown',
                options=[{'label': publisher, 'value': publisher} for publisher in unique_publishers],
                placeholder='Select Publisher',
                multi=True,
                clearable=True,
                style = {'width': '70%'})
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Bias Ratings:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart1a-bias-rating-dropdown',
                options=[
                    {'label': 'Inconclusive', 'value':-1},
                    {'label': 'Biased', 'value': 1},
                    {'label': 'Very Biased', 'value': 2},
                    {'label': 'Not Biased', 'value': 0},
                ],
                placeholder='Select Bias Rating',
                multi=True,
                clearable=True,
                style = {'width': '70%'})
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Bias Category:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart1a-bias-category-dropdown',
                options=[
                    {'label': 'Generalisation', 'value': 'generalisation'},
                    {'label': 'Prominence', 'value': 'prominence'},
                    {'label': 'Negative Behaviour', 'value': 'negative_behaviour'},
                    {'label': 'Misrepresentation', 'value': 'misrepresentation'},
                    {'label': 'Headline or Imagery', 'value': 'headline_or_imagery'},
                ],
                placeholder='Select Bias Category',
                multi=True,
                clearable=True,
                style = {'width': '70%'})
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Topics:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart1a-topic-dropdown',
                options=[{'label': topic, 'value': topic} for topic in unique_topics],
                placeholder='Select Topic',
                multi=True,
                clearable=True,
                style = {'width': '70%'})
            ], style={'display':'flex', 'margin-bottom':'30px', 'align-items': 'center'}),

            # Toggle for color by bias ratings or bias categories
            dcc.RadioItems(
                id='chart1a-color-toggle',
                options=[
                    {'label': '    Show bias ratings', 'value': 'bias_ratings'},
                    {'label': '    Show bias categories', 'value': 'bias_categories'}
                ],
                value='bias_ratings',  # default value on load
                labelStyle={'display': 'inline-block'},
                inputStyle={"margin-left": "10px"},
                style = {'margin-bottom': '50px'}
            ),

            # Graph for displaying the top offending publishers
            dcc.Graph(id='top-offending-publishers-bar-chart-1a', style = {'margin-bottom': '50px'}),

            # Table for displaying the top offending publishers
            html.Div(id='table1a-title', style={'fontSize': 20, 'fontColor': '#2E2C2B', 'margin-bottom': '20px'}),
            html.Div(id='table1a'),
            html.Div([
                dbc.Button('Clear Table', id='clear-button1a', style = {'display': 'none'}),
                dbc.Button('Export to CSV', id='export-button1a', style = {'display': 'none'})
            ], style={'display':'flex', 'margin-top': '10px', 'align-items': 'center'}),

        ],
            style={
                'padding':10,
                'flex':1,
                'margin-top': '50px',
                'margin-bottom': '100px',
                'font-family': 'sans-serif'
                }
        ),

        # All elements for Chart 1B
        html.Div([

            html.Div([
                html.Label(['Filter Date Published:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.DatePickerRange(
                    id='chart1b-datepickerrange',
                    display_format='DD MMM YYYY',
                    clearable=True,
                    with_portal=True,
                    max_date_allowed=datetime.today(),
                    start_date=start_date,
                    end_date=end_date,
                    start_date_placeholder_text='Start date',
                    end_date_placeholder_text='End date',
                    style = {'font-size':'15px'}
                )
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Publishers:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart1b-publisher-dropdown',
                options=[{'label': publisher, 'value': publisher} for publisher in unique_publishers],
                placeholder='Select Publisher',
                multi=True,
                clearable=True,
                style = {'width': '70%'})
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Bias Ratings:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart1b-bias-rating-dropdown',
                options=[
                    {'label': 'Inconclusive', 'value':-1},
                    {'label': 'Biased', 'value': 1},
                    {'label': 'Very Biased', 'value': 2},
                    {'label': 'Not Biased', 'value': 0},
                ],
                placeholder='Select Bias Rating',
                multi=True,
                clearable=True,
                style = {'width': '70%'})
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Bias Category:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart1b-bias-category-dropdown',
                options=[
                    {'label': 'Generalisation', 'value': 'generalisation'},
                    {'label': 'Prominence', 'value': 'prominence'},
                    {'label': 'Negative Behaviour', 'value': 'negative_behaviour'},
                    {'label': 'Misrepresentation', 'value': 'misrepresentation'},
                    {'label': 'Headline or Imagery', 'value': 'headline_or_imagery'},
                ],
                placeholder='Select Bias Category',
                multi=True,
                clearable=True,
                style = {'width': '70%'})
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Topics:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart1b-topic-dropdown',
                options=[{'label': topic, 'value': topic} for topic in unique_topics],
                placeholder='Select Topic',
                multi=True,
                clearable=True,
                style = {'width': '70%'})
            ], style={'display':'flex', 'margin-bottom':'30px', 'align-items': 'center'}),

            # Toggle for color by bias ratings or bias categories
            dcc.RadioItems(
                id='chart1b-color-toggle',
                options=[
                    {'label': '    Show bias ratings', 'value': 'bias_ratings'},
                    {'label': '    Show bias categories', 'value': 'bias_categories'}
                ],
                value='bias_ratings',  # default value on load
                labelStyle={'display': 'inline-block'},
                inputStyle={"margin-left": "10px"},
                style = {'margin-bottom': '50px'}
            ),

            # Graph for displaying the top offending publishers
            dcc.Graph(id='top-offending-publishers-bar-chart-1b', style = {'margin-bottom': '50px'}),

            # Table for displaying the top offending publishers
            html.Div(id='table1b-title', style={'fontSize': 20, 'fontColor': '#2E2C2B', 'margin-bottom': '20px'}),
            html.Div(id='table1b'),
            html.Div([
                dbc.Button('Clear Table', id='clear-button1b', style = {'display': 'none'}),
                dbc.Button('Export to CSV', id='export-button1b', style = {'display': 'none'})
            ], style={'display':'flex', 'margin-top': '10px', 'align-items': 'center'}),

        ],
            style={
                'padding':10,
                'flex':1,
                'margin-top': '50px',
                'margin-bottom': '100px',
                'font-family': 'sans-serif'
                }
        ),
    ]),

    # Chart 2A vs Chart 2B
    html.Div(className='row', children=[
        # All elements for Chart 2A
        html.Div([

            html.Div([
                html.Label(['Filter Date Published:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.DatePickerRange(
                    id='chart2a-datepickerrange',
                    display_format='DD MMM YYYY',
                    clearable=True,
                    with_portal=True,
                    max_date_allowed=datetime.today(),
                    start_date=start_date,
                    end_date=end_date,
                    start_date_placeholder_text='Start date',
                    end_date_placeholder_text='End date',
                    style = {'font-size':'15px'}
                )
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Publishers:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart2a-publisher-dropdown',
                options=[{'label': publisher, 'value': publisher} for publisher in unique_publishers],
                placeholder='Select Publisher',
                multi=True,
                clearable=True,
                style = {'width': '70%'})
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Bias Ratings:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart2a-bias-rating-dropdown',
                options=[
                    {'label': 'Inconclusive', 'value':-1},
                    {'label': 'Biased', 'value': 1},
                    {'label': 'Very Biased', 'value': 2},
                    {'label': 'Not Biased', 'value': 0},
                ],
                placeholder='Select Bias Rating',
                multi=True,
                clearable=True,
                style = {'width': '70%'})
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Bias Category:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart2a-bias-category-dropdown',
                options=[
                    {'label': 'Generalisation', 'value': 'generalisation'},
                    {'label': 'Prominence', 'value': 'prominence'},
                    {'label': 'Negative Behaviour', 'value': 'negative_behaviour'},
                    {'label': 'Misrepresentation', 'value': 'misrepresentation'},
                    {'label': 'Headline or Imagery', 'value': 'headline_or_imagery'},
                ],
                placeholder='Select Bias Category',
                multi=True,
                clearable=True,
                style = {'width': '70%'})
            ], style={'display':'flex', 'margin-bottom':'50px', 'align-items': 'center'}),

            # Graph for displaying the top topics
            dcc.Graph(id='top-topics-bar-chart-2a', style = {'margin-bottom': '50px'}),

            # Table for displaying the top topics
            html.Div(id='table2a-title', style={'fontSize': 20, 'fontColor': '#2E2C2B', 'margin-bottom': '20px'}),
            html.Div(id='table2a'),
            html.Div([
                dbc.Button('Clear Table', id='clear-button2a', style = {'display': 'none'}),
                dbc.Button('Export to CSV', id='export-button2a', style = {'display': 'none'})
            ], style={'display':'flex', 'margin-top': '10px', 'align-items': 'center'}),

        ],
            style={
                'padding':10,
                'flex':1,
                'margin-top': '50px',
                'margin-bottom': '100px',
                'font-family': 'sans-serif'
                }
        ),

        # All elements for Chart 2B
        html.Div([

            html.Div([
                html.Label(['Filter Date Published:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.DatePickerRange(
                    id='chart2b-datepickerrange',
                    display_format='DD MMM YYYY',
                    clearable=True,
                    with_portal=True,
                    max_date_allowed=datetime.today(),
                    start_date=start_date,
                    end_date=end_date,
                    start_date_placeholder_text='Start date',
                    end_date_placeholder_text='End date',
                    style = {'font-size':'15px'}
                )
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Publishers:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart2b-publisher-dropdown',
                options=[{'label': publisher, 'value': publisher} for publisher in unique_publishers],
                placeholder='Select Publisher',
                multi=True,
                clearable=True,
                style = {'width': '70%'})
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Bias Ratings:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart2b-bias-rating-dropdown',
                options=[
                    {'label': 'Inconclusive', 'value':-1},
                    {'label': 'Biased', 'value': 1},
                    {'label': 'Very Biased', 'value': 2},
                    {'label': 'Not Biased', 'value': 0},
                ],
                placeholder='Select Bias Rating',
                multi=True,
                clearable=True,
                style = {'width': '70%'})
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Bias Category:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart2b-bias-category-dropdown',
                options=[
                    {'label': 'Generalisation', 'value': 'generalisation'},
                    {'label': 'Prominence', 'value': 'prominence'},
                    {'label': 'Negative Behaviour', 'value': 'negative_behaviour'},
                    {'label': 'Misrepresentation', 'value': 'misrepresentation'},
                    {'label': 'Headline or Imagery', 'value': 'headline_or_imagery'},
                ],
                placeholder='Select Bias Category',
                multi=True,
                clearable=True,
                style = {'width': '70%'})
            ], style={'display':'flex', 'margin-bottom':'50px', 'align-items': 'center'}),

            # Graph for displaying the top topics
            dcc.Graph(id='top-topics-bar-chart-2b', style = {'margin-bottom': '50px'}),

            # Table for displaying the top topics
            html.Div(id='table2b-title', style={'fontSize': 20, 'fontColor': '#2E2C2B', 'margin-bottom': '20px'}),
            html.Div(id='table2b'),
            html.Div([
                dbc.Button('Clear Table', id='clear-button2b', style = {'display': 'none'}),
                dbc.Button('Export to CSV', id='export-button2b', style = {'display': 'none'})
            ], style={'display':'flex', 'margin-top': '10px', 'align-items': 'center'}),

        ],
            style={
                'padding':10,
                'flex':1,
                'margin-top': '50px',
                'margin-bottom': '100px',
                'font-family': 'sans-serif'
                }
        ),

    ]),

    # Chart 3A vs Chart 3B
    html.Div(className='row', children=[

        # All elements for Chart 3A
        html.Div([

            html.Div([
                html.Label(['Filter Date Published:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.DatePickerRange(
                    id='chart3a-datepickerrange',
                    display_format='DD MMM YYYY',
                    clearable=True,
                    with_portal=True,
                    max_date_allowed=datetime.today(),
                    start_date=start_date,
                    end_date=end_date,
                    start_date_placeholder_text='Start date',
                    end_date_placeholder_text='End date',
                    style = {'font-size':'15px'}
                )
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Publishers:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart3a-publisher-dropdown',
                options=[{'label': publisher, 'value': publisher} for publisher in unique_publishers],
                placeholder='Select Publisher',
                multi=True,
                clearable=True,
                style = {'width': '70%'})
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Bias Category:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart3a-bias-category-dropdown',
                options=[
                    {'label': 'Generalisation', 'value': 'generalisation'},
                    {'label': 'Prominence', 'value': 'prominence'},
                    {'label': 'Negative Behaviour', 'value': 'negative_behaviour'},
                    {'label': 'Misrepresentation', 'value': 'misrepresentation'},
                    {'label': 'Headline or Imagery', 'value': 'headline_or_imagery'},
                ],
                placeholder='Select Bias Category',
                multi=True,
                clearable=True,
                style = {'width': '70%'})
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Topics:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart3a-topic-dropdown',
                options=[{'label': topic, 'value': topic} for topic in unique_topics],
                placeholder='Select Topic',
                multi=True,
                clearable=True,
                style = {'width': '70%'})
            ], style={'display':'flex', 'margin-bottom':'50px', 'align-items': 'center'}),

            # Graph for displaying the top topics
            dcc.Graph(id='top-offending-articles-bar-chart-3a', style = {'margin-bottom': '50px'}),

            # Table for displaying the top topics
            html.Div(id='table3a-title', style={'fontSize': 20, 'fontColor': '#2E2C2B', 'margin-bottom': '20px'}),
            html.Div(id='table3a'),
            html.Div([
                dbc.Button('Clear Table', id='clear-button3a', style = {'display': 'none'}),
                dbc.Button('Export to CSV', id='export-button3a', style = {'display': 'none'})
            ], style={'display':'flex', 'margin-top': '10px', 'align-items': 'center'}),

        ],
            style={
                'padding':10,
                'flex':1,
                'margin-top': '50px',
                'margin-bottom': '100px',
                'font-family': 'sans-serif'
                }
        ),

        # All elements for Chart 3B
        html.Div([

            html.Div([
                html.Label(['Filter Date Published:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.DatePickerRange(
                    id='chart3b-datepickerrange',
                    display_format='DD MMM YYYY',
                    clearable=True,
                    with_portal=True,
                    max_date_allowed=datetime.today(),
                    start_date=start_date,
                    end_date=end_date,
                    start_date_placeholder_text='Start date',
                    end_date_placeholder_text='End date',
                    style = {'font-size':'15px'}
                )
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Publishers:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart3b-publisher-dropdown',
                options=[{'label': publisher, 'value': publisher} for publisher in unique_publishers],
                placeholder='Select Publisher',
                multi=True,
                clearable=True,
                style = {'width': '70%'})
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Bias Category:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart3b-bias-category-dropdown',
                options=[
                    {'label': 'Generalisation', 'value': 'generalisation'},
                    {'label': 'Prominence', 'value': 'prominence'},
                    {'label': 'Negative Behaviour', 'value': 'negative_behaviour'},
                    {'label': 'Misrepresentation', 'value': 'misrepresentation'},
                    {'label': 'Headline or Imagery', 'value': 'headline_or_imagery'},
                ],
                placeholder='Select Bias Category',
                multi=True,
                clearable=True,
                style = {'width': '70%'})
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Topics:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart3b-topic-dropdown',
                options=[{'label': topic, 'value': topic} for topic in unique_topics],
                placeholder='Select Topic',
                multi=True,
                clearable=True,
                style = {'width': '70%'})
            ], style={'display':'flex', 'margin-bottom':'50px', 'align-items': 'center'}),

            # Graph for displaying the top topics
            dcc.Graph(id='top-offending-articles-bar-chart-3b', style = {'margin-bottom': '50px'}),

            # Table for displaying the top topics
            html.Div(id='table3b-title', style={'fontSize': 20, 'fontColor': '#2E2C2B', 'margin-bottom': '20px'}),
            html.Div(id='table3b'),
            html.Div([
                dbc.Button('Clear Table', id='clear-button3b', style = {'display': 'none'}),
                dbc.Button('Export to CSV', id='export-button3b', style = {'display': 'none'})
            ], style={'display':'flex', 'margin-top': '10px', 'align-items': 'center'}),

        ],
            style={
                'padding':10,
                'flex':1,
                'margin-top': '50px',
                'margin-bottom': '100px',
                'font-family': 'sans-serif'
                }
        ),
    ]),

    # Chart 4A vs Chart 4B
    html.Div(className='row', children=[

        # All elements for Chart 4
        html.Div([

            html.Div([
                html.Label(['Filter Date Published:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.DatePickerRange(
                    id='chart4a-datepickerrange',
                    display_format='DD MMM YYYY',
                    clearable=True,
                    with_portal=True,
                    max_date_allowed=datetime.today(),
                    start_date=start_date,
                    end_date=end_date,
                    start_date_placeholder_text='Start date',
                    end_date_placeholder_text='End date',
                    style={'font-size': '15px'}
                )
            ], style={'display': 'flex', 'margin-bottom': '10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Publishers:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                    id='chart4a-publisher-dropdown',
                    options=[{'label': publisher, 'value': publisher} for publisher in unique_publishers],
                    placeholder='Select Publisher',
                    multi=True,
                    clearable=True,
                    style={'width': '70%'}
                )
            ], style={'display': 'flex', 'margin-bottom': '10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Bias Rating:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                    id='chart4a-bias-rating-dropdown',
                    options=[
                        {'label': 'Biased', 'value': 2},
                        {'label': 'Very Biased', 'value': 1},
                        {'label': 'Not Biased', 'value': 0},
                        {'label': 'Inconclusive', 'value': -1},
                    ],
                    placeholder='Select Bias Rating',
                    multi=True,
                    clearable=True,
                    style={'width': '70%'}
                )
            ], style={'display': 'flex', 'margin-bottom': '10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Bias Category:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                    id='chart4a-bias-category-dropdown',
                    options=[
                        {'label': 'Generalisation', 'value': 'generalisation'},
                        {'label': 'Prominence', 'value': 'prominence'},
                        {'label': 'Negative Behaviour', 'value': 'negative_behaviour'},
                        {'label': 'Misrepresentation', 'value': 'misrepresentation'},
                        {'label': 'Headline or Imagery', 'value': 'headline_or_imagery'},
                    ],
                    placeholder='Select Bias Category',
                    multi=True,
                    clearable=True,
                    style={'width': '70%'}
                )
            ], style={'display': 'flex', 'margin-bottom': '10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Topics:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                    id='chart4a-topic-dropdown',
                    options=[{'label': topic, 'value': topic} for topic in unique_topics],
                    placeholder='Select Topic',
                    multi=True,
                    clearable=True,
                    style={'width': '70%'}
                )
            ], style={'display': 'flex', 'margin-bottom': '10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Select Word Grouping:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                    id='chart4a-ngram-dropdown',
                    options=[
                        {'label': 'Single Word', 'value': 1},
                        {'label': 'Two-Word Phrases', 'value': 2},
                        {'label': 'Three-Word Phrases', 'value': 3}
                    ],
                    value=1,  # default value on load
                    clearable=False,
                    style={'width': '70%'}
                )
            ], style={'display': 'flex', 'margin-bottom': '30px', 'align-items': 'center'}),

            # Toggle for headline-only or full-text word clouds
            dcc.RadioItems(
                id='chart4a-text-toggle',
                options=[
                    {'label': '    Headline-only', 'value': 'title'},
                    {'label': '    Full-text', 'value': 'text'}
                ],
                value='title',  # default value on load
                labelStyle={'display': 'inline-block'},
                inputStyle={"margin-left": "10px"},
                style={'margin-bottom': '50px'}
            ),


            # Word search input and button
            html.Div([
                html.Label(['Word Search:'], style={'font-weight': 'bold', 'width': '20%', 'display': 'block'}),
                dcc.Input(id='word-search-4a', type='text', style={'width': '49%', 'display': 'block'}),
                dbc.Button('Search', id='search-button4a', style={'margin-left': '30px', 'width': '10%', 'display': 'block'})
            ], style={'display': 'flex', 'margin-top': '30px', 'margin-bottom': '30px', 'align-items': 'center'}),

            # Graph for displaying the word cloud
            dcc.Graph(id='wordcloud-container-4a'),

            # Table for displaying the result for word search
            html.Div(id='table4a-title', style={'fontSize': 20, 'color': '#2E2C2B', 'margin-bottom': '20px'}),
            html.Div(id='table4a'),
            html.Div([
                dbc.Button('Clear Table', id='clear-button4a', style={'display': 'none'}),
                dbc.Button('Export to CSV', id='export-button4a', style={'display': 'none'})
            ], style={'display': 'flex', 'margin-top': '10px', 'align-items': 'center'}),
        ], 
            style={
                'padding': 10,
                'flex': 1,
                'margin-top': '50px',
                'margin-bottom': '100px',
                'font-family': 'sans-serif'
                }
        ),

        # All elements for Chart 4B
        html.Div([

            html.Div([
                html.Label(['Filter Date Published:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.DatePickerRange(
                    id='chart4b-datepickerrange',
                    display_format='DD MMM YYYY',
                    clearable=True,
                    with_portal=True,
                    max_date_allowed=datetime.today(),
                    start_date=start_date,
                    end_date=end_date,
                    start_date_placeholder_text='Start date',
                    end_date_placeholder_text='End date',
                    style={'font-size': '15px'}
                )
            ], style={'display': 'flex', 'margin-bottom': '10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Publishers:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                    id='chart4b-publisher-dropdown',
                    options=[{'label': publisher, 'value': publisher} for publisher in unique_publishers],
                    placeholder='Select Publisher',
                    multi=True,
                    clearable=True,
                    style={'width': '70%'}
                )
            ], style={'display': 'flex', 'margin-bottom': '10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Bias Rating:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                    id='chart4b-bias-rating-dropdown',
                    options=[
                        {'label': 'Biased', 'value': 2},
                        {'label': 'Very Biased', 'value': 1},
                        {'label': 'Not Biased', 'value': 0},
                        {'label': 'Inconclusive', 'value': -1},
                    ],
                    placeholder='Select Bias Rating',
                    multi=True,
                    clearable=True,
                    style={'width': '70%'}
                )
            ], style={'display': 'flex', 'margin-bottom': '10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Bias Category:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                    id='chart4b-bias-category-dropdown',
                    options=[
                        {'label': 'Generalisation', 'value': 'generalisation'},
                        {'label': 'Prominence', 'value': 'prominence'},
                        {'label': 'Negative Behaviour', 'value': 'negative_behaviour'},
                        {'label': 'Misrepresentation', 'value': 'misrepresentation'},
                        {'label': 'Headline or Imagery', 'value': 'headline_or_imagery'},
                    ],
                    placeholder='Select Bias Category',
                    multi=True,
                    clearable=True,
                    style={'width': '70%'}
                )
            ], style={'display': 'flex', 'margin-bottom': '10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Filter Topics:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                    id='chart4b-topic-dropdown',
                    options=[{'label': topic, 'value': topic} for topic in unique_topics],
                    placeholder='Select Topic',
                    multi=True,
                    clearable=True,
                    style={'width': '70%'}
                )
            ], style={'display': 'flex', 'margin-bottom': '10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Select Word Grouping:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                    id='chart4b-ngram-dropdown',
                    options=[
                        {'label': 'Single Word', 'value': 1},
                        {'label': 'Two-Word Phrases', 'value': 2},
                        {'label': 'Three-Word Phrases', 'value': 3}
                    ],
                    value=1,  # default value on load
                    clearable=False,
                    style={'width': '70%'}
                )
            ], style={'display': 'flex', 'margin-bottom': '30px', 'align-items': 'center'}),

            # Toggle for headline-only or full-text word clouds
            dcc.RadioItems(
                id='chart4b-text-toggle',
                options=[
                    {'label': '    Headline-only', 'value': 'title'},
                    {'label': '    Full-text', 'value': 'text'}
                ],
                value='title',  # default value on load
                labelStyle={'display': 'inline-block'},
                inputStyle={"margin-left": "10px"},
                style={'margin-bottom': '50px'}
            ),

            # Word search input and button
            html.Div([
                html.Label(['Word Search:'], style={'font-weight': 'bold', 'width': '20%', 'display': 'block'}),
                dcc.Input(id='word-search-4b', type='text', style={'width': '49%', 'display': 'block'}),
                dbc.Button('Search', id='search-button4b', style={'margin-left': '30px', 'width': '10%', 'display': 'block'})
            ], style={'display': 'flex', 'margin-top': '30px', 'margin-bottom': '30px', 'align-items': 'center'}),

            # Graph for displaying the word cloud
            dcc.Graph(id='wordcloud-container-4b'),

            # Table for displaying the result for word search
            html.Div(id='table4b-title', style={'fontSize': 20, 'color': '#2E2C2B', 'margin-bottom': '20px'}),
            html.Div(id='table4b'),
            html.Div([
                dbc.Button('Clear Table', id='clear-button4b', style={'display': 'none'}),
                dbc.Button('Export to CSV', id='export-button4b', style={'display': 'none'})
            ], style={'display': 'flex', 'margin-top': '10px', 'align-items': 'center'}),
        ], 
            style={
                'padding': 10,
                'flex': 1,
                'margin-top': '50px',
                'margin-bottom': '100px',
                'font-family': 'sans-serif'
                }
        ),

    ])


])

# Callback for Chart 1A
@app.callback(

    Output('top-offending-publishers-bar-chart-1a', 'figure'),
    [
        Input('chart1a-datepickerrange', 'start_date'),
        Input('chart1a-datepickerrange', 'end_date'),
        Input('chart1a-publisher-dropdown', 'value'),
        Input('chart1a-bias-rating-dropdown', 'value'),
        Input('chart1a-bias-category-dropdown', 'value'),
        Input('chart1a-topic-dropdown', 'value'),
        Input('chart1a-color-toggle', 'value')
    ]
)

def update_chart1a(selected_start_date, selected_end_date, selected_publishers, selected_bias_ratings, selected_bias_categories, selected_topics, color_by):
    filtered_df = df_corpus.copy()

    # Apply filters for quarters, publishers, and topics
    if (selected_start_date is not None) & (selected_end_date is not None):
        start_date = pd.to_datetime(str(selected_start_date))
        end_date = pd.to_datetime(str(selected_end_date))
        filtered_df = filtered_df[(filtered_df['date_published']>=start_date) & (filtered_df['date_published']<=end_date)]
    if selected_publishers:
        filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
    if selected_bias_ratings:
        filtered_df = filtered_df[filtered_df['bias_rating'].isin(selected_bias_ratings)]
    if selected_bias_categories:
        filtered_df = filtered_df[filtered_df[selected_bias_categories].sum(axis=1) > 0]
    if selected_topics:
        filtered_df = filtered_df[filtered_df['topic'].str.contains('|'.join(selected_topics))]

    # If chart is empty, show text instead
    if filtered_df.shape[0]==0:
        data = []
        layout = {
            'xaxis': {'visible': False},
            'yaxis': {'visible': False},
            'template': 'simple_white',
            'height': 400,
            'annotations': [{
                'text': 'No articles found in the current selection.',
                'showarrow': False,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'font': {'size': 20, 'color': '#2E2C2B'}
            }]
        }

    else:
        # Calculate the total counts of very biased and biased articles for each publisher
        publisher_totals = filtered_df.groupby('publisher', observed=True).size()

        # Sort publishers by this count and get the top 10
        top_publishers = publisher_totals.sort_values(ascending=False).head(10).index[::-1]

        # Filter the dataframe to include only the top publishers
        filtered_df = filtered_df[filtered_df['publisher'].isin(top_publishers)]
        filtered_df['publisher'] = pd.Categorical(filtered_df['publisher'], ordered=True, categories=top_publishers)
        filtered_df = filtered_df.sort_values('publisher')

        if color_by == 'bias_ratings':
            # Color mapping for bias ratings
            color_map = {
                -1: ('#CAC6C2', 'Inconclusive'),
                0: ('#f2eadf', 'Not Biased'), # #FFE5DC
                1: ('#eb8483', 'Biased'),
                2: ('#C22625', 'Very Biased')
            }
            # Prepare legend tracking
            legend_added = set()
            data = []
            for publisher in top_publishers:
                total_biased_articles = filtered_df[filtered_df['publisher'] == publisher]['bias_rating'].count()

                for rating, (color, name) in color_map.items():
                    articles = filtered_df[(filtered_df['publisher'] == publisher) &
                                            (filtered_df['bias_rating'] == rating)]['bias_rating'].count()

                    percentage_of_total = (articles / total_biased_articles) * 100 if total_biased_articles > 0 else 0

                    tooltip_text = (
                        f"<b>Publisher: </b>{publisher}<br>"
                        f"<b>Bias Rating:</b> {name}<br>"
                        f"<b>Number of Articles:</b> {articles}<br>"
                        f"This accounts for <b>{percentage_of_total:.2f}%</b> of the total available articles in the current selection.<br>"
                        # f"<b>Percentage of Total:</b> {percentage_of_total:.2f}%"
                    )

                    showlegend = name not in legend_added
                    legend_added.add(name)

                    data.append(go.Bar(
                        x=[articles],
                        y=[publisher],
                        name=name,
                        orientation='h',
                        marker=dict(color=color),
                        showlegend=showlegend,
                        text=tooltip_text,
                        hoverinfo='text',
                        textposition='none'
                    ))

            # Update the layout
            layout = go.Layout(
                title=f"""<b>Who are the top offending publishers?</b>""",
                xaxis=dict(title='Number of Articles'),
                yaxis=dict(title='Publisher'),
                hovermode='closest',
                barmode='stack',
                showlegend=True,
                hoverlabel=dict(
                    align='left'
                ),
                template="simple_white",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_color='#2E2C2B',
                font_size=14,
                height=800,
                margin={'l': 150, 'r': 20, 'b': 40, 't': 40}
            )

        elif color_by == 'bias_categories':
            categories = ['generalisation', 'prominence', 'negative_behaviour', 'misrepresentation', 'headline_or_imagery']
            category_colors = ['#4185A0', '#AA4D71', '#B85C3B', '#C5BE71', '#7658A0']  # example colors

            # Prepare legend tracking
            legend_added = set()
            data = []
            filtered_df['total_bias_category'] = filtered_df[categories].sum(axis=1)

            for i, category in enumerate(categories):
                articles_list = []
                tooltip_text_list = []
                for publisher in filtered_df['publisher'].unique():
                    # Summing the 'total_bias_category' column which was pre-calculated
                    total_biased_articles = filtered_df[filtered_df['publisher'] == publisher].shape[0]

                    # Count the number of rows where the category column has a 1 for this publisher
                    articles = filtered_df[(filtered_df['publisher'] == publisher) & (filtered_df[category] == 1)].shape[0]
                    articles_list += [articles]

                    # Calculate the percentage of total articles for the current category
                    percentage_of_total = (articles / total_biased_articles * 100) if total_biased_articles > 0 else 0
                    tooltip_text = (
                            f"<b>Publisher: </b>{publisher}<br>"
                            f"<b>Bias Category: </b>{category.replace('_', ' ').title().replace('Or', 'or')}<br>"
                            f"Of the {total_biased_articles} articles, <b>{articles}</b> of them committed <b>{category.replace('_', ' ').title().replace('Or', 'or')}</b>.<br>"
                            f"This accounts for <b>{percentage_of_total:.2f}%</b> of the total available articles for <b>{category.replace('_', ' ').title().replace('Or', 'or')}</b>.<br>"
                            # f"<b>Percentage of Total: </b>{percentage_of_total:.2f}%"
                    )
                    tooltip_text_list += [tooltip_text]

                showlegend = category not in legend_added  # determine showlegend based on current category
                legend_added.add(category)

                data.append(go.Bar(
                    x=articles_list,
                    y=top_publishers,
                    name=category.replace('_', ' ').title().replace('Or', 'or'),
                    orientation='h',
                    marker=dict(color=category_colors[i]),
                    showlegend=showlegend,
                    text=tooltip_text_list,
                    hoverinfo='text',
                    textposition='none'
                ))

            # Update the layout
            layout = go.Layout(
                title=f"""<b>Who are the top offending publishers?</b>""",
                xaxis=dict(title='Number of Articles'),
                yaxis=dict(title='Publisher'),
                hovermode='closest',
                barmode='group',
                showlegend=True,
                hoverlabel=dict(
                    align='left'
                ),
                template="simple_white",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_color='#2E2C2B',
                font_size=14,
                height=800,
                margin={'l': 150, 'r': 20, 'b': 40, 't': 40}
            )

    return {'data': data, 'layout': layout}


# Callback for Chart 1B
@app.callback(

    Output('top-offending-publishers-bar-chart-1b', 'figure'),
    [
        Input('chart1b-datepickerrange', 'start_date'),
        Input('chart1b-datepickerrange', 'end_date'),
        Input('chart1b-publisher-dropdown', 'value'),
        Input('chart1b-bias-rating-dropdown', 'value'),
        Input('chart1b-bias-category-dropdown', 'value'),
        Input('chart1b-topic-dropdown', 'value'),
        Input('chart1b-color-toggle', 'value')
    ]
)

def update_chart1b(selected_start_date, selected_end_date, selected_publishers, selected_bias_ratings, selected_bias_categories, selected_topics, color_by):
    filtered_df = df_corpus.copy()

    # Apply filters for quarters, publishers, and topics
    if (selected_start_date is not None) & (selected_end_date is not None):
        start_date = pd.to_datetime(str(selected_start_date))
        end_date = pd.to_datetime(str(selected_end_date))
        filtered_df = filtered_df[(filtered_df['date_published']>=start_date) & (filtered_df['date_published']<=end_date)]
    if selected_publishers:
        filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
    if selected_bias_ratings:
        filtered_df = filtered_df[filtered_df['bias_rating'].isin(selected_bias_ratings)]
    if selected_bias_categories:
        filtered_df = filtered_df[filtered_df[selected_bias_categories].sum(axis=1) > 0]
    if selected_topics:
        filtered_df = filtered_df[filtered_df['topic'].str.contains('|'.join(selected_topics))]

    # If chart is empty, show text instead
    if filtered_df.shape[0]==0:
        data = []
        layout = {
            'xaxis': {'visible': False},
            'yaxis': {'visible': False},
            'template': 'simple_white',
            'height': 400,
            'annotations': [{
                'text': 'No articles found in the current selection.',
                'showarrow': False,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'font': {'size': 20, 'color': '#2E2C2B'}
            }]
        }

    else:
        # Calculate the total counts of very biased and biased articles for each publisher
        publisher_totals = filtered_df.groupby('publisher', observed=True).size()

        # Sort publishers by this count and get the top 10
        top_publishers = publisher_totals.sort_values(ascending=False).head(10).index[::-1]

        # Filter the dataframe to include only the top publishers
        filtered_df = filtered_df[filtered_df['publisher'].isin(top_publishers)]
        filtered_df['publisher'] = pd.Categorical(filtered_df['publisher'], ordered=True, categories=top_publishers)
        filtered_df = filtered_df.sort_values('publisher')

        if color_by == 'bias_ratings':
            # Color mapping for bias ratings
            color_map = {
                -1: ('#CAC6C2', 'Inconclusive'),
                0: ('#f2eadf', 'Not Biased'), # #FFE5DC
                1: ('#eb8483', 'Biased'),
                2: ('#C22625', 'Very Biased')
            }
            # Prepare legend tracking
            legend_added = set()
            data = []
            for publisher in top_publishers:
                total_biased_articles = filtered_df[filtered_df['publisher'] == publisher]['bias_rating'].count()

                for rating, (color, name) in color_map.items():
                    articles = filtered_df[(filtered_df['publisher'] == publisher) &
                                            (filtered_df['bias_rating'] == rating)]['bias_rating'].count()

                    percentage_of_total = (articles / total_biased_articles) * 100 if total_biased_articles > 0 else 0

                    tooltip_text = (
                        f"<b>Publisher: </b>{publisher}<br>"
                        f"<b>Bias Rating:</b> {name}<br>"
                        f"<b>Number of Articles:</b> {articles}<br>"
                        f"This accounts for <b>{percentage_of_total:.2f}%</b> of the total available articles in the current selection.<br>"
                        # f"<b>Percentage of Total:</b> {percentage_of_total:.2f}%"
                    )

                    showlegend = name not in legend_added
                    legend_added.add(name)

                    data.append(go.Bar(
                        x=[articles],
                        y=[publisher],
                        name=name,
                        orientation='h',
                        marker=dict(color=color),
                        showlegend=showlegend,
                        text=tooltip_text,
                        hoverinfo='text',
                        textposition='none'
                    ))

            # Update the layout
            layout = go.Layout(
                title=f"""<b>Who are the top offending publishers?</b>""",
                xaxis=dict(title='Number of Articles'),
                yaxis=dict(title='Publisher'),
                hovermode='closest',
                barmode='stack',
                showlegend=True,
                hoverlabel=dict(
                    align='left'
                ),
                template="simple_white",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_color='#2E2C2B',
                font_size=14,
                height=800,
                margin={'l': 150, 'r': 20, 'b': 40, 't': 40}
            )

        elif color_by == 'bias_categories':
            categories = ['generalisation', 'prominence', 'negative_behaviour', 'misrepresentation', 'headline_or_imagery']
            category_colors = ['#4185A0', '#AA4D71', '#B85C3B', '#C5BE71', '#7658A0']  # example colors

            # Prepare legend tracking
            legend_added = set()
            data = []
            filtered_df['total_bias_category'] = filtered_df[categories].sum(axis=1)

            for i, category in enumerate(categories):
                articles_list = []
                tooltip_text_list = []
                for publisher in filtered_df['publisher'].unique():
                    # Summing the 'total_bias_category' column which was pre-calculated
                    total_biased_articles = filtered_df[filtered_df['publisher'] == publisher].shape[0]

                    # Count the number of rows where the category column has a 1 for this publisher
                    articles = filtered_df[(filtered_df['publisher'] == publisher) & (filtered_df[category] == 1)].shape[0]
                    articles_list += [articles]

                    # Calculate the percentage of total articles for the current category
                    percentage_of_total = (articles / total_biased_articles * 100) if total_biased_articles > 0 else 0
                    tooltip_text = (
                            f"<b>Publisher: </b>{publisher}<br>"
                            f"<b>Bias Category: </b>{category.replace('_', ' ').title().replace('Or', 'or')}<br>"
                            f"Of the {total_biased_articles} articles, <b>{articles}</b> of them committed <b>{category.replace('_', ' ').title().replace('Or', 'or')}</b>.<br>"
                            f"This accounts for <b>{percentage_of_total:.2f}%</b> of the total available articles for <b>{category.replace('_', ' ').title().replace('Or', 'or')}</b>.<br>"
                            # f"<b>Percentage of Total: </b>{percentage_of_total:.2f}%"
                    )
                    tooltip_text_list += [tooltip_text]

                showlegend = category not in legend_added  # determine showlegend based on current category
                legend_added.add(category)

                data.append(go.Bar(
                    x=articles_list,
                    y=top_publishers,
                    name=category.replace('_', ' ').title().replace('Or', 'or'),
                    orientation='h',
                    marker=dict(color=category_colors[i]),
                    showlegend=showlegend,
                    text=tooltip_text_list,
                    hoverinfo='text',
                    textposition='none'
                ))

            # Update the layout
            layout = go.Layout(
                title=f"""<b>Who are the top offending publishers?</b>""",
                xaxis=dict(title='Number of Articles'),
                yaxis=dict(title='Publisher'),
                hovermode='closest',
                barmode='group',
                showlegend=True,
                hoverlabel=dict(
                    align='left'
                ),
                template="simple_white",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_color='#2E2C2B',
                font_size=14,
                height=800,
                margin={'l': 150, 'r': 20, 'b': 40, 't': 40}
            )

    return {'data': data, 'layout': layout}

# Callback for Chart 2A
@app.callback(
    Output('top-topics-bar-chart-2a', 'figure'),
    [
        Input('chart2a-datepickerrange', 'start_date'),
        Input('chart2a-datepickerrange', 'end_date'),
        Input('chart2a-publisher-dropdown', 'value'),
        Input('chart2a-bias-rating-dropdown', 'value'),
        Input('chart2a-bias-category-dropdown', 'value')
    ]
)

def update_chart2a(selected_start_date, selected_end_date, selected_publishers, selected_bias_ratings, selected_bias_categories):
    filtered_df = df_corpus.copy()

    # Apply filters for dates, publishers, ratings, and categories
    if (selected_start_date is not None) & (selected_end_date is not None):
        start_date = pd.to_datetime(str(selected_start_date))
        end_date = pd.to_datetime(str(selected_end_date))
        filtered_df = filtered_df[(filtered_df['date_published']>=start_date) & (filtered_df['date_published']<=end_date)]
    if selected_publishers:
        filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
    if selected_bias_ratings:
        filtered_df = filtered_df[filtered_df['bias_rating'].isin(selected_bias_ratings)]
    if selected_bias_categories:
        filtered_df = filtered_df[filtered_df[selected_bias_categories].sum(axis=1) > 0]

    # If chart is empty, show text instead
    if filtered_df.shape[0]==0:
        data = []
        layout = {
            'xaxis': {'visible': False},
            'yaxis': {'visible': False},
            'template': 'simple_white',
            'height': 400,
            'annotations': [{
                'text': 'No articles found in the current selection.',
                'showarrow': False,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'font': {'size': 20, 'color': '#2E2C2B'}
            }]
        }

    else:
        # Aggregate topics
        filtered_df_exploded = filtered_df[['article_url', 'topic_list']].explode('topic_list')
        topic_counts = filtered_df_exploded.groupby('topic_list', observed=True).size().sort_values(ascending=False)
        total_articles = topic_counts.sum()

        # Predefine colors for the top 5 topics
        top_colors = ['#4185A0', '#AA4D71', '#B85C3B', '#C5BE71', '#7658A0']
        gray_color = '#CAC6C2' # Add gray color for the remaining topics

        # Create bars for the bar chart
        data = []
        for i, (topic, count) in enumerate(topic_counts.items()):
            tooltip_text = (
                f"<b>Topic: </b>{topic}<br>"
                f"<b>Number of Articles: </b>{count}<br>"
                f"This accounts for <b>{count/total_articles:.2%}%</b> of the total available articles in the current selection.<br>"
                # f"<b>Percentage of Total: </b>{count/total_articles:.2%}"
            )

            bar = go.Bar(
                y=[topic],
                x=[count],
                orientation='h',
                marker=dict(color=top_colors[i] if i < 5 else gray_color),
                text=tooltip_text,
                hoverinfo='text',
                textposition='none'
            )
            data.append(bar)

        # Update the layout
        layout = go.Layout(
            title='<b>What are the most popular topics?</b>',
            xaxis=dict(title='Number of Articles'),
            yaxis=dict(title='Topics', autorange='reversed', tickmode='array', tickvals=list(range(len(topic_counts))), ticktext=topic_counts.index.tolist()),
            hovermode='closest',
            barmode='stack',
            showlegend=False,
            hoverlabel=dict(
                align='left'
            ),
            template="simple_white",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='#2E2C2B',
            font_size=14,
            height=800,
            margin={'l': 150, 'r': 20, 'b': 40, 't': 40}
        )

    return {'data': data, 'layout': layout}


# Callback for Chart 2B
@app.callback(
    Output('top-topics-bar-chart-2b', 'figure'),
    [
        Input('chart2b-datepickerrange', 'start_date'),
        Input('chart2b-datepickerrange', 'end_date'),
        Input('chart2b-publisher-dropdown', 'value'),
        Input('chart2b-bias-rating-dropdown', 'value'),
        Input('chart2b-bias-category-dropdown', 'value')
    ]
)

def update_chart2b(selected_start_date, selected_end_date, selected_publishers, selected_bias_ratings, selected_bias_categories):
    filtered_df = df_corpus.copy()

    # Apply filters for dates, publishers, ratings, and categories
    if (selected_start_date is not None) & (selected_end_date is not None):
        start_date = pd.to_datetime(str(selected_start_date))
        end_date = pd.to_datetime(str(selected_end_date))
        filtered_df = filtered_df[(filtered_df['date_published']>=start_date) & (filtered_df['date_published']<=end_date)]
    if selected_publishers:
        filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
    if selected_bias_ratings:
        filtered_df = filtered_df[filtered_df['bias_rating'].isin(selected_bias_ratings)]
    if selected_bias_categories:
        filtered_df = filtered_df[filtered_df[selected_bias_categories].sum(axis=1) > 0]

    # If chart is empty, show text instead
    if filtered_df.shape[0]==0:
        data = []
        layout = {
            'xaxis': {'visible': False},
            'yaxis': {'visible': False},
            'template': 'simple_white',
            'height': 400,
            'annotations': [{
                'text': 'No articles found in the current selection.',
                'showarrow': False,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'font': {'size': 20, 'color': '#2E2C2B'}
            }]
        }

    else:
        # Aggregate topics
        filtered_df_exploded = filtered_df[['article_url', 'topic_list']].explode('topic_list')
        topic_counts = filtered_df_exploded.groupby('topic_list', observed=True).size().sort_values(ascending=False)
        total_articles = topic_counts.sum()

        # Predefine colors for the top 5 topics
        top_colors = ['#4185A0', '#AA4D71', '#B85C3B', '#C5BE71', '#7658A0']
        gray_color = '#CAC6C2' # Add gray color for the remaining topics

        # Create bars for the bar chart
        data = []
        for i, (topic, count) in enumerate(topic_counts.items()):
            tooltip_text = (
                f"<b>Topic: </b>{topic}<br>"
                f"<b>Number of Articles: </b>{count}<br>"
                f"This accounts for <b>{count/total_articles:.2%}%</b> of the total available articles in the current selection.<br>"
                # f"<b>Percentage of Total: </b>{count/total_articles:.2%}"
            )

            bar = go.Bar(
                y=[topic],
                x=[count],
                orientation='h',
                marker=dict(color=top_colors[i] if i < 5 else gray_color),
                text=tooltip_text,
                hoverinfo='text',
                textposition='none'
            )
            data.append(bar)

        # Update the layout
        layout = go.Layout(
            title='<b>What are the most popular topics?</b>',
            xaxis=dict(title='Number of Articles'),
            yaxis=dict(title='Topics', autorange='reversed', tickmode='array', tickvals=list(range(len(topic_counts))), ticktext=topic_counts.index.tolist()),
            hovermode='closest',
            barmode='stack',
            showlegend=False,
            hoverlabel=dict(
                align='left'
            ),
            template="simple_white",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='#2E2C2B',
            font_size=14,
            height=800,
            margin={'l': 150, 'r': 20, 'b': 40, 't': 40}
        )

    return {'data': data, 'layout': layout}


# Callback for Chart 3A
@app.callback(
    Output('top-offending-articles-bar-chart-3a', 'figure'),
    [
        Input('chart3a-datepickerrange', 'start_date'),
        Input('chart3a-datepickerrange', 'end_date'),
        Input('chart3a-publisher-dropdown', 'value'),
        Input('chart3a-bias-category-dropdown', 'value'),
        Input('chart3a-topic-dropdown', 'value')
    ]
)

def update_chart3a(selected_start_date, selected_end_date, selected_publishers, selected_bias_categories, selected_topics):
    filtered_df = df_corpus.copy()

    # Apply filters for dates, publishers, ratings, and categories
    if (selected_start_date is not None) & (selected_end_date is not None):
        start_date = pd.to_datetime(str(selected_start_date))
        end_date = pd.to_datetime(str(selected_end_date))
        filtered_df = filtered_df[(filtered_df['date_published']>=start_date) & (filtered_df['date_published']<=end_date)]
    if selected_publishers:
        filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
    if selected_bias_categories:
        filtered_df = filtered_df[filtered_df[selected_bias_categories].sum(axis=1) > 0]
    if selected_topics:
        filtered_df = filtered_df[filtered_df['topic'].str.contains('|'.join(selected_topics))]

    # If chart is empty, show text instead
    if filtered_df.shape[0]==0:
        data = []
        layout = {
            'xaxis': {'visible': False},
            'yaxis': {'visible': False},
            'template': 'simple_white',
            'height': 400,
            'annotations': [{
                'text': 'No articles found in the current selection.',
                'showarrow': False,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'font': {'size': 20, 'color': '#2E2C2B'}
            }]
        }

    else:
        # Aggregate count per bias rating
        label_map = {
                -1: 'Inconclusive',
                0: 'Not Biased',
                1: 'Biased',
                2: 'Very Biased'
            }
        filtered_df['bias_rating_label'] = filtered_df['bias_rating'].map(label_map)
        filtered_df['bias_rating_label'] = pd.Categorical(filtered_df['bias_rating_label'], categories=['Inconclusive', 'Not Biased', 'Biased', 'Very Biased'], ordered=True)
        bias_counts = filtered_df.groupby('bias_rating_label', observed=True).size()
        total_articles = bias_counts.sum()

        # Predefine colors for the top 5 topics
        color_map = {
                'Inconclusive': '#CAC6C2',
                'Not Biased': '#f2eadf',
                'Biased': '#eb8483',
                'Very Biased': '#C22625'
            }

        # Create bars for the bar chart
        data = []
        for (bias, count) in bias_counts.items():
            tooltip_text = (
                f"<b>Bias Rating: </b>{bias}<br>"
                f"<b>Number of Articles: </b>{count}<br>"
                f"This accounts for <b>{count/total_articles:.2%}%</b> of the total available articles in the current selection.<br>"
                # f"<b>Percentage of Total: </b>{count/total_articles:.2%}"
            )

            bar = go.Bar(
                y=[bias],
                x=[count],
                orientation='h',
                marker=dict(color=color_map[bias]),
                text=tooltip_text,
                hoverinfo='text',
                textposition='none'
            )
            data.append(bar)

        # Update the layout
        layout = go.Layout(
            title='<b>Which are the top offending articles?</b>',
            xaxis=dict(title='Number of Articles'),
            yaxis=dict(title='Bias Rating', tickmode='array', tickvals=list(range(len(bias_counts))), ticktext=bias_counts.index.tolist()),
            hovermode='closest',
            barmode='stack',
            showlegend=False,
            hoverlabel=dict(
                align='left'
            ),
            template="simple_white",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='#2E2C2B',
            font_size=14,
            height=800,
            margin={'l': 150, 'r': 20, 'b': 40, 't': 40}
        )

        # If chart is empty, show text instead
        if filtered_df.shape[0]==0:
            data = []
            layout = {
                'xaxis': {'visible': False},
                'yaxis': {'visible': False},
                'template': 'simple_white',
                'height': 400,
                'annotations': [{
                    'text': 'No articles found in the current selection.',
                    'showarrow': False,
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'font': {'size': 20, 'color': '#2E2C2B'}
                }]
            }

    return {'data': data, 'layout': layout}


# Callback for Chart 3B
@app.callback(
    Output('top-offending-articles-bar-chart-3b', 'figure'),
    [
        Input('chart3b-datepickerrange', 'start_date'),
        Input('chart3b-datepickerrange', 'end_date'),
        Input('chart3b-publisher-dropdown', 'value'),
        Input('chart3b-bias-category-dropdown', 'value'),
        Input('chart3b-topic-dropdown', 'value')
    ]
)

def update_chart3b(selected_start_date, selected_end_date, selected_publishers, selected_bias_categories, selected_topics):
    filtered_df = df_corpus.copy()

    # Apply filters for dates, publishers, ratings, and categories
    if (selected_start_date is not None) & (selected_end_date is not None):
        start_date = pd.to_datetime(str(selected_start_date))
        end_date = pd.to_datetime(str(selected_end_date))
        filtered_df = filtered_df[(filtered_df['date_published']>=start_date) & (filtered_df['date_published']<=end_date)]
    if selected_publishers:
        filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
    if selected_bias_categories:
        filtered_df = filtered_df[filtered_df[selected_bias_categories].sum(axis=1) > 0]
    if selected_topics:
        filtered_df = filtered_df[filtered_df['topic'].str.contains('|'.join(selected_topics))]

    # If chart is empty, show text instead
    if filtered_df.shape[0]==0:
        data = []
        layout = {
            'xaxis': {'visible': False},
            'yaxis': {'visible': False},
            'template': 'simple_white',
            'height': 400,
            'annotations': [{
                'text': 'No articles found in the current selection.',
                'showarrow': False,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'font': {'size': 20, 'color': '#2E2C2B'}
            }]
        }

    else:
        # Aggregate count per bias rating
        label_map = {
                -1: 'Inconclusive',
                0: 'Not Biased',
                1: 'Biased',
                2: 'Very Biased'
            }
        filtered_df['bias_rating_label'] = filtered_df['bias_rating'].map(label_map)
        filtered_df['bias_rating_label'] = pd.Categorical(filtered_df['bias_rating_label'], categories=['Inconclusive', 'Not Biased', 'Biased', 'Very Biased'], ordered=True)
        bias_counts = filtered_df.groupby('bias_rating_label', observed=True).size()
        total_articles = bias_counts.sum()

        # Predefine colors for the top 5 topics
        color_map = {
                'Inconclusive': '#CAC6C2',
                'Not Biased': '#f2eadf',
                'Biased': '#eb8483',
                'Very Biased': '#C22625'
            }

        # Create bars for the bar chart
        data = []
        for (bias, count) in bias_counts.items():
            tooltip_text = (
                f"<b>Bias Rating: </b>{bias}<br>"
                f"<b>Number of Articles: </b>{count}<br>"
                f"This accounts for <b>{count/total_articles:.2%}%</b> of the total available articles in the current selection.<br>"
                # f"<b>Percentage of Total: </b>{count/total_articles:.2%}"
            )

            bar = go.Bar(
                y=[bias],
                x=[count],
                orientation='h',
                marker=dict(color=color_map[bias]),
                text=tooltip_text,
                hoverinfo='text',
                textposition='none'
            )
            data.append(bar)

        # Update the layout
        layout = go.Layout(
            title='<b>Which are the top offending articles?</b>',
            xaxis=dict(title='Number of Articles'),
            yaxis=dict(title='Bias Rating', tickmode='array', tickvals=list(range(len(bias_counts))), ticktext=bias_counts.index.tolist()),
            hovermode='closest',
            barmode='stack',
            showlegend=False,
            hoverlabel=dict(
                align='left'
            ),
            template="simple_white",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='#2E2C2B',
            font_size=14,
            height=800,
            margin={'l': 150, 'r': 20, 'b': 40, 't': 40}
        )

        # If chart is empty, show text instead
        if filtered_df.shape[0]==0:
            data = []
            layout = {
                'xaxis': {'visible': False},
                'yaxis': {'visible': False},
                'template': 'simple_white',
                'height': 400,
                'annotations': [{
                    'text': 'No articles found in the current selection.',
                    'showarrow': False,
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'font': {'size': 20, 'color': '#2E2C2B'}
                }]
            }

    return {'data': data, 'layout': layout}

# Callback for Chart 4A
@app.callback(
    Output('wordcloud-container-4a', 'figure'),
    [
        Input('chart4a-datepickerrange', 'start_date'),
        Input('chart4a-datepickerrange', 'end_date'),
        Input('chart4a-publisher-dropdown', 'value'),
        Input('chart4a-topic-dropdown', 'value'),
        Input('chart4a-bias-category-dropdown', 'value'),
        Input('chart4a-bias-rating-dropdown', 'value'),
        Input('chart4a-text-toggle', 'value'),
        Input('chart4a-ngram-dropdown', 'value')
    ]
)
def update_chart4a(selected_start_date, selected_end_date, selected_publishers, selected_topics, selected_bias_categories, selected_bias_ratings, text_by, ngram_value):
    filtered_df = df_corpus.copy()

    # Apply filters for dates, publishers, and topics
    if selected_start_date and selected_end_date:
        start_date = pd.to_datetime(selected_start_date)
        end_date = pd.to_datetime(selected_end_date)
        filtered_df = filtered_df[(filtered_df['date_published'] >= start_date) & (filtered_df['date_published'] <= end_date)]
    if selected_publishers:
        filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
    if selected_topics:
        filtered_df = filtered_df[filtered_df['topic'].str.contains('|'.join(selected_topics))]
    if selected_bias_ratings:
        filtered_df = filtered_df[filtered_df['bias_rating'].isin(selected_bias_ratings)]
    if selected_bias_categories:
        filtered_df = filtered_df[filtered_df[selected_bias_categories].sum(axis=1) > 0]

    # If chart is empty, show text instead
    if filtered_df.shape[0]==0:
        data = []
        layout = {
            'xaxis': {'visible': False},
            'yaxis': {'visible': False},
            'template': 'simple_white',
            'height': 400,
            'annotations': [{
                'text': 'No articles found in the current selection.',
                'showarrow': False,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'font': {'size': 20, 'color': '#2E2C2B'}
            }]
        }
        fig = go.Figure(data=data, layout=layout)

    else:
        # Generate n-grams
        text = ' '.join(filtered_df[text_by].dropna().values)
        vectorizer = CountVectorizer(ngram_range=(ngram_value, ngram_value), stop_words='english')
        ngram_counts = vectorizer.fit_transform([text])
        ngram_freq = ngram_counts.toarray().flatten()
        ngram_names = vectorizer.get_feature_names_out()
        word_counts = dict(zip(ngram_names, ngram_freq))

        total_words = sum(word_counts.values())
        wc = WordCloud(background_color='white',
                      max_words=100,
                      width=1600,
                      height=1200,
                      scale=1.5,
                      margin=100,
                      max_font_size=100,
                      stopwords=set(STOPWORDS)
                      ).generate_from_frequencies(word_counts)

        # Get word positions and frequencies
        word_positions = wc.layout_

        # Extract positions and other data for Scatter plot
        words = []
        x = []
        y = []
        sizes = []
        hover_texts = []
        frequencies = []

        for (word, freq), font_size, position, orientation, color in word_positions:
            words.append(word)
            x.append(position[0])
            y.append(position[1])
            sizes.append(font_size)
            frequencies.append(freq)
            raw_count = word_counts[word]
            percentage = (raw_count / total_words) * 100
            hover_texts.append(f"<b>Word: </b>{word}<br>"
                              f"The word <b>'{word}'</b> appeared <b>{raw_count}</b> times across all articles in the current selection.<br>"
                              f"This accounts for <b>{percentage:.2f}%</b> of the total available word/phrases.<br>"
                              f"<br>"
                              f"Type <b>'{word}'</b> in the Word Search below to find out which articles used this word.")
#                               f"<b>Percentage of Total: x</b>{percentage:.2f}%")

        # Identify top 10 words by frequency
        top_10_indices = np.argsort(frequencies)[-10:]
        colors = ['#CFCFCF'] * len(words)
        custom_colors = [
            # '#413F42', #top 5
            # '#6B2C32',
            # '#983835',
            # '#BF4238',
            # '#C42625', #top 1

            '#413F42', # top 10

            '#6B2C32', # top 9
            '#6B2C32', # top 8

            '#983835', # top 7
            '#983835', # top 6

            '#BF4238', # top 5
            '#BF4238', # top 4

            '#C42625', #top 3
            '#C42625', #top 2
            '#C42625', #top 1
        ]

        # Apply custom colors to the top 10 words
        for i, idx in enumerate(top_10_indices):
            colors[idx] = custom_colors[i % len(custom_colors)]

        # Sort words by frequency to ensure top words appear on top
        sorted_indices = np.argsort(frequencies)
        words = [words[i] for i in sorted_indices]
        x = [x[i] for i in sorted_indices]
        y = [y[i] for i in sorted_indices]
        sizes = [sizes[i] for i in sorted_indices]
        hover_texts = [hover_texts[i] for i in sorted_indices]
        colors = [colors[i] for i in sorted_indices]

        # Create the Plotly figure with Scatter plot
        fig = go.Figure()

        # Add words as Scatter plot points
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='text',
            text=words,
            textfont=dict(size=sizes, color=colors),
            hovertext=hover_texts,
            hoverinfo='text'
        ))

        # Update the layout
        fig.update_layout(
            title='<b>What are the trending words or phrases?</b>',
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            template='simple_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='#2E2C2B',
            font_size=14,
            height=800,
            margin={'l': 150, 'r': 20, 'b': 40, 't': 40}
        )

        # Reverse the y-axis to match the word cloud orientation
        fig.update_yaxes(autorange="reversed")

    return fig


# Callback for Chart 4B
@app.callback(
    Output('wordcloud-container-4b', 'figure'),
    [
        Input('chart4b-datepickerrange', 'start_date'),
        Input('chart4b-datepickerrange', 'end_date'),
        Input('chart4b-publisher-dropdown', 'value'),
        Input('chart4b-topic-dropdown', 'value'),
        Input('chart4b-bias-category-dropdown', 'value'),
        Input('chart4b-bias-rating-dropdown', 'value'),
        Input('chart4b-text-toggle', 'value'),
        Input('chart4b-ngram-dropdown', 'value')
    ]
)
def update_chart4b(selected_start_date, selected_end_date, selected_publishers, selected_topics, selected_bias_categories, selected_bias_ratings, text_by, ngram_value):
    filtered_df = df_corpus.copy()

    # Apply filters for dates, publishers, and topics
    if selected_start_date and selected_end_date:
        start_date = pd.to_datetime(selected_start_date)
        end_date = pd.to_datetime(selected_end_date)
        filtered_df = filtered_df[(filtered_df['date_published'] >= start_date) & (filtered_df['date_published'] <= end_date)]
    if selected_publishers:
        filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
    if selected_topics:
        filtered_df = filtered_df[filtered_df['topic'].str.contains('|'.join(selected_topics))]
    if selected_bias_ratings:
        filtered_df = filtered_df[filtered_df['bias_rating'].isin(selected_bias_ratings)]
    if selected_bias_categories:
        filtered_df = filtered_df[filtered_df[selected_bias_categories].sum(axis=1) > 0]

    # If chart is empty, show text instead
    if filtered_df.shape[0]==0:
        data = []
        layout = {
            'xaxis': {'visible': False},
            'yaxis': {'visible': False},
            'template': 'simple_white',
            'height': 400,
            'annotations': [{
                'text': 'No articles found in the current selection.',
                'showarrow': False,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'font': {'size': 20, 'color': '#2E2C2B'}
            }]
        }
        fig = go.Figure(data=data, layout=layout)

    else:
        # Generate n-grams
        text = ' '.join(filtered_df[text_by].dropna().values)
        vectorizer = CountVectorizer(ngram_range=(ngram_value, ngram_value), stop_words='english')
        ngram_counts = vectorizer.fit_transform([text])
        ngram_freq = ngram_counts.toarray().flatten()
        ngram_names = vectorizer.get_feature_names_out()
        word_counts = dict(zip(ngram_names, ngram_freq))

        total_words = sum(word_counts.values())
        wc = WordCloud(background_color='white',
                      max_words=100,
                      width=1600,
                      height=1200,
                      scale=1.5,
                      margin=100,
                      max_font_size=100,
                      stopwords=set(STOPWORDS)
                      ).generate_from_frequencies(word_counts)

        # Get word positions and frequencies
        word_positions = wc.layout_

        # Extract positions and other data for Scatter plot
        words = []
        x = []
        y = []
        sizes = []
        hover_texts = []
        frequencies = []

        for (word, freq), font_size, position, orientation, color in word_positions:
            words.append(word)
            x.append(position[0])
            y.append(position[1])
            sizes.append(font_size)
            frequencies.append(freq)
            raw_count = word_counts[word]
            percentage = (raw_count / total_words) * 100
            hover_texts.append(f"<b>Word: </b>{word}<br>"
                              f"The word <b>'{word}'</b> appeared <b>{raw_count}</b> times across all articles in the current selection.<br>"
                              f"This accounts for <b>{percentage:.2f}%</b> of the total available word/phrases.<br>"
                              f"<br>"
                              f"Type <b>'{word}'</b> in the Word Search below to find out which articles used this word.")
#                               f"<b>Percentage of Total: x</b>{percentage:.2f}%")

        # Identify top 10 words by frequency
        top_10_indices = np.argsort(frequencies)[-10:]
        colors = ['#CFCFCF'] * len(words)
        custom_colors = [
            # '#413F42', #top 5
            # '#6B2C32',
            # '#983835',
            # '#BF4238',
            # '#C42625', #top 1

            '#413F42', # top 10

            '#6B2C32', # top 9
            '#6B2C32', # top 8

            '#983835', # top 7
            '#983835', # top 6

            '#BF4238', # top 5
            '#BF4238', # top 4

            '#C42625', #top 3
            '#C42625', #top 2
            '#C42625', #top 1
        ]

        # Apply custom colors to the top 10 words
        for i, idx in enumerate(top_10_indices):
            colors[idx] = custom_colors[i % len(custom_colors)]

        # Sort words by frequency to ensure top words appear on top
        sorted_indices = np.argsort(frequencies)
        words = [words[i] for i in sorted_indices]
        x = [x[i] for i in sorted_indices]
        y = [y[i] for i in sorted_indices]
        sizes = [sizes[i] for i in sorted_indices]
        hover_texts = [hover_texts[i] for i in sorted_indices]
        colors = [colors[i] for i in sorted_indices]

        # Create the Plotly figure with Scatter plot
        fig = go.Figure()

        # Add words as Scatter plot points
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='text',
            text=words,
            textfont=dict(size=sizes, color=colors),
            hovertext=hover_texts,
            hoverinfo='text'
        ))

        # Update the layout
        fig.update_layout(
            title='<b>What are the trending words or phrases?</b>',
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            template='simple_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='#2E2C2B',
            font_size=14,
            height=800,
            margin={'l': 150, 'r': 20, 'b': 40, 't': 40}
        )

        # Reverse the y-axis to match the word cloud orientation
        fig.update_yaxes(autorange="reversed")

    return fig


# Callback for Table 1A
@app.callback(
    [
        Output('table1a-title', 'children'),
        Output(component_id='table1a', component_property='children'),
        Output('clear-button1a', 'style'),
        Output('export-button1a', 'style'),
        Output('export-button1a', 'href')
    ],
    [
        Input('chart1a-datepickerrange', 'start_date'),
        Input('chart1a-datepickerrange', 'end_date'),
        Input('chart1a-publisher-dropdown', 'value'),
        Input('chart1a-bias-rating-dropdown', 'value'),
        Input('chart1a-bias-category-dropdown', 'value'),
        Input('chart1a-topic-dropdown', 'value'),
        Input('chart1a-color-toggle', 'value'),
        Input('top-offending-publishers-bar-chart-1a', 'clickData'),
        Input('clear-button1a', 'n_clicks')
    ]
)

def update_table1a(selected_start_date, selected_end_date, selected_publishers, selected_bias_ratings, selected_bias_categories, selected_topics, color_by, clickData, n_clicks):
    # triggered = None
    triggered = dash.callback_context.triggered
    topics = ''

    if triggered:
        id = triggered[0]['prop_id'].split('.')[0]

        if id in ['top-offending-publishers-bar-chart-1a', 'export-button1a']:
            filtered_df = df_corpus.copy()

            # Apply filters for quarters, publishers, and topics
            if (selected_start_date is not None) & (selected_end_date is not None):
                start_date = pd.to_datetime(str(selected_start_date))
                end_date = pd.to_datetime(str(selected_end_date))
                filtered_df = filtered_df[(filtered_df['date_published']>=start_date) & (filtered_df['date_published']<=end_date)]
            if selected_publishers:
                filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
            if selected_bias_ratings:
                filtered_df = filtered_df[filtered_df['bias_rating'].isin(selected_bias_ratings)]
            if selected_bias_categories:
                filtered_df = filtered_df[filtered_df[selected_bias_categories].sum(axis=1) > 0]
            if selected_topics:
                filtered_df = filtered_df[filtered_df['topic'].str.contains('|'.join(selected_topics))]
                topics = 'having any of the selected topics'

            if (clickData is not None) or (clickData is None & id=='export-button1a'):
                publisher = str(clickData['points'][0]['label'])
                filtered_df = filtered_df[filtered_df['publisher']==publisher]
                start_date = pd.to_datetime(str(selected_start_date)).strftime('%d %b %Y')
                end_date = pd.to_datetime(str(selected_end_date)).strftime('%d %b %Y')

                if color_by == 'bias_ratings':
                    # Table title
                    title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(f'Showing all articles from <b>{publisher}</b> published <b>{start_date}</b> to <b>{end_date}</b> {topics}')

                    # Apply formatting
                    filtered_df['color'] = np.select(
                        [
                            filtered_df['bias_rating'] == 2,
                            filtered_df['bias_rating'] == 1
                        ],
                        [
                            'white',
                            '#2E2C2B'
                        ],
                        '#2E2C2B'
                    )
                    filtered_df['title_label'] = "<a href='" + filtered_df['article_url'] + "' target='_blank' style='color:" + filtered_df['color'] + ";'>" + filtered_df['title'] + "</a>"
                    filtered_df['bias_rating_label'] = np.select(
                        [
                            filtered_df['bias_rating']==-1,
                            filtered_df['bias_rating']==0,
                            filtered_df['bias_rating']==1,
                            filtered_df['bias_rating']==2
                        ],
                        [
                            'Inconclusive',
                            'Not Biased',
                            'Biased',
                            'Very Biased'
                        ],
                        default='Unknown'
                    )
                    filtered_df['date_published_label_(yyyy-mm-dd)'] = filtered_df['date_published'].dt.date
                    # filtered_df['explore_further'] = "<a href='" + filtered_df['explainability_url'] + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>"
                    filtered_df['explore_further'] = "<a href='" + '' + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>"  # CHANGE THIS TO URL LATER

                    # Save to csv
                    csv_df = filtered_df[['publisher', 'title', 'article_url', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label']]
                    csv_df.columns = ['Publisher', 'Title', 'Article URL', 'Date Published (YYYY-MM-DD)', 'Topic', 'Bias Rating']
                    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_df.to_csv(index=False, encoding='utf-8'))

                    # Dash
                    filtered_df = filtered_df.sort_values('date_published_label_(yyyy-mm-dd)', ascending=False)[['publisher', 'title_label', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label', 'explore_further']]
                    table = dash_table.DataTable(
                        css=[dict(selector= "p", rule = "margin: 0; text-align: left")],
                        columns=[{'id': x, 'name': x.replace('_', ' ').title(), 'presentation': 'markdown'} if 'title' or 'explore' in x else {'id': x, 'name': x.replace('_', ' ').replace('label', '').title().replace('Or', 'or').replace('Yyyy-Mm-Dd', 'yyyy-mm-dd')} for x in filtered_df.columns],
                        markdown_options={"html": True},
                        data=filtered_df.to_dict('records'),
                        sort_action='native',
                        filter_action='native',
                        filter_options={'case': 'insensitive'},

                        page_current=0,
                        page_size=20,
                        style_table={'margin': 'auto', 'padding': '0 5px', 'overflowX': 'auto', 'overflowY': 'auto'},
                        style_header={'textAlign': 'center', 'fontWeight': 'bold'},
                        style_data={'minWidth':'120px', 'maxWidth':'120px', 'width':'120px'},
                        style_data_conditional=[
                            {
                                'if': {
                                    'filter_query': '{bias_rating_label}="Very Biased"',
                                    'column_id': ['title_label', 'bias_rating_label']
                                    },
                                'backgroundColor': '#C22625',
                                'color': 'white'
                            },
                            {
                                'if': {
                                    'filter_query': '{bias_rating_label}="Biased"',
                                    'column_id': ['title_label', 'bias_rating_label']
                                    },
                                'backgroundColor': '#eb8483',
                                'color': '#2E2C2B'
                            }
                        ],
                        style_cell={'textAlign': 'left', 'padding': '5px', 'font-family':'sans-serif', 'whiteSpace': 'normal', 'height': 'auto', 'minWidth':'180px', 'maxWidth':'180px', 'width':'180px'},
                        style_cell_conditional=[
                            {
                                'if': {
                                    'column_id': ['topic', 'title']
                                },
                                'width': '600px'
                            }
                        ]
                    )

                else:
                    title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(f'Showing biased/very biased articles from <b>{publisher}</b> published <b>{start_date}</b> to <b>{end_date}</b> {topics}')
                    filtered_df['color'] = np.select(
                        [
                            filtered_df['bias_rating'] == 2,
                            filtered_df['bias_rating'] == 1
                        ],
                        [
                            'white',
                            '#2E2C2B'
                        ],
                        '#2E2C2B'
                    )
                    filtered_df['title_label'] = "<a href='" + filtered_df['article_url'] + "' target='_blank' style='color:" + filtered_df['color'] + ";'>" + filtered_df['title'] + "</a>"
                    filtered_df['bias_rating_label'] = np.select(
                        [
                            filtered_df['bias_rating']==-1,
                            filtered_df['bias_rating']==0,
                            filtered_df['bias_rating']==1,
                            filtered_df['bias_rating']==2
                        ],
                        [
                            'Inconclusive',
                            'Not Biased',
                            'Biased',
                            'Very Biased'
                        ],
                        default='Unknown'
                    )
                    categories = ['generalisation', 'prominence', 'negative_behaviour', 'misrepresentation', 'headline_or_imagery']
                    for category in categories:
                        filtered_df[category] = np.where(filtered_df[category]==1, 'Y', 'N')
                    filtered_df['date_published_label_(yyyy-mm-dd)'] = filtered_df['date_published'].dt.date
                    # filtered_df['explore_further'] = "<a href='" + filtered_df['explainability_url'] + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>"
                    filtered_df['explore_further'] = "<a href='" + '' + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>" # CHANGE THIS TO URL LATER

                    # Save to csv
                    csv_df = filtered_df[['publisher', 'title', 'article_url', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label']+categories]
                    csv_df.columns = ['Publisher', 'Title', 'Article URL', 'Date Published (YYYY-MM-DD)', 'Topic', 'Bias Rating'] + [c.upper() for c in categories]
                    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_df.to_csv(index=False, encoding='utf-8'))

                    # Dash
                    filtered_df = filtered_df.sort_values('date_published_label_(yyyy-mm-dd)', ascending=False)[['publisher', 'title_label', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label']+categories+['explore_further']]
                    table = dash_table.DataTable(
                        css=[dict(selector="p", rule="margin:0; text-align:left")],
                        columns=[{'id': x, 'name': x.replace('_', ' ').title(), 'presentation': 'markdown'} if 'title' or 'explore' in x else {'id': x, 'name': x.replace('_', ' ').replace('label', '').title().replace('Or', 'or').replace('Yyyy-Mm-Dd', 'yyyy-mm-dd')} for x in filtered_df.columns],
                        markdown_options={"html": True},
                        data=filtered_df.to_dict('records'),
                        sort_action='native',
                        filter_action='native',
                        filter_options={'case': 'insensitive'},

                        page_current=0,
                        page_size=20,
                        style_table={'margin': 'auto', 'padding': '0 5px', 'overflowX': 'auto', 'overflowY': 'auto'},
                        style_header={'textAlign': 'center', 'fontWeight': 'bold'},
                        style_data={'minWidth':'120px', 'maxWidth':'120px', 'width':'120px'},
                        style_data_conditional=[
                            {
                                'if': {
                                    'filter_query': '{bias_rating_label}="Very Biased"',
                                    'column_id': ['title_label', 'bias_rating_label']
                                    },
                                'backgroundColor': '#C22625',
                                'color': 'white'
                            },
                            {
                                'if': {
                                    'filter_query': '{bias_rating_label}="Biased"',
                                    'column_id': ['title_label', 'bias_rating_label']
                                    },
                                'backgroundColor': '#eb8483',
                                'color': '#2E2C2B'
                            },
                            {
                                'if': {
                                    'filter_query': '{generalisation}="Y"',
                                    'column_id': 'generalisation'
                                    },
                                'backgroundColor': '#4185A0',
                                'color': 'white'
                            },
                            {
                                'if': {
                                    'filter_query': '{prominence}="Y"',
                                    'column_id': 'prominence'
                                    },
                                'backgroundColor': '#AA4D71',
                                'color': 'white'
                            },
                            {
                                'if': {
                                    'filter_query': '{negative_behaviour}="Y"',
                                    'column_id': 'negative_behaviour'
                                    },
                                'backgroundColor': '#B85C3B',
                                'color': 'white'
                            },
                            {
                                'if': {
                                    'filter_query': '{misrepresentation}="Y"',
                                    'column_id': 'misrepresentation'
                                    },
                                'backgroundColor': '#C5BE71',
                                'color': '#2E2C2B'
                            },
                            {
                                'if': {
                                    'filter_query': '{headline_or_imagery}="Y"',
                                    'column_id': 'headline_or_imagery'
                                    },
                                'backgroundColor': '#7658A0',
                                'color': 'white'
                            }
                        ],
                        style_cell={'textAlign': 'left', 'padding': '5px', 'font-family':'sans-serif', 'whiteSpace': 'normal', 'height': 'auto'},
                        style_cell_conditional=[
                            {
                                'if': {'column_id': ['topic', 'title']},
                                'width': '600px'
                            }
                        ]
                    )

            if id == 'export-button1a':
                return [title], table, {'fontSize':14, 'display': 'block'}, {'fontSize':14, 'display': 'block', 'margin-left': '10px'}, csv_string

            return [title], table, {'fontSize':14, 'display': 'block'}, {'fontSize':14, 'display': 'block', 'margin-left': '10px'}, csv_string

        elif id in ['chart1a-datepickerrange', 'chart1a-topic-dropdown', 'chart1a-publisher-dropdown', 'chart1a-bias-rating-dropdown', 'chart1a-bias-category-dropdown', 'chart1a-color-toggle', 'clear-button1a']:
            return [], None, {'display': 'none'}, {'display': 'none'}, ''

    else:
        return [], None, {'display': 'none'}, {'display': 'none'}, ''
    

# Callback for Table 1B
@app.callback(
    [
        Output('table1b-title', 'children'),
        Output(component_id='table1b', component_property='children'),
        Output('clear-button1b', 'style'),
        Output('export-button1b', 'style'),
        Output('export-button1b', 'href')
    ],
    [
        Input('chart1b-datepickerrange', 'start_date'),
        Input('chart1b-datepickerrange', 'end_date'),
        Input('chart1b-publisher-dropdown', 'value'),
        Input('chart1b-bias-rating-dropdown', 'value'),
        Input('chart1b-bias-category-dropdown', 'value'),
        Input('chart1b-topic-dropdown', 'value'),
        Input('chart1b-color-toggle', 'value'),
        Input('top-offending-publishers-bar-chart-1b', 'clickData'),
        Input('clear-button1b', 'n_clicks')
    ]
)

def update_table1b(selected_start_date, selected_end_date, selected_publishers, selected_bias_ratings, selected_bias_categories, selected_topics, color_by, clickData, n_clicks):
    # triggered = None
    triggered = dash.callback_context.triggered
    topics = ''

    if triggered:
        id = triggered[0]['prop_id'].split('.')[0]

        if id in ['top-offending-publishers-bar-chart-1b', 'export-button1b']:
            filtered_df = df_corpus.copy()

            # Apply filters for quarters, publishers, and topics
            if (selected_start_date is not None) & (selected_end_date is not None):
                start_date = pd.to_datetime(str(selected_start_date))
                end_date = pd.to_datetime(str(selected_end_date))
                filtered_df = filtered_df[(filtered_df['date_published']>=start_date) & (filtered_df['date_published']<=end_date)]
            if selected_publishers:
                filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
            if selected_bias_ratings:
                filtered_df = filtered_df[filtered_df['bias_rating'].isin(selected_bias_ratings)]
            if selected_bias_categories:
                filtered_df = filtered_df[filtered_df[selected_bias_categories].sum(axis=1) > 0]
            if selected_topics:
                filtered_df = filtered_df[filtered_df['topic'].str.contains('|'.join(selected_topics))]
                topics = 'having any of the selected topics'

            if (clickData is not None) or (clickData is None & id=='export-button1b'):
                publisher = str(clickData['points'][0]['label'])
                filtered_df = filtered_df[filtered_df['publisher']==publisher]
                start_date = pd.to_datetime(str(selected_start_date)).strftime('%d %b %Y')
                end_date = pd.to_datetime(str(selected_end_date)).strftime('%d %b %Y')

                if color_by == 'bias_ratings':
                    # Table title
                    title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(f'Showing all articles from <b>{publisher}</b> published <b>{start_date}</b> to <b>{end_date}</b> {topics}')

                    # Apply formatting
                    filtered_df['color'] = np.select(
                        [
                            filtered_df['bias_rating'] == 2,
                            filtered_df['bias_rating'] == 1
                        ],
                        [
                            'white',
                            '#2E2C2B'
                        ],
                        '#2E2C2B'
                    )
                    filtered_df['title_label'] = "<a href='" + filtered_df['article_url'] + "' target='_blank' style='color:" + filtered_df['color'] + ";'>" + filtered_df['title'] + "</a>"
                    filtered_df['bias_rating_label'] = np.select(
                        [
                            filtered_df['bias_rating']==-1,
                            filtered_df['bias_rating']==0,
                            filtered_df['bias_rating']==1,
                            filtered_df['bias_rating']==2
                        ],
                        [
                            'Inconclusive',
                            'Not Biased',
                            'Biased',
                            'Very Biased'
                        ],
                        default='Unknown'
                    )
                    filtered_df['date_published_label_(yyyy-mm-dd)'] = filtered_df['date_published'].dt.date
                    # filtered_df['explore_further'] = "<a href='" + filtered_df['explainability_url'] + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>"
                    filtered_df['explore_further'] = "<a href='" + '' + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>"  # CHANGE THIS TO URL LATER

                    # Save to csv
                    csv_df = filtered_df[['publisher', 'title', 'article_url', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label']]
                    csv_df.columns = ['Publisher', 'Title', 'Article URL', 'Date Published (YYYY-MM-DD)', 'Topic', 'Bias Rating']
                    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_df.to_csv(index=False, encoding='utf-8'))

                    # Dash
                    filtered_df = filtered_df.sort_values('date_published_label_(yyyy-mm-dd)', ascending=False)[['publisher', 'title_label', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label', 'explore_further']]
                    table = dash_table.DataTable(
                        css=[dict(selector= "p", rule = "margin: 0; text-align: left")],
                        columns=[{'id': x, 'name': x.replace('_', ' ').title(), 'presentation': 'markdown'} if 'title' or 'explore' in x else {'id': x, 'name': x.replace('_', ' ').replace('label', '').title().replace('Or', 'or').replace('Yyyy-Mm-Dd', 'yyyy-mm-dd')} for x in filtered_df.columns],
                        markdown_options={"html": True},
                        data=filtered_df.to_dict('records'),
                        sort_action='native',
                        filter_action='native',
                        filter_options={'case': 'insensitive'},

                        page_current=0,
                        page_size=20,
                        style_table={'margin': 'auto', 'padding': '0 5px', 'overflowX': 'auto', 'overflowY': 'auto'},
                        style_header={'textAlign': 'center', 'fontWeight': 'bold'},
                        style_data={'minWidth':'120px', 'maxWidth':'120px', 'width':'120px'},
                        style_data_conditional=[
                            {
                                'if': {
                                    'filter_query': '{bias_rating_label}="Very Biased"',
                                    'column_id': ['title_label', 'bias_rating_label']
                                    },
                                'backgroundColor': '#C22625',
                                'color': 'white'
                            },
                            {
                                'if': {
                                    'filter_query': '{bias_rating_label}="Biased"',
                                    'column_id': ['title_label', 'bias_rating_label']
                                    },
                                'backgroundColor': '#eb8483',
                                'color': '#2E2C2B'
                            }
                        ],
                        style_cell={'textAlign': 'left', 'padding': '5px', 'font-family':'sans-serif', 'whiteSpace': 'normal', 'height': 'auto', 'minWidth':'180px', 'maxWidth':'180px', 'width':'180px'},
                        style_cell_conditional=[
                            {
                                'if': {
                                    'column_id': ['topic', 'title']
                                },
                                'width': '600px'
                            }
                        ]
                    )

                else:
                    title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(f'Showing biased/very biased articles from <b>{publisher}</b> published <b>{start_date}</b> to <b>{end_date}</b> {topics}')
                    filtered_df['color'] = np.select(
                        [
                            filtered_df['bias_rating'] == 2,
                            filtered_df['bias_rating'] == 1
                        ],
                        [
                            'white',
                            '#2E2C2B'
                        ],
                        '#2E2C2B'
                    )
                    filtered_df['title_label'] = "<a href='" + filtered_df['article_url'] + "' target='_blank' style='color:" + filtered_df['color'] + ";'>" + filtered_df['title'] + "</a>"
                    filtered_df['bias_rating_label'] = np.select(
                        [
                            filtered_df['bias_rating']==-1,
                            filtered_df['bias_rating']==0,
                            filtered_df['bias_rating']==1,
                            filtered_df['bias_rating']==2
                        ],
                        [
                            'Inconclusive',
                            'Not Biased',
                            'Biased',
                            'Very Biased'
                        ],
                        default='Unknown'
                    )
                    categories = ['generalisation', 'prominence', 'negative_behaviour', 'misrepresentation', 'headline_or_imagery']
                    for category in categories:
                        filtered_df[category] = np.where(filtered_df[category]==1, 'Y', 'N')
                    filtered_df['date_published_label_(yyyy-mm-dd)'] = filtered_df['date_published'].dt.date
                    # filtered_df['explore_further'] = "<a href='" + filtered_df['explainability_url'] + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>"
                    filtered_df['explore_further'] = "<a href='" + '' + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>" # CHANGE THIS TO URL LATER

                    # Save to csv
                    csv_df = filtered_df[['publisher', 'title', 'article_url', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label']+categories]
                    csv_df.columns = ['Publisher', 'Title', 'Article URL', 'Date Published (YYYY-MM-DD)', 'Topic', 'Bias Rating'] + [c.upper() for c in categories]
                    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_df.to_csv(index=False, encoding='utf-8'))

                    # Dash
                    filtered_df = filtered_df.sort_values('date_published_label_(yyyy-mm-dd)', ascending=False)[['publisher', 'title_label', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label']+categories+['explore_further']]
                    table = dash_table.DataTable(
                        css=[dict(selector="p", rule="margin:0; text-align:left")],
                        columns=[{'id': x, 'name': x.replace('_', ' ').title(), 'presentation': 'markdown'} if 'title' or 'explore' in x else {'id': x, 'name': x.replace('_', ' ').replace('label', '').title().replace('Or', 'or').replace('Yyyy-Mm-Dd', 'yyyy-mm-dd')} for x in filtered_df.columns],
                        markdown_options={"html": True},
                        data=filtered_df.to_dict('records'),
                        sort_action='native',
                        filter_action='native',
                        filter_options={'case': 'insensitive'},

                        page_current=0,
                        page_size=20,
                        style_table={'margin': 'auto', 'padding': '0 5px', 'overflowX': 'auto', 'overflowY': 'auto'},
                        style_header={'textAlign': 'center', 'fontWeight': 'bold'},
                        style_data={'minWidth':'120px', 'maxWidth':'120px', 'width':'120px'},
                        style_data_conditional=[
                            {
                                'if': {
                                    'filter_query': '{bias_rating_label}="Very Biased"',
                                    'column_id': ['title_label', 'bias_rating_label']
                                    },
                                'backgroundColor': '#C22625',
                                'color': 'white'
                            },
                            {
                                'if': {
                                    'filter_query': '{bias_rating_label}="Biased"',
                                    'column_id': ['title_label', 'bias_rating_label']
                                    },
                                'backgroundColor': '#eb8483',
                                'color': '#2E2C2B'
                            },
                            {
                                'if': {
                                    'filter_query': '{generalisation}="Y"',
                                    'column_id': 'generalisation'
                                    },
                                'backgroundColor': '#4185A0',
                                'color': 'white'
                            },
                            {
                                'if': {
                                    'filter_query': '{prominence}="Y"',
                                    'column_id': 'prominence'
                                    },
                                'backgroundColor': '#AA4D71',
                                'color': 'white'
                            },
                            {
                                'if': {
                                    'filter_query': '{negative_behaviour}="Y"',
                                    'column_id': 'negative_behaviour'
                                    },
                                'backgroundColor': '#B85C3B',
                                'color': 'white'
                            },
                            {
                                'if': {
                                    'filter_query': '{misrepresentation}="Y"',
                                    'column_id': 'misrepresentation'
                                    },
                                'backgroundColor': '#C5BE71',
                                'color': '#2E2C2B'
                            },
                            {
                                'if': {
                                    'filter_query': '{headline_or_imagery}="Y"',
                                    'column_id': 'headline_or_imagery'
                                    },
                                'backgroundColor': '#7658A0',
                                'color': 'white'
                            }
                        ],
                        style_cell={'textAlign': 'left', 'padding': '5px', 'font-family':'sans-serif', 'whiteSpace': 'normal', 'height': 'auto'},
                        style_cell_conditional=[
                            {
                                'if': {'column_id': ['topic', 'title']},
                                'width': '600px'
                            }
                        ]
                    )

            if id == 'export-button1b':
                return [title], table, {'fontSize':14, 'display': 'block'}, {'fontSize':14, 'display': 'block', 'margin-left': '10px'}, csv_string

            return [title], table, {'fontSize':14, 'display': 'block'}, {'fontSize':14, 'display': 'block', 'margin-left': '10px'}, csv_string

        elif id in ['chart1b-datepickerrange', 'chart1b-topic-dropdown', 'chart1b-publisher-dropdown', 'chart1b-bias-rating-dropdown', 'chart1b-bias-category-dropdown', 'chart1b-color-toggle', 'clear-button1b']:
            return [], None, {'display': 'none'}, {'display': 'none'}, ''

    else:
        return [], None, {'display': 'none'}, {'display': 'none'}, ''


# Callback for Table 2A
@app.callback(
    [
        Output('table2a-title', 'children'),
        Output(component_id='table2a', component_property='children'),
        Output('clear-button2a', 'style'),
        Output('export-button2a', 'style'),
        Output('export-button2a', 'href')
    ],
    [
        Input('chart2a-datepickerrange', 'start_date'),
        Input('chart2a-datepickerrange', 'end_date'),
        Input('chart2a-publisher-dropdown', 'value'),
        Input('chart2a-bias-rating-dropdown', 'value'),
        Input('chart2a-bias-category-dropdown', 'value'),
        Input('top-topics-bar-chart-2a', 'clickData'),
        Input('clear-button2a', 'n_clicks')
    ]
)

def update_table2a(selected_start_date, selected_end_date, selected_publishers, selected_bias_ratings, selected_bias_categories, clickData, n_clicks):
    triggered = dash.callback_context.triggered

    if triggered:
        id = triggered[0]['prop_id'].split('.')[0]

        if id in ['top-topics-bar-chart-2a', 'export-button2a']:
            filtered_df = df_corpus.copy()

            # Apply filters for quarters, publishers, and topics
            if (selected_start_date is not None) & (selected_end_date is not None):
                start_date = pd.to_datetime(str(selected_start_date))
                end_date = pd.to_datetime(str(selected_end_date))
                filtered_df = filtered_df[(filtered_df['date_published']>=start_date) & (filtered_df['date_published']<=end_date)]
            if selected_publishers:
                filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
            if selected_bias_ratings:
                filtered_df = filtered_df[filtered_df[selected_bias_categories].sum(axis=1) > 0]
            if selected_bias_categories:
                filtered_df = filtered_df[filtered_df['bias_category'].isin(selected_bias_categories)]

            if (clickData is not None) or (clickData is None & id=='export-button2a'):
                topic = str(clickData['points'][0]['label'])

                # Table title
                title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(f'Showing all articles about <b>{topic}</b>')

                # Apply formatting
                filtered_df = filtered_df[filtered_df['topic'].str.contains('|'.join([topic]))]
                filtered_df['color'] = np.select(
                    [
                        filtered_df['bias_rating'] == 2,
                        filtered_df['bias_rating'] == 1
                    ],
                    [
                        'white',
                        '#2E2C2B'
                    ],
                    '#2E2C2B'
                )
                filtered_df['title_label'] = "<a href='" + filtered_df['article_url'] + "' target='_blank' style='color:" + filtered_df['color'] + ";'>" + filtered_df['title'] + "</a>"
                filtered_df['bias_rating_label'] = np.select(
                    [
                        filtered_df['bias_rating']==-1,
                        filtered_df['bias_rating']==0,
                        filtered_df['bias_rating']==1,
                        filtered_df['bias_rating']==2
                    ],
                    [
                        'Inconclusive',
                        'Not Biased',
                        'Biased',
                        'Very Biased'
                    ],
                    default='Unknown'
                )
                categories = ['generalisation', 'prominence', 'negative_behaviour', 'misrepresentation', 'headline_or_imagery']
                for category in categories:
                    filtered_df[category] = np.where(filtered_df[category]==1, 'Y', 'N')
                filtered_df['date_published_label_(yyyy-mm-dd)'] = filtered_df['date_published'].dt.date
                # filtered_df['explore_further'] = "<a href='" + filtered_df['explainability_url'] + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>"
                filtered_df['explore_further'] = "<a href='" + '' + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>" # CHANGE THIS TO URL LATER

                # Save to csv
                csv_df = filtered_df[['publisher', 'title', 'article_url', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label']+categories]
                csv_df.columns = ['Publisher', 'Title', 'Article URL', 'Date Published (YYYY-MM-DD)', 'Topic', 'Bias Rating'] + [c.upper() for c in categories]
                csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_df.to_csv(index=False, encoding='utf-8'))

                # Dash
                filtered_df = filtered_df.sort_values('date_published_label_(yyyy-mm-dd)', ascending=False)[['publisher', 'title_label', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label']+categories+['explore_further']]
                table = dash_table.DataTable(
                    css=[dict(selector="p", rule="margin:0; text-align:left")],
                    columns=[{'id': x, 'name': x.replace('_', ' ').title(), 'presentation': 'markdown'} if 'title' or 'explore' in x else {'id': x, 'name': x.replace('_', ' ').replace('label', '').title().replace('Or', 'or').replace('Yyyy-Mm-Dd', 'yyyy-mm-dd')} for x in filtered_df.columns],
                    markdown_options={"html": True},
                    data=filtered_df.to_dict('records'),
                    sort_action='native',
                    filter_action='native',
                    filter_options={'case': 'insensitive'},

                    page_current=0,
                    page_size=20,
                    style_table={'margin': 'auto', 'padding': '0 5px', 'overflowX': 'auto', 'overflowY': 'auto'},
                    style_header={'textAlign': 'center', 'fontWeight': 'bold'},
                    style_data={'minWidth':'120px', 'maxWidth':'120px', 'width':'120px'},
                    style_data_conditional=[
                        {
                            'if': {
                                'filter_query': '{bias_rating_label}="Very Biased"',
                                'column_id': ['title_label', 'bias_rating_label']
                                },
                            'backgroundColor': '#C22625',
                            'color': 'white'
                        },
                        {
                            'if': {
                                'filter_query': '{bias_rating_label}="Biased"',
                                'column_id': ['title_label', 'bias_rating_label']
                                },
                            'backgroundColor': '#eb8483',
                            'color': '#2E2C2B'
                        }
                    ],
                    style_cell={'textAlign': 'left', 'padding': '5px', 'font-family':'sans-serif', 'whiteSpace': 'normal', 'height': 'auto'},
                        style_cell_conditional=[
                            {
                                'if': {'column_id': ['topic', 'title']},
                                'width': '600px'
                            }
                        ]
                )

            if id == 'export-button2a':
                return [title], table, {'fontSize':14, 'display': 'block'}, {'fontSize':14, 'display': 'block', 'margin-left': '10px'}, csv_string

            return [title], table, {'fontSize':14, 'display': 'block'}, {'fontSize':14, 'display': 'block', 'margin-left': '10px'}, csv_string

        elif id in ['chart2a-datepickerrange', 'chart2a-publisher-dropdown', 'chart2a-bias-rating-dropdown', 'chart2a-bias-category-dropdown', 'chart2a-color-toggle', 'clear-button2a']:
            return [], None, {'display': 'none'}, {'display': 'none'}, ''

    else:
        return [], None, {'display': 'none'}, {'display': 'none'}, ''


# Callback for Table 2B
@app.callback(
    [
        Output('table2b-title', 'children'),
        Output(component_id='table2b', component_property='children'),
        Output('clear-button2b', 'style'),
        Output('export-button2b', 'style'),
        Output('export-button2b', 'href')
    ],
    [
        Input('chart2b-datepickerrange', 'start_date'),
        Input('chart2b-datepickerrange', 'end_date'),
        Input('chart2b-publisher-dropdown', 'value'),
        Input('chart2b-bias-rating-dropdown', 'value'),
        Input('chart2b-bias-category-dropdown', 'value'),
        Input('top-topics-bar-chart-2b', 'clickData'),
        Input('clear-button2b', 'n_clicks')
    ]
)

def update_table2b(selected_start_date, selected_end_date, selected_publishers, selected_bias_ratings, selected_bias_categories, clickData, n_clicks):
    triggered = dash.callback_context.triggered

    if triggered:
        id = triggered[0]['prop_id'].split('.')[0]

        if id in ['top-topics-bar-chart-2b', 'export-button2b']:
            filtered_df = df_corpus.copy()

            # Apply filters for quarters, publishers, and topics
            if (selected_start_date is not None) & (selected_end_date is not None):
                start_date = pd.to_datetime(str(selected_start_date))
                end_date = pd.to_datetime(str(selected_end_date))
                filtered_df = filtered_df[(filtered_df['date_published']>=start_date) & (filtered_df['date_published']<=end_date)]
            if selected_publishers:
                filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
            if selected_bias_ratings:
                filtered_df = filtered_df[filtered_df[selected_bias_categories].sum(axis=1) > 0]
            if selected_bias_categories:
                filtered_df = filtered_df[filtered_df['bias_category'].isin(selected_bias_categories)]

            if (clickData is not None) or (clickData is None & id=='export-button2b'):
                topic = str(clickData['points'][0]['label'])

                # Table title
                title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(f'Showing all articles about <b>{topic}</b>')

                # Apply formatting
                filtered_df = filtered_df[filtered_df['topic'].str.contains('|'.join([topic]))]
                filtered_df['color'] = np.select(
                    [
                        filtered_df['bias_rating'] == 2,
                        filtered_df['bias_rating'] == 1
                    ],
                    [
                        'white',
                        '#2E2C2B'
                    ],
                    '#2E2C2B'
                )
                filtered_df['title_label'] = "<a href='" + filtered_df['article_url'] + "' target='_blank' style='color:" + filtered_df['color'] + ";'>" + filtered_df['title'] + "</a>"
                filtered_df['bias_rating_label'] = np.select(
                    [
                        filtered_df['bias_rating']==-1,
                        filtered_df['bias_rating']==0,
                        filtered_df['bias_rating']==1,
                        filtered_df['bias_rating']==2
                    ],
                    [
                        'Inconclusive',
                        'Not Biased',
                        'Biased',
                        'Very Biased'
                    ],
                    default='Unknown'
                )
                categories = ['generalisation', 'prominence', 'negative_behaviour', 'misrepresentation', 'headline_or_imagery']
                for category in categories:
                    filtered_df[category] = np.where(filtered_df[category]==1, 'Y', 'N')
                filtered_df['date_published_label_(yyyy-mm-dd)'] = filtered_df['date_published'].dt.date
                # filtered_df['explore_further'] = "<a href='" + filtered_df['explainability_url'] + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>"
                filtered_df['explore_further'] = "<a href='" + '' + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>" # CHANGE THIS TO URL LATER

                # Save to csv
                csv_df = filtered_df[['publisher', 'title', 'article_url', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label']+categories]
                csv_df.columns = ['Publisher', 'Title', 'Article URL', 'Date Published (YYYY-MM-DD)', 'Topic', 'Bias Rating'] + [c.upper() for c in categories]
                csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_df.to_csv(index=False, encoding='utf-8'))

                # Dash
                filtered_df = filtered_df.sort_values('date_published_label_(yyyy-mm-dd)', ascending=False)[['publisher', 'title_label', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label']+categories+['explore_further']]
                table = dash_table.DataTable(
                    css=[dict(selector="p", rule="margin:0; text-align:left")],
                    columns=[{'id': x, 'name': x.replace('_', ' ').title(), 'presentation': 'markdown'} if 'title' or 'explore' in x else {'id': x, 'name': x.replace('_', ' ').replace('label', '').title().replace('Or', 'or').replace('Yyyy-Mm-Dd', 'yyyy-mm-dd')} for x in filtered_df.columns],
                    markdown_options={"html": True},
                    data=filtered_df.to_dict('records'),
                    sort_action='native',
                    filter_action='native',
                    filter_options={'case': 'insensitive'},

                    page_current=0,
                    page_size=20,
                    style_table={'margin': 'auto', 'padding': '0 5px', 'overflowX': 'auto', 'overflowY': 'auto'},
                    style_header={'textAlign': 'center', 'fontWeight': 'bold'},
                    style_data={'minWidth':'120px', 'maxWidth':'120px', 'width':'120px'},
                    style_data_conditional=[
                        {
                            'if': {
                                'filter_query': '{bias_rating_label}="Very Biased"',
                                'column_id': ['title_label', 'bias_rating_label']
                                },
                            'backgroundColor': '#C22625',
                            'color': 'white'
                        },
                        {
                            'if': {
                                'filter_query': '{bias_rating_label}="Biased"',
                                'column_id': ['title_label', 'bias_rating_label']
                                },
                            'backgroundColor': '#eb8483',
                            'color': '#2E2C2B'
                        }
                    ],
                    style_cell={'textAlign': 'left', 'padding': '5px', 'font-family':'sans-serif', 'whiteSpace': 'normal', 'height': 'auto'},
                        style_cell_conditional=[
                            {
                                'if': {'column_id': ['topic', 'title']},
                                'width': '600px'
                            }
                        ]
                )

            if id == 'export-button2b':
                return [title], table, {'fontSize':14, 'display': 'block'}, {'fontSize':14, 'display': 'block', 'margin-left': '10px'}, csv_string

            return [title], table, {'fontSize':14, 'display': 'block'}, {'fontSize':14, 'display': 'block', 'margin-left': '10px'}, csv_string

        elif id in ['chart2b-datepickerrange', 'chart2b-publisher-dropdown', 'chart2b-bias-rating-dropdown', 'chart2b-bias-category-dropdown', 'chart2b-color-toggle', 'clear-button2b']:
            return [], None, {'display': 'none'}, {'display': 'none'}, ''

    else:
        return [], None, {'display': 'none'}, {'display': 'none'}, ''


# Callback for Table 3A
@app.callback(
    [
        Output('table3a-title', 'children'),
        Output(component_id='table3a', component_property='children'),
        Output('clear-button3a', 'style'),
        Output('export-button3a', 'style'),
        Output('export-button3a', 'href')
    ],
    [
        Input('chart3a-datepickerrange', 'start_date'),
        Input('chart3a-datepickerrange', 'end_date'),
        Input('chart3a-publisher-dropdown', 'value'),
        Input('chart3a-bias-category-dropdown', 'value'),
        Input('chart3a-topic-dropdown', 'value'),
        Input('top-offending-articles-bar-chart-3a', 'clickData'),
        Input('clear-button3a', 'n_clicks')
    ]
)

def update_table3a(selected_start_date, selected_end_date, selected_publishers, selected_bias_categories, selected_topics, clickData, n_clicks):
    triggered = dash.callback_context.triggered
    topics = ''

    if triggered:
        id = triggered[0]['prop_id'].split('.')[0]

        if id in ['top-offending-articles-bar-chart-3a', 'export-button3a']:
            filtered_df = df_corpus.copy()

            # Apply filters for quarters, publishers, and topics
            if (selected_start_date is not None) & (selected_end_date is not None):
                start_date = pd.to_datetime(str(selected_start_date))
                end_date = pd.to_datetime(str(selected_end_date))
                filtered_df = filtered_df[(filtered_df['date_published']>=start_date) & (filtered_df['date_published']<=end_date)]
            if selected_publishers:
                filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
            if selected_bias_categories:
                filtered_df = filtered_df[filtered_df['bias_category'].isin(selected_bias_categories)]
            if selected_topics:
                filtered_df = filtered_df[filtered_df['topic'].str.contains('|'.join(selected_topics))]
                topics = 'having any of the selected topics'

            label_map = {
                    -1: 'Inconclusive',
                    0: 'Not Biased',
                    1: 'Biased',
                    2: 'Very Biased'
                }
            filtered_df['bias_rating_label'] = filtered_df['bias_rating'].map(label_map)
            filtered_df['bias_rating_label'] = pd.Categorical(filtered_df['bias_rating_label'], categories=['Inconclusive', 'Not Biased', 'Biased', 'Very Biased'], ordered=True)

            if (clickData is not None) or (clickData is None & id=='export-button3a'):
                bias = str(clickData['points'][0]['label'])

                # Table title
                title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(f'Showing all articles that were rated <b>{bias}</b> by the model.')

                # Apply formatting
                filtered_df = filtered_df[filtered_df['bias_rating_label']==bias]
                filtered_df['color'] = np.select(
                    [
                        filtered_df['bias_rating'] == 2,
                        filtered_df['bias_rating'] == 1
                    ],
                    [
                        'white',
                        '#2E2C2B'
                    ],
                    '#2E2C2B'
                )
                filtered_df['title_label'] = "<a href='" + filtered_df['article_url'] + "' target='_blank' style='color:" + filtered_df['color'] + ";'>" + filtered_df['title'] + "</a>"

                categories = ['generalisation', 'prominence', 'negative_behaviour', 'misrepresentation', 'headline_or_imagery']
                for category in categories:
                    filtered_df[category] = np.where(filtered_df[category]==1, 'Y', 'N')
                filtered_df['date_published_label_(yyyy-mm-dd)'] = filtered_df['date_published'].dt.date
                # filtered_df['explore_further'] = "<a href='" + filtered_df['explainability_url'] + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>"
                filtered_df['explore_further'] = "<a href='" + '' + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>" # CHANGE THIS TO URL LATER

                # Save to csv
                csv_df = filtered_df[['publisher', 'title', 'article_url', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label']+categories]
                csv_df.columns = ['Publisher', 'Title', 'Article URL', 'Date Published (YYYY-MM-DD)', 'Topic', 'Bias Rating'] + [c.upper() for c in categories]
                csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_df.to_csv(index=False, encoding='utf-8'))

                # Dash
                filtered_df = filtered_df.sort_values('date_published_label_(yyyy-mm-dd)', ascending=False)[['publisher', 'title_label', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label']+categories+['explore_further']]
                table = dash_table.DataTable(
                    css=[dict(selector="p", rule="margin:0; text-align:left")],
                    columns=[{'id': x, 'name': x.replace('_', ' ').title(), 'presentation': 'markdown'} if 'title' or 'explore' in x else {'id': x, 'name': x.replace('_', ' ').replace('label', '').title().replace('Or', 'or').replace('Yyyy-Mm-Dd', 'yyyy-mm-dd')} for x in filtered_df.columns],
                    markdown_options={"html": True},
                    data=filtered_df.to_dict('records'),
                    sort_action='native',
                    filter_action='native',
                    filter_options={'case': 'insensitive'},

                    page_current=0,
                    page_size=20,
                    style_table={'margin': 'auto', 'padding': '0 5px', 'overflowX': 'auto', 'overflowY': 'auto'},
                    style_header={'textAlign': 'center', 'fontWeight': 'bold'},
                    style_data={'minWidth':'120px', 'maxWidth':'120px', 'width':'120px'},
                    style_data_conditional=[
                        {
                            'if': {
                                'filter_query': '{bias_rating_label}="Very Biased"',
                                'column_id': ['title_label', 'bias_rating_label']
                                },
                            'backgroundColor': '#C22625',
                            'color': 'white'
                        },
                        {
                            'if': {
                                'filter_query': '{bias_rating_label}="Biased"',
                                'column_id': ['title_label', 'bias_rating_label']
                                },
                            'backgroundColor': '#eb8483',
                            'color': '#2E2C2B'
                        }
                    ],
                    style_cell={'textAlign': 'left', 'padding': '5px', 'font-family':'sans-serif', 'whiteSpace': 'normal', 'height': 'auto'},
                        style_cell_conditional=[
                            {
                                'if': {'column_id': ['topic', 'title_label']},
                                'width': '600px'
                            }
                        ]
                )

            if id == 'export-button3a':
                return [title], table, {'fontSize':14, 'display': 'block'}, {'fontSize':14, 'display': 'block', 'margin-left': '10px'}, csv_string

            return [title], table, {'fontSize':14, 'display': 'block'}, {'fontSize':14, 'display': 'block', 'margin-left': '10px'}, csv_string

        elif id in ['chart3a-datepickerrange', 'chart3a-publisher-dropdown', 'chart3a-bias-category-dropdown', 'chart3a-topic-dropdown', 'clear-button3a']:
            return [], None, {'display': 'none'}, {'display': 'none'}, ''

    else:
        return [], None, {'display': 'none'}, {'display': 'none'}, ''
    

# Callback for Table 3B
@app.callback(
    [
        Output('table3b-title', 'children'),
        Output(component_id='table3b', component_property='children'),
        Output('clear-button3b', 'style'),
        Output('export-button3b', 'style'),
        Output('export-button3b', 'href')
    ],
    [
        Input('chart3b-datepickerrange', 'start_date'),
        Input('chart3b-datepickerrange', 'end_date'),
        Input('chart3b-publisher-dropdown', 'value'),
        Input('chart3b-bias-category-dropdown', 'value'),
        Input('chart3b-topic-dropdown', 'value'),
        Input('top-offending-articles-bar-chart-3b', 'clickData'),
        Input('clear-button3b', 'n_clicks')
    ]
)

def update_table3b(selected_start_date, selected_end_date, selected_publishers, selected_bias_categories, selected_topics, clickData, n_clicks):
    triggered = dash.callback_context.triggered
    topics = ''

    if triggered:
        id = triggered[0]['prop_id'].split('.')[0]

        if id in ['top-offending-articles-bar-chart-3b', 'export-button3bx']:
            filtered_df = df_corpus.copy()

            # Apply filters for quarters, publishers, and topics
            if (selected_start_date is not None) & (selected_end_date is not None):
                start_date = pd.to_datetime(str(selected_start_date))
                end_date = pd.to_datetime(str(selected_end_date))
                filtered_df = filtered_df[(filtered_df['date_published']>=start_date) & (filtered_df['date_published']<=end_date)]
            if selected_publishers:
                filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
            if selected_bias_categories:
                filtered_df = filtered_df[filtered_df['bias_category'].isin(selected_bias_categories)]
            if selected_topics:
                filtered_df = filtered_df[filtered_df['topic'].str.contains('|'.join(selected_topics))]
                topics = 'having any of the selected topics'

            label_map = {
                    -1: 'Inconclusive',
                    0: 'Not Biased',
                    1: 'Biased',
                    2: 'Very Biased'
                }
            filtered_df['bias_rating_label'] = filtered_df['bias_rating'].map(label_map)
            filtered_df['bias_rating_label'] = pd.Categorical(filtered_df['bias_rating_label'], categories=['Inconclusive', 'Not Biased', 'Biased', 'Very Biased'], ordered=True)

            if (clickData is not None) or (clickData is None & id=='export-button3b'):
                bias = str(clickData['points'][0]['label'])

                # Table title
                title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(f'Showing all articles that were rated <b>{bias}</b> by the model.')

                # Apply formatting
                filtered_df = filtered_df[filtered_df['bias_rating_label']==bias]
                filtered_df['color'] = np.select(
                    [
                        filtered_df['bias_rating'] == 2,
                        filtered_df['bias_rating'] == 1
                    ],
                    [
                        'white',
                        '#2E2C2B'
                    ],
                    '#2E2C2B'
                )
                filtered_df['title_label'] = "<a href='" + filtered_df['article_url'] + "' target='_blank' style='color:" + filtered_df['color'] + ";'>" + filtered_df['title'] + "</a>"

                categories = ['generalisation', 'prominence', 'negative_behaviour', 'misrepresentation', 'headline_or_imagery']
                for category in categories:
                    filtered_df[category] = np.where(filtered_df[category]==1, 'Y', 'N')
                filtered_df['date_published_label_(yyyy-mm-dd)'] = filtered_df['date_published'].dt.date
                # filtered_df['explore_further'] = "<a href='" + filtered_df['explainability_url'] + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>"
                filtered_df['explore_further'] = "<a href='" + '' + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>" # CHANGE THIS TO URL LATER

                # Save to csv
                csv_df = filtered_df[['publisher', 'title', 'article_url', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label']+categories]
                csv_df.columns = ['Publisher', 'Title', 'Article URL', 'Date Published (YYYY-MM-DD)', 'Topic', 'Bias Rating'] + [c.upper() for c in categories]
                csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_df.to_csv(index=False, encoding='utf-8'))

                # Dash
                filtered_df = filtered_df.sort_values('date_published_label_(yyyy-mm-dd)', ascending=False)[['publisher', 'title_label', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label']+categories+['explore_further']]
                table = dash_table.DataTable(
                    css=[dict(selector="p", rule="margin:0; text-align:left")],
                    columns=[{'id': x, 'name': x.replace('_', ' ').title(), 'presentation': 'markdown'} if 'title' or 'explore' in x else {'id': x, 'name': x.replace('_', ' ').replace('label', '').title().replace('Or', 'or').replace('Yyyy-Mm-Dd', 'yyyy-mm-dd')} for x in filtered_df.columns],
                    markdown_options={"html": True},
                    data=filtered_df.to_dict('records'),
                    sort_action='native',
                    filter_action='native',
                    filter_options={'case': 'insensitive'},

                    page_current=0,
                    page_size=20,
                    style_table={'margin': 'auto', 'padding': '0 5px', 'overflowX': 'auto', 'overflowY': 'auto'},
                    style_header={'textAlign': 'center', 'fontWeight': 'bold'},
                    style_data={'minWidth':'120px', 'maxWidth':'120px', 'width':'120px'},
                    style_data_conditional=[
                        {
                            'if': {
                                'filter_query': '{bias_rating_label}="Very Biased"',
                                'column_id': ['title_label', 'bias_rating_label']
                                },
                            'backgroundColor': '#C22625',
                            'color': 'white'
                        },
                        {
                            'if': {
                                'filter_query': '{bias_rating_label}="Biased"',
                                'column_id': ['title_label', 'bias_rating_label']
                                },
                            'backgroundColor': '#eb8483',
                            'color': '#2E2C2B'
                        }
                    ],
                    style_cell={'textAlign': 'left', 'padding': '5px', 'font-family':'sans-serif', 'whiteSpace': 'normal', 'height': 'auto'},
                        style_cell_conditional=[
                            {
                                'if': {'column_id': ['topic', 'title_label']},
                                'width': '600px'
                            }
                        ]
                )

            if id == 'export-button3b':
                return [title], table, {'fontSize':14, 'display': 'block'}, {'fontSize':14, 'display': 'block', 'margin-left': '10px'}, csv_string

            return [title], table, {'fontSize':14, 'display': 'block'}, {'fontSize':14, 'display': 'block', 'margin-left': '10px'}, csv_string

        elif id in ['chart3b-datepickerrange', 'chart3b-publisher-dropdown', 'chart3b-bias-category-dropdown', 'chart3b-topic-dropdown', 'clear-button3b']:
            return [], None, {'display': 'none'}, {'display': 'none'}, ''

    else:
        return [], None, {'display': 'none'}, {'display': 'none'}, ''



# Callback for Table 4A
@app.callback(
    [
        Output('table4a-title', 'children'),
        Output('table4a', 'children'),
        Output('clear-button4a', 'style'),
        Output('export-button4a', 'style'),
        Output('export-button4a', 'href')
    ],
    [
        Input('search-button4a', 'n_clicks'),
        Input('clear-button4a', 'n_clicks'),

        Input('chart4a-datepickerrange', 'start_date'),
        Input('chart4a-datepickerrange', 'end_date'),
        Input('chart4a-publisher-dropdown', 'value'),
        Input('chart4a-topic-dropdown', 'value'),
        Input('chart4a-bias-category-dropdown', 'value'),
        Input('chart4a-bias-rating-dropdown', 'value'),
        Input('chart4a-ngram-dropdown', 'value'),
        Input('chart4a-text-toggle', 'value'),

        Input('word-search-4a', 'value')
    ]
)

def update_table4a(n_clicks_search, n_clicks_clear, selected_start_date, selected_end_date, selected_publishers, selected_topics, selected_bias_categories, selected_bias_ratings, selected_ngrams, text_by, search_word):
    triggered = dash.callback_context.triggered
    topics = ''

    if triggered:
        id = triggered[0]['prop_id'].split('.')[0]

        if id == 'search-button4a':
            filtered_df = df_corpus.copy()

            # Apply filters for dates, publishers, and topics
            if selected_start_date and selected_end_date:
                start_date = pd.to_datetime(str(selected_start_date))
                end_date = pd.to_datetime(str(selected_end_date))
                filtered_df = filtered_df[(filtered_df['date_published'] >= start_date) & (filtered_df['date_published'] <= end_date)]
            if selected_publishers:
                filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
            if selected_topics:
                filtered_df = filtered_df[filtered_df['topic'].str.contains('|'.join(selected_topics))]
                topics = 'having any of the selected topics'
            if selected_bias_ratings:
                filtered_df = filtered_df[filtered_df['bias_rating'].isin(selected_bias_ratings)]
            if selected_bias_categories:
                filtered_df = filtered_df[filtered_df[selected_bias_categories].sum(axis=1) > 0]
            if search_word:
                if text_by ==  'title':
                    filtered_df = filtered_df[filtered_df['title'].str.contains(search_word, case=False, na=False)]
                    text = 'headline'
                elif text_by == 'text':
                    filtered_df = filtered_df[filtered_df['text'].str.contains(search_word, case=False, na=False)]
                    text = 'full-text content'

            # Title
            title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(f"Showing {filtered_df.shape[0]} articles having <b>'{search_word}'</b> in their <b>{text}</b>")

            # Formatting
            filtered_df['color'] = np.select(
                [
                    filtered_df['bias_rating'] == 2,
                    filtered_df['bias_rating'] == 1
                ],
                [
                    'white',
                    '#2E2C2B'
                ],
                '#2E2C2B'
            )
            filtered_df['title_label'] = "<a href='" + filtered_df['article_url'] + "' target='_blank' style='color:" + filtered_df['color'] + ";'>" + filtered_df['title'] + "</a>"
            filtered_df['bias_rating_label'] = np.select(
                [
                    filtered_df['bias_rating'] == -1,
                    filtered_df['bias_rating'] == 0,
                    filtered_df['bias_rating'] == 1,
                    filtered_df['bias_rating'] == 2
                ],
                [
                    'Inconclusive',
                    'Not Biased',
                    'Biased',
                    'Very Biased'
                ],
                default='Unknown'
            )
            filtered_df['date_published_label_(yyyy-mm-dd)'] = filtered_df['date_published'].dt.date
            # filtered_df['explore_further'] = "<a href='" + filtered_df['explainability_url'] + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>"
            filtered_df['explore_further'] = "<a href='" + '' + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>" # CHANGE THIS TO URL LATER

            # Save to csv
            csv_df = filtered_df[['publisher', 'title', 'article_url', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label']]
            csv_df.columns = ['Publisher', 'Title', 'Article URL', 'Date Published (YYYY-MM-DD)', 'Topic', 'Bias Rating']
            csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_df.to_csv(index=False, encoding='utf-8'))

            # Dash
            filtered_df = filtered_df.sort_values('date_published_label_(yyyy-mm-dd)', ascending=False)[['title_label', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label', 'explore_further']]
            table = dash_table.DataTable(
                css=[dict(selector="p", rule="margin: 0; text-align: left")],
                columns=[{'id': x, 'name': x.replace('_', ' ').title(), 'presentation': 'markdown'} if 'title' or 'explore' in x else {'id': x, 'name': x.replace('_', ' ').replace('label', '').title().replace('Or', 'or').replace('Yyyy-Mm-Dd', 'yyyy-mm-dd')} for x in filtered_df.columns],
                markdown_options={"html": True},
                data=filtered_df.to_dict('records'),
                sort_action='native',
                filter_action='native',
                filter_options={'case': 'insensitive'},

                page_current=0,
                page_size=20,
                style_table={'margin': 'auto', 'padding': '0 5px', 'overflowX': 'auto', 'overflowY': 'auto'},
                style_header={'textAlign': 'center', 'fontWeight': 'bold'},
                style_data={'minWidth': '120px', 'maxWidth': '120px', 'width': '120px'},
                style_data_conditional=[
                    {
                        'if': {
                            'filter_query': '{bias_rating_label}="Very Biased"',
                            'column_id': ['title_label', 'bias_rating_label']
                        },
                        'backgroundColor': '#C22625',
                        'color': 'white'
                    },
                    {
                        'if': {
                            'filter_query': '{bias_rating_label}="Biased"',
                            'column_id': ['title_label', 'bias_rating_label']
                        },
                        'backgroundColor': '#eb8483',
                        'color': '#2E2C2B'
                    }
                ],
                style_cell={'textAlign': 'left', 'padding': '5px', 'font-family': 'sans-serif', 'whiteSpace': 'normal', 'height': 'auto', 'minWidth': '180px', 'maxWidth': '180px', 'width': '180px'},
                style_cell_conditional=[
                    {
                        'if': {
                            'column_id': ['topic', 'title']
                        },
                        'width': '600px'
                    }
                ]
            )

            if id == 'export-button4a':
                return [title], table, {'fontSize': 14, 'display': 'block'}, {'fontSize': 14, 'display': 'block', 'margin-left': '10px'}, csv_string

            return [title], table, {'fontSize': 14, 'display': 'block'}, {'fontSize': 14, 'display': 'block', 'margin-left': '10px'}, csv_string

        elif id in ['chart4a-datepickerrange', 'chart4a-publisher-dropdown', 'chart4a-bias-rating-dropdown', 'chart4a-bias-category-dropdown', 'chart4a-topic-dropdown', 'chart4a-ngram-dropdown', 'chart4a-text-toggle', 'clear-button4a']:
            return [], None, {'display': 'none'}, {'display': 'none'}, ''

    return [], None, {'display': 'none'}, {'display': 'none'}, ''


# Callback for Table 4B
@app.callback(
    [
        Output('table4b-title', 'children'),
        Output('table4b', 'children'),
        Output('clear-button4b', 'style'),
        Output('export-button4b', 'style'),
        Output('export-button4b', 'href')
    ],
    [
        Input('search-button4b', 'n_clicks'),
        Input('clear-button4b', 'n_clicks'),

        Input('chart4b-datepickerrange', 'start_date'),
        Input('chart4b-datepickerrange', 'end_date'),
        Input('chart4b-publisher-dropdown', 'value'),
        Input('chart4b-topic-dropdown', 'value'),
        Input('chart4b-bias-category-dropdown', 'value'),
        Input('chart4b-bias-rating-dropdown', 'value'),
        Input('chart4b-ngram-dropdown', 'value'),
        Input('chart4b-text-toggle', 'value'),

        Input('word-search-4b', 'value')
    ]
)

def update_table4b(n_clicks_search, n_clicks_clear, selected_start_date, selected_end_date, selected_publishers, selected_topics, selected_bias_categories, selected_bias_ratings, selected_ngrams, text_by, search_word):
    triggered = dash.callback_context.triggered
    topics = ''

    if triggered:
        id = triggered[0]['prop_id'].split('.')[0]

        if id == 'search-button4b':
            filtered_df = df_corpus.copy()

            # Apply filters for dates, publishers, and topics
            if selected_start_date and selected_end_date:
                start_date = pd.to_datetime(str(selected_start_date))
                end_date = pd.to_datetime(str(selected_end_date))
                filtered_df = filtered_df[(filtered_df['date_published'] >= start_date) & (filtered_df['date_published'] <= end_date)]
            if selected_publishers:
                filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
            if selected_topics:
                filtered_df = filtered_df[filtered_df['topic'].str.contains('|'.join(selected_topics))]
                topics = 'having any of the selected topics'
            if selected_bias_ratings:
                filtered_df = filtered_df[filtered_df['bias_rating'].isin(selected_bias_ratings)]
            if selected_bias_categories:
                filtered_df = filtered_df[filtered_df[selected_bias_categories].sum(axis=1) > 0]
            if search_word:
                if text_by ==  'title':
                    filtered_df = filtered_df[filtered_df['title'].str.contains(search_word, case=False, na=False)]
                    text = 'headline'
                elif text_by == 'text':
                    filtered_df = filtered_df[filtered_df['text'].str.contains(search_word, case=False, na=False)]
                    text = 'full-text content'

            # Title
            title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(f"Showing {filtered_df.shape[0]} articles having <b>'{search_word}'</b> in their <b>{text}</b>")

            # Formatting
            filtered_df['color'] = np.select(
                [
                    filtered_df['bias_rating'] == 2,
                    filtered_df['bias_rating'] == 1
                ],
                [
                    'white',
                    '#2E2C2B'
                ],
                '#2E2C2B'
            )
            filtered_df['title_label'] = "<a href='" + filtered_df['article_url'] + "' target='_blank' style='color:" + filtered_df['color'] + ";'>" + filtered_df['title'] + "</a>"
            filtered_df['bias_rating_label'] = np.select(
                [
                    filtered_df['bias_rating'] == -1,
                    filtered_df['bias_rating'] == 0,
                    filtered_df['bias_rating'] == 1,
                    filtered_df['bias_rating'] == 2
                ],
                [
                    'Inconclusive',
                    'Not Biased',
                    'Biased',
                    'Very Biased'
                ],
                default='Unknown'
            )
            filtered_df['date_published_label_(yyyy-mm-dd)'] = filtered_df['date_published'].dt.date
            # filtered_df['explore_further'] = "<a href='" + filtered_df['explainability_url'] + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>"
            filtered_df['explore_further'] = "<a href='" + '' + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>" # CHANGE THIS TO URL LATER

            # Save to csv
            csv_df = filtered_df[['publisher', 'title', 'article_url', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label']]
            csv_df.columns = ['Publisher', 'Title', 'Article URL', 'Date Published (YYYY-MM-DD)', 'Topic', 'Bias Rating']
            csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_df.to_csv(index=False, encoding='utf-8'))

            # Dash
            filtered_df = filtered_df.sort_values('date_published_label_(yyyy-mm-dd)', ascending=False)[['title_label', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label', 'explore_further']]
            table = dash_table.DataTable(
                css=[dict(selector="p", rule="margin: 0; text-align: left")],
                columns=[{'id': x, 'name': x.replace('_', ' ').title(), 'presentation': 'markdown'} if 'title' or 'explore' in x else {'id': x, 'name': x.replace('_', ' ').replace('label', '').title().replace('Or', 'or').replace('Yyyy-Mm-Dd', 'yyyy-mm-dd')} for x in filtered_df.columns],
                markdown_options={"html": True},
                data=filtered_df.to_dict('records'),
                sort_action='native',
                filter_action='native',
                filter_options={'case': 'insensitive'},

                page_current=0,
                page_size=20,
                style_table={'margin': 'auto', 'padding': '0 5px', 'overflowX': 'auto', 'overflowY': 'auto'},
                style_header={'textAlign': 'center', 'fontWeight': 'bold'},
                style_data={'minWidth': '120px', 'maxWidth': '120px', 'width': '120px'},
                style_data_conditional=[
                    {
                        'if': {
                            'filter_query': '{bias_rating_label}="Very Biased"',
                            'column_id': ['title_label', 'bias_rating_label']
                        },
                        'backgroundColor': '#C22625',
                        'color': 'white'
                    },
                    {
                        'if': {
                            'filter_query': '{bias_rating_label}="Biased"',
                            'column_id': ['title_label', 'bias_rating_label']
                        },
                        'backgroundColor': '#eb8483',
                        'color': '#2E2C2B'
                    }
                ],
                style_cell={'textAlign': 'left', 'padding': '5px', 'font-family': 'sans-serif', 'whiteSpace': 'normal', 'height': 'auto', 'minWidth': '180px', 'maxWidth': '180px', 'width': '180px'},
                style_cell_conditional=[
                    {
                        'if': {
                            'column_id': ['topic', 'title']
                        },
                        'width': '600px'
                    }
                ]
            )

            if id == 'export-button4b':
                return [title], table, {'fontSize': 14, 'display': 'block'}, {'fontSize': 14, 'display': 'block', 'margin-left': '10px'}, csv_string

            return [title], table, {'fontSize': 14, 'display': 'block'}, {'fontSize': 14, 'display': 'block', 'margin-left': '10px'}, csv_string

        elif id in ['chart4b-datepickerrange', 'chart4b-publisher-dropdown', 'chart4b-bias-rating-dropdown', 'chart4b-bias-category-dropdown', 'chart4b-topic-dropdown', 'chart4b-ngram-dropdown', 'chart4b-text-toggle', 'clear-button4b']:
            return [], None, {'display': 'none'}, {'display': 'none'}, ''

    return [], None, {'display': 'none'}, {'display': 'none'}, ''

# # Run the Dash app
# if __name__ == '__main__':
#     app.run_server(host='0.0.0.0', debug=True)

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
