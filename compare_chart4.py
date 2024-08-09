# This file contains only the comparison logic for Charts 4A and 4B.

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

# Define the comparison layout for Chart 1A and Chart 1B
def create_layout():
    layout = html.Div(style={'justify-content': 'center'}, className='row', children=[
        html.H1(children="Trending Words or Phrases", style={'textAlign': 'center'}),

        # Chart 4A vs Chart 4B
        html.Div([

            # All elements for Chart 4A
            html.H2("A", style={'textAlign': 'center'}),
            dbc.Button('Explore', id='explore-button4a', style={'margin-left': '1000px', 'width': '10%', 'display': 'block', 'background-color': '#D90429'}),

            html.Div([
                html.Label(['Date Published:'], style={'font-weight': 'bold', 'width': '20%'}),
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
                html.Label(['Publishers:'], style={'font-weight': 'bold', 'width': '20%'}),
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
                html.Label(['Overall Bias Score:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                    id='chart4a-bias-rating-dropdown',
                    options=[
                        {'label': 'Biased', 'value': 2},
                        {'label': 'Very Biased', 'value': 1},
                        {'label': 'Not Biased', 'value': 0},
                        {'label': 'Inconclusive', 'value': -1},
                    ],
                    value=[1,2],
                    placeholder='Select Overall Bias Score',
                    multi=True,
                    clearable=True,
                    style={'width': '70%'}
                )
            ], style={'display': 'flex', 'margin-bottom': '10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Category of Bias:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                    id='chart4a-bias-category-dropdown',
                    options=[
                        {'label': 'Generalisation', 'value': 'generalisation'},
                        {'label': 'Prominence', 'value': 'prominence'},
                        {'label': 'Negative Behaviour', 'value': 'negative_behaviour'},
                        {'label': 'Misrepresentation', 'value': 'misrepresentation'},
                        {'label': 'Headline or Imagery', 'value': 'headline_or_imagery'},
                    ],
                    placeholder='Select Category of Bias',
                    multi=True,
                    clearable=True,
                    style={'width': '70%'}
                )
            ], style={'display': 'flex', 'margin-bottom': '10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Topics:'], style={'font-weight': 'bold', 'width': '20%'}),
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
                    value=[1,2,3],  # default value on load
                    multi=True,
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
        style={'width': '48%', 'display': 'inline-block', 'border': '1px solid black'}),

        # All elements for Chart 4B
        html.Div([
            html.H2("B", style={'textAlign': 'center'}),
            dbc.Button('Explore', id='explore-button4b', style={'margin-left': '1000px', 'width': '10%', 'display': 'block', 'background-color': '#D90429'}),

            html.Div([
                html.Label(['Date Published:'], style={'font-weight': 'bold', 'width': '20%'}),
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
                html.Label(['Publishers:'], style={'font-weight': 'bold', 'width': '20%'}),
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
                html.Label(['Overall Bias Score:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                    id='chart4b-bias-rating-dropdown',
                    options=[
                        {'label': 'Biased', 'value': 2},
                        {'label': 'Very Biased', 'value': 1},
                        {'label': 'Not Biased', 'value': 0},
                        {'label': 'Inconclusive', 'value': -1},
                    ],
                    value=[1,2],
                    placeholder='Select Overall Bias Score',
                    multi=True,
                    clearable=True,
                    style={'width': '70%'}
                )
            ], style={'display': 'flex', 'margin-bottom': '10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Category of Bias:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                    id='chart4b-bias-category-dropdown',
                    options=[
                        {'label': 'Generalisation', 'value': 'generalisation'},
                        {'label': 'Prominence', 'value': 'prominence'},
                        {'label': 'Negative Behaviour', 'value': 'negative_behaviour'},
                        {'label': 'Misrepresentation', 'value': 'misrepresentation'},
                        {'label': 'Headline or Imagery', 'value': 'headline_or_imagery'},
                    ],
                    placeholder='Select Category of Bias',
                    multi=True,
                    clearable=True,
                    style={'width': '70%'}
                )
            ], style={'display': 'flex', 'margin-bottom': '10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Topics:'], style={'font-weight': 'bold', 'width': '20%'}),
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
                    value=[1,2,3],  # default value on load
                    clearable=False,
                    multi=True,
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
        style={'width': '48%', 'display': 'inline-block', 'border': '1px solid black'}),
    ])

    return layout

def register_callbacks(app):
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
            if ngram_value:
                if len(ngram_value)>1:
                    ngram_range = (ngram_value[0], ngram_value[-1])
                else:
                    ngram_range = (ngram_value[0], ngram_value[0])
            else:
                ngram_range = (1, 3)
            vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
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
                                   f"<b>Count: </b>{raw_count}"
                                   # f"<b>Proportion: </b>{percentage:.2f}%<br>"
                                # f"The word <b>'{word}'</b> appeared <b>{raw_count}</b> times across all articles in the current selection.<br>"
                                # f"This accounts for <b>{percentage:.2f}%</b> of the total available word/phrases.<br>"
                                f"<br>"
                                f"Type <b>'{word}'</b> in the Word Search below <br>"
                                f"to find out which articles used this word.<br>")

            # Identify top 10 words by frequency
            top_10_indices = np.argsort(frequencies)[-10:]
            colors = ['#CFCFCF'] * len(words)
            custom_colors = [
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
                title="<b>What are the trending words/phrases in today's biased/very biased articles?</b>",
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
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
            if ngram_value:
                if len(ngram_value)>1:
                    ngram_range = (ngram_value[0], ngram_value[-1])
                else:
                    ngram_range = (ngram_value[0], ngram_value[0])
            else:
                ngram_range = (1, 3)
            vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
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
                                   f"<b>Count: </b>{raw_count}"
                                   # f"<b>Proportion: </b>{percentage:.2f}%<br>"
                                # f"The word <b>'{word}'</b> appeared <b>{raw_count}</b> times across all articles in the current selection.<br>"
                                # f"This accounts for <b>{percentage:.2f}%</b> of the total available word/phrases.<br>"
                                f"<br>"
                                f"Type <b>'{word}'</b> in the Word Search below <br>"
                                f"to find out which articles used this word.<br>")

            # Identify top 10 words by frequency
            top_10_indices = np.argsort(frequencies)[-10:]
            colors = ['#CFCFCF'] * len(words)
            custom_colors = [
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
                title="<b>What are the trending words/phrases in today's biased/very biased articles?</b>",
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
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

            # Reverse the y-axis to match the word cloud orientation
            fig.update_yaxes(autorange="reversed")

        return fig


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
                    if text_by == 'title':
                        filtered_df = filtered_df[filtered_df['title'].str.contains(search_word, case=False, na=False)]
                        text = 'headline'
                    elif text_by == 'text':
                        filtered_df = filtered_df[filtered_df['text'].str.contains(search_word, case=False, na=False)]
                        text = 'full-text content'

                # Title
                main_title = f"Showing {filtered_df.shape[0]} articles having <b>'{search_word}'</b> in their <b>{text}</b>"
                keys = '<b>Legend: G =</b> Generalisation, <b>O =</b> Omit Due Prominence, <b>N =</b> Negative Behaviour, <b>M =</b> Misinterpretation, <b>H =</b> Headline'
                title_html = f'{main_title}<br>{keys}'
                
                title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(title_html)

                # Formatting
                filtered_df['color'] = '#0066CB'
                # filtered_df['color'] = np.select(
                #     [
                #         filtered_df['bias_rating'] == 2,
                #         filtered_df['bias_rating'] == 1
                #     ],
                #     [
                #         'white',
                #         '#2E2C2B'
                #     ],
                #     '#2E2C2B'
                # )
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
                filtered_df['explore_further'] = "<a href='" + '' + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>"

                # Save to csv
                csv_df = filtered_df[['publisher', 'title', 'article_url', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label']]
                csv_df.columns = ['Publisher', 'Title', 'Article URL', 'Date Published (YYYY-MM-DD)', 'Topic', 'Bias Rating']
                csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_df.to_csv(index=False, encoding='utf-8'))

                # Mapping for specific columns to their new names
                column_name_map = {
                    'generalisation': 'G',
                    'prominence': 'O',
                    'negative_behaviour': 'N',
                    'misrepresentation': 'M',
                    'headline_or_imagery': 'H',
                    'publisher': 'Publisher',
                    'title_label': 'Title',
                    'date_published_label_(yyyy-mm-dd)': 'Date Published (YYYY-MM-DD)',
                    'topic': 'Topic',
                    'bias_rating_label': 'Bias Rating',
                    'explore_further': 'Explore Further'
                }

                # Dash
                filtered_df = filtered_df.sort_values('date_published_label_(yyyy-mm-dd)', ascending=False)[['publisher', 'title_label', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label', 'generalisation', 'prominence', 'negative_behaviour', 'misrepresentation', 'headline_or_imagery', 'explore_further']]
                table = dash_table.DataTable(
                    css=[dict(selector="p", rule="margin:0; text-align:left")],
                    columns=[{'id': x, 'name': column_name_map.get(x, x.replace('_', ' ').title()), 'presentation': 'markdown'} if 'title' in x or 'explore' in x else {'id': x, 'name': column_name_map.get(x, x.replace('_', ' ').replace('label', '').title().replace('Or', 'or').replace('Yyyy-Mm-Dd', 'yyyy-mm-dd'))} for x in filtered_df.columns],
                    markdown_options={"html": True},
                    data=filtered_df.to_dict('records'),
                    sort_action='native',
                    filter_action='native',
                    filter_options={'case': 'insensitive'},

                    page_current=0,
                    page_size=20,
                    style_table={'margin': 'auto', 'padding': '0 5px', 'overflowX': 'auto', 'overflowY': 'auto'},
                    style_header={'textAlign': 'center', 'fontWeight': 'bold'},
                    style_data={'textAlign': 'left', 'padding': '5px', 'font-family': 'sans-serif', 'whiteSpace': 'normal', 'height': 'auto'},
                    # style_data_conditional=[
                    #     {
                    #         'if': {
                    #             'filter_query': '{bias_rating_label}="Very Biased"',
                    #             'column_id': ['title_label', 'bias_rating_label']
                    #         },
                    #         'backgroundColor': '#C22625',
                    #         'color': 'white'
                    #     },
                    #     {
                    #         'if': {
                    #             'filter_query': '{bias_rating_label}="Biased"',
                    #             'column_id': ['title_label', 'bias_rating_label']
                    #         },
                    #         'backgroundColor': '#eb8483',
                    #         'color': '#2E2C2B'
                    #     }
                    # ],
                    style_cell={'textAlign': 'left', 'padding': '5px', 'font-family':'sans-serif', 'whiteSpace': 'normal', 'height': 'auto'},
                    style_cell_conditional=[
                        {'if': {'column_id': 'publisher'}, 'width': '150px'},
                        {'if': {'column_id': 'title_label'}, 'width': '300px'},
                        {'if': {'column_id': 'date_published_label_(yyyy-mm-dd)'}, 'width': '150px'},
                        {'if': {'column_id': 'topic'}, 'width': '200px'},
                        {'if': {'column_id': 'bias_rating_label'}, 'width': '150px'},
                        {'if': {'column_id': 'generalisation'}, 'width': '50px'},
                        {'if': {'column_id': 'prominence'}, 'width': '50px'},
                        {'if': {'column_id': 'negative_behaviour'}, 'width': '50px'},
                        {'if': {'column_id': 'misrepresentation'}, 'width': '50px'},
                        {'if': {'column_id': 'headline_or_imagery'}, 'width': '50px'},
                        {'if': {'column_id': 'explore_further'}, 'width': '200px'}
                    ]
                )

                if id == 'export-button4a':
                    return [title], table, {'fontSize': 14, 'display': 'block'}, {'fontSize': 14, 'display': 'block', 'margin-left': '10px'}, csv_string

                return [title], table, {'fontSize': 14, 'display': 'block'}, {'fontSize': 14, 'display': 'block', 'margin-left': '10px'}, csv_string

            elif id in ['chart4a-datepickerrange', 'chart4a-publisher-dropdown', 'chart4a-bias-rating-dropdown', 'chart4a-bias-category-dropdown', 'chart4a-topic-dropdown', 'chart4a-ngram-dropdown', 'chart4a-text-toggle', 'clear-button4a']:
                return [], None, {'display': 'none'}, {'display': 'none'}, ''

        return [], None, {'display': 'none'}, {'display': 'none'}, ''
    

    # Callback for Chart 4B
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
                    if text_by == 'title':
                        filtered_df = filtered_df[filtered_df['title'].str.contains(search_word, case=False, na=False)]
                        text = 'headline'
                    elif text_by == 'text':
                        filtered_df = filtered_df[filtered_df['text'].str.contains(search_word, case=False, na=False)]
                        text = 'full-text content'

                # Title
                main_title = f"Showing {filtered_df.shape[0]} articles having <b>'{search_word}'</b> in their <b>{text}</b>"
                keys = '<b>Legend: G =</b> Generalisation, <b>O =</b> Omit Due Prominence, <b>N =</b> Negative Behaviour, <b>M =</b> Misinterpretation, <b>H =</b> Headline'
                title_html = f'{main_title}<br>{keys}'
                
                title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(title_html)

                # Formatting
                filtered_df['color'] = '#0066CB'
                # filtered_df['color'] = np.select(
                #     [
                #         filtered_df['bias_rating'] == 2,
                #         filtered_df['bias_rating'] == 1
                #     ],
                #     [
                #         'white',
                #         '#2E2C2B'
                #     ],
                #     '#2E2C2B'
                # )
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
                filtered_df['explore_further'] = "<a href='" + '' + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>"

                # Save to csv
                csv_df = filtered_df[['publisher', 'title', 'article_url', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label']]
                csv_df.columns = ['Publisher', 'Title', 'Article URL', 'Date Published (YYYY-MM-DD)', 'Topic', 'Bias Rating']
                csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_df.to_csv(index=False, encoding='utf-8'))

                # Mapping for specific columns to their new names
                column_name_map = {
                    'generalisation': 'G',
                    'prominence': 'O',
                    'negative_behaviour': 'N',
                    'misrepresentation': 'M',
                    'headline_or_imagery': 'H',
                    'publisher': 'Publisher',
                    'title_label': 'Title',
                    'date_published_label_(yyyy-mm-dd)': 'Date Published (YYYY-MM-DD)',
                    'topic': 'Topic',
                    'bias_rating_label': 'Bias Rating',
                    'explore_further': 'Explore Further'
                }

                # Dash
                filtered_df = filtered_df.sort_values('date_published_label_(yyyy-mm-dd)', ascending=False)[['publisher', 'title_label', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label', 'generalisation', 'prominence', 'negative_behaviour', 'misrepresentation', 'headline_or_imagery', 'explore_further']]
                table = dash_table.DataTable(
                    css=[dict(selector="p", rule="margin:0; text-align:left")],
                    columns=[{'id': x, 'name': column_name_map.get(x, x.replace('_', ' ').title()), 'presentation': 'markdown'} if 'title' in x or 'explore' in x else {'id': x, 'name': column_name_map.get(x, x.replace('_', ' ').replace('label', '').title().replace('Or', 'or').replace('Yyyy-Mm-Dd', 'yyyy-mm-dd'))} for x in filtered_df.columns],
                    markdown_options={"html": True},
                    data=filtered_df.to_dict('records'),
                    sort_action='native',
                    filter_action='native',
                    filter_options={'case': 'insensitive'},

                    page_current=0,
                    page_size=20,
                    style_table={'margin': 'auto', 'padding': '0 5px', 'overflowX': 'auto', 'overflowY': 'auto'},
                    style_header={'textAlign': 'center', 'fontWeight': 'bold'},
                    style_data={'textAlign': 'left', 'padding': '5px', 'font-family': 'sans-serif', 'whiteSpace': 'normal', 'height': 'auto'},
                    # style_data_conditional=[
                    #     {
                    #         'if': {
                    #             'filter_query': '{bias_rating_label}="Very Biased"',
                    #             'column_id': ['title_label', 'bias_rating_label']
                    #         },
                    #         'backgroundColor': '#C22625',
                    #         'color': 'white'
                    #     },
                    #     {
                    #         'if': {
                    #             'filter_query': '{bias_rating_label}="Biased"',
                    #             'column_id': ['title_label', 'bias_rating_label']
                    #         },
                    #         'backgroundColor': '#eb8483',
                    #         'color': '#2E2C2B'
                    #     }
                    # ],
                    style_cell={'textAlign': 'left', 'padding': '5px', 'font-family':'sans-serif', 'whiteSpace': 'normal', 'height': 'auto'},
                    style_cell_conditional=[
                        {'if': {'column_id': 'publisher'}, 'width': '150px'},
                        {'if': {'column_id': 'title_label'}, 'width': '300px'},
                        {'if': {'column_id': 'date_published_label_(yyyy-mm-dd)'}, 'width': '150px'},
                        {'if': {'column_id': 'topic'}, 'width': '200px'},
                        {'if': {'column_id': 'bias_rating_label'}, 'width': '150px'},
                        {'if': {'column_id': 'generalisation'}, 'width': '50px'},
                        {'if': {'column_id': 'prominence'}, 'width': '50px'},
                        {'if': {'column_id': 'negative_behaviour'}, 'width': '50px'},
                        {'if': {'column_id': 'misrepresentation'}, 'width': '50px'},
                        {'if': {'column_id': 'headline_or_imagery'}, 'width': '50px'},
                        {'if': {'column_id': 'explore_further'}, 'width': '200px'}
                    ]
                )

                if id == 'export-button4b':
                    return [title], table, {'fontSize': 14, 'display': 'block'}, {'fontSize': 14, 'display': 'block', 'margin-left': '10px'}, csv_string

                return [title], table, {'fontSize': 14, 'display': 'block'}, {'fontSize': 14, 'display': 'block', 'margin-left': '10px'}, csv_string

            elif id in ['chart4b-datepickerrange', 'chart4b-publisher-dropdown', 'chart4b-bias-rating-dropdown', 'chart4b-bias-category-dropdown', 'chart4b-topic-dropdown', 'chart4b-ngram-dropdown', 'chart4b-text-toggle', 'clear-button4b']:
                return [], None, {'display': 'none'}, {'display': 'none'}, ''

        return [], None, {'display': 'none'}, {'display': 'none'}, ''
