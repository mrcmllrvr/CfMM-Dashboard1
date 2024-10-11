# This file contains only the comparison logic for Charts 1A and 1B.

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
stylesheets = [
    dbc.themes.BOOTSTRAP,
    dbc.icons.BOOTSTRAP,
    '/assets/custom.css'
]
app = dash.Dash(__name__, external_stylesheets=stylesheets)

# Define the comparison layout for Chart 1A and Chart 1B
def create_layout():
    layout = html.Div(style={'justify-content': 'center', 'backgroundColor': '#ffffff'}, className='row', children=[
        html.H3(children=f"""Who are the top offending publishers during the selected period?""", style={'textAlign': 'center', 'font-weight': 'bold'}),

        # Chart 1A vs Chart 1B
        html.Div([

            # All elements for Chart 2A
            html.H3("Scenario A", style={'textAlign': 'center', 'font-weight': 'normal'}),
            html.H3(''),

            html.Div([
                html.Label(
                    [
                        html.I(className="bi-calendar-week", style={'vertical-align': 'middle', 'font-size': '1.5em'}),
                        html.Span(' Date Published:', style={'vertical-align': 'middle'})
                    ],
                    style={'font-weight': 'bold', 'width': '30%'}
                ),
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
                    style = {'font-size':'10px'}
                )
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(
                    [
                        html.I(className="bi-person-fill", style={'vertical-align': 'middle', 'font-size': '1.5em'}),
                        html.Span(' Publishers:', style={'vertical-align': 'middle'})
                    ],
                    style={'font-weight': 'bold', 'width': '30%'}
                ),
                dcc.Dropdown(
                    id='chart1a-publisher-dropdown',
                    options=[{'label': publisher, 'value': publisher} for publisher in unique_publishers],
                    placeholder='Select Publisher',
                    multi=True,
                    clearable=True,
                    style = {'width': '60%'}
                )
            ], style={'font-size':'10px', 'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(
                    [
                        html.I(className="bi-speedometer2", style={'vertical-align': 'middle', 'font-size': '1.5em'}),
                        html.Span(' Overall Bias Score:', style={'vertical-align': 'middle'})
                    ],
                    style={'font-weight': 'bold', 'width': '30%'}
                ),
                dcc.Dropdown(
                    id='chart1a-bias-rating-dropdown',
                    options=[
                        {'label': 'Inconclusive', 'value':-1},
                        {'label': 'Biased', 'value': 1},
                        {'label': 'Very Biased', 'value': 2},
                        {'label': 'Not Biased', 'value': 0},
                    ],
                    placeholder='Select Overall Bias Score',
                    multi=True,
                    clearable=True,
                    style = {'width': '60%'}
                )
            ], style={'font-size':'10px', 'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(
                    [
                        html.I(className="bi-boxes", style={'vertical-align': 'middle', 'font-size': '1.5em'}),
                        html.Span(' Category of Bias:', style={'vertical-align': 'middle'})
                    ],
                    style={'font-weight': 'bold', 'width': '30%'}
                ),
                dcc.Dropdown(
                    id='chart1a-bias-category-dropdown',
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
                    style = {'width': '60%'}
                )
            ], style={'font-size':'10px', 'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(
                    [
                        html.I(className="bi-chat-dots", style={'vertical-align': 'middle', 'font-size': '1.5em'}),
                        html.Span(' Topics:', style={'vertical-align': 'middle'})
                    ],
                    style={'font-weight': 'bold', 'width': '30%'}
                ),
                dcc.Dropdown(
                    id='chart1a-topic-dropdown',
                    options=[{'label': topic, 'value': topic} for topic in unique_topics],
                    placeholder='Select Topic',
                    multi=True,
                    clearable=True,
                    style = {'width': '60%'}
                )
            ], style={'font-size':'10px', 'display':'flex', 'margin-bottom':'30px', 'align-items': 'center'}),

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
                style = {'font-size':'10px', 'margin-bottom': '50px'}
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
        style={'width': '48%', 'display': 'inline-block', 'backgroundColor': 'white', 'border': '2px solid #d3d3d3', 'border-radius': '8px', 'padding': '10px', 'margin': '5px'}),

        # All elements for Chart 1B
        html.Div([
            html.H3("Scenario B", style={'textAlign': 'center', 'font-weight': 'normal'}),
            html.H3(''),
            
            html.Div([
                html.Label(
                    [
                        html.I(className="bi-calendar-week", style={'vertical-align': 'middle', 'font-size': '1.5em'}),
                        html.Span(' Date Published:', style={'vertical-align': 'middle'})
                    ],
                    style={'font-weight': 'bold', 'width': '30%'}
                ),
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
                    style = {'font-size':'13px'}
                )
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(
                    [
                        html.I(className="bi-person-fill", style={'vertical-align': 'middle', 'font-size': '1.5em'}),
                        html.Span(' Publishers:', style={'vertical-align': 'middle'})
                    ],
                    style={'font-weight': 'bold', 'width': '30%'}
                ),
                dcc.Dropdown(
                    id='chart1b-publisher-dropdown',
                    options=[{'label': publisher, 'value': publisher} for publisher in unique_publishers],
                    placeholder='Select Publisher',
                    multi=True,
                    clearable=True,
                    style = {'width': '60%'}
                )
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(
                    [
                        html.I(className="bi-speedometer2", style={'vertical-align': 'middle', 'font-size': '1.5em'}),
                        html.Span(' Overall Bias Score:', style={'vertical-align': 'middle'})
                    ],
                    style={'font-weight': 'bold', 'width': '30%'}
                ),
                dcc.Dropdown(
                    id='chart1b-bias-rating-dropdown',
                    options=[
                        {'label': 'Inconclusive', 'value':-1},
                        {'label': 'Biased', 'value': 1},
                        {'label': 'Very Biased', 'value': 2},
                        {'label': 'Not Biased', 'value': 0},
                    ],
                    placeholder='Select Overall Bias Score',
                    multi=True,
                    clearable=True,
                    style = {'width': '60%'}
                )
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(
                    [
                        html.I(className="bi-boxes", style={'vertical-align': 'middle', 'font-size': '1.5em'}),
                        html.Span(' Category of Bias:', style={'vertical-align': 'middle'})
                    ],
                    style={'font-weight': 'bold', 'width': '30%'}
                ),
                dcc.Dropdown(
                    id='chart1b-bias-category-dropdown',
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
                    style = {'width': '60%'}
                )
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(
                    [
                        html.I(className="bi-chat-dots", style={'vertical-align': 'middle', 'font-size': '1.5em'}),
                        html.Span(' Topics:', style={'vertical-align': 'middle'})
                    ],
                    style={'font-weight': 'bold', 'width': '30%'}
                ),
                dcc.Dropdown(
                    id='chart1b-topic-dropdown',
                    options=[{'label': topic, 'value': topic} for topic in unique_topics],
                    placeholder='Select Topic',
                    multi=True,
                    clearable=True,
                    style = {'width': '60%'}
                )
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
        style={'width': '48%', 'display': 'inline-block', 'backgroundColor': 'white', 'border': '2px solid #d3d3d3', 'border-radius': '8px', 'padding': '10px', 'margin': '5px'}),
    ])

    return layout


def register_callbacks(app):
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
        # Apply filters for Chart 1A
        if selected_start_date and selected_end_date:
            start_date = pd.to_datetime(selected_start_date)
            end_date = pd.to_datetime(selected_end_date)
            filtered_df = filtered_df[(filtered_df['date_published'] >= start_date) & (filtered_df['date_published'] <= end_date)]
        if selected_publishers:
            filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
        if selected_bias_ratings:
            filtered_df = filtered_df[filtered_df['bias_rating'].isin(selected_bias_ratings)]
        if selected_bias_categories:
            filtered_df = filtered_df[filtered_df[selected_bias_categories].sum(axis=1) > 0]
        if selected_topics:
            filtered_df = filtered_df[filtered_df['topic'].str.contains('|'.join(selected_topics))]

        # Logic to update the chart
        publisher_totals = filtered_df[filtered_df['bias_rating']>=1].groupby('publisher', observed=True).size()
        top_publishers = publisher_totals.sort_values(ascending=False).head(10).index[::-1]
        filtered_df = filtered_df[filtered_df['publisher'].isin(top_publishers)]
        filtered_df['publisher'] = pd.Categorical(filtered_df['publisher'], ordered=True, categories=top_publishers)
        filtered_df = filtered_df.sort_values('publisher')

        if color_by == 'bias_ratings':
            # If chart is empty, show text instead
            if filtered_df.shape[0]==0:
                data = []
                layout = {
                    'xaxis': {'visible': False},
                    'yaxis': {'visible': False},
                    'template': 'simple_white',
                    'height': 400,
                    'annotations': [{
                        'text': 'No biased articles found in the current selection.',
                        'showarrow': False,
                        'xref': 'paper',
                        'yref': 'paper',
                        'x': 0.5,
                        'y': 0.5,
                        'font': {'size': 20, 'color': '#2E2C2B'}
                    }]
                }
            else:
                color_map = {
                    2: ('#C22625', 'Very Biased'),
                    1: ('#eb8483', 'Biased'),
                    0: ('#f2eadf', 'Not Biased'), # #FFE5DC
                    -1: ('#CAC6C2', 'Inconclusive')
                }
                legend_added = set()
                data = []
                for publisher in top_publishers:
                    total_biased_articles = filtered_df[filtered_df['publisher'] == publisher]['bias_rating'].count()
                    for rating, (color, name) in color_map.items():
                        articles = filtered_df[(filtered_df['publisher'] == publisher) & (filtered_df['bias_rating'] == rating)]['bias_rating'].count()
                        percentage_of_total = (articles / total_biased_articles) * 100 if total_biased_articles > 0 else 0
                        tooltip_text = (
                            # f"<b>Publisher: </b>{publisher}<br>"
                            f"<b>Overall Bias Score:</b> {name}<br>"
                            f"<b>Count:</b> {articles}<br>"
                            f"<b>Proportion: </b> {percentage_of_total:.1f}%"
                            # f"This accounts for <b>{percentage_of_total:.2f}%</b> of the total available articles in the current selection.<br>"
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
                layout = go.Layout(
                    title='',
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
            # If chart is empty, show text instead
            if filtered_df.shape[0]==0:
                data = []
                layout = {
                    'xaxis': {'visible': False},
                    'yaxis': {'visible': False},
                    'template': 'simple_white',
                    'height': 400,
                    'annotations': [{
                        'text': 'No biased articles found in the current selection.',
                        'showarrow': False,
                        'xref': 'paper',
                        'yref': 'paper',
                        'x': 0.5,
                        'y': 0.5,
                        'font': {'size': 20, 'color': '#2E2C2B'}
                    }]
                }
            else:
                categories = ['generalisation', 'prominence', 'negative_behaviour', 'misrepresentation', 'headline_or_imagery']
                category_colors = ['#4185A0', '#AA4D71', '#B85C3B', '#C5BE71', '#7658A0']
                legend_added = set()
                data = []
                filtered_df['total_bias_category'] = filtered_df[categories].sum(axis=1)
                for i, category in enumerate(categories):
                    articles_list = []
                    tooltip_text_list = []
                    for publisher in filtered_df['publisher'].unique():
                        total_biased_articles = filtered_df[filtered_df['publisher'] == publisher]['article_url'].nunique()
                        articles = filtered_df[(filtered_df['publisher'] == publisher) & (filtered_df[category] == 1)].shape[0]
                        articles_list += [articles]
                        percentage_of_total = (articles / total_biased_articles * 100) if total_biased_articles > 0 else 0
                        tooltip_text = (
                            # f"<b>Publisher: </b>{publisher}<br>"
                            f"<b>Category of Bias: </b>{category.replace('_', ' ').title().replace('Or', 'or')}<br>"
                            f"<b>Count: </b> {articles}<br>"
                            f"<b>Proportion: </b>{percentage_of_total:.1f}%<br>"
                            # f"Of the {total_biased_articles} articles, <b>{articles}</b> of them committed <b>{category.replace('_', ' ').title().replace('Or', 'or')}</b>.<br>"
                            # f"This accounts for <b>{percentage_of_total:.2f}%</b> of the total available articles for <b>{category.replace('_', ' ').title().replace('Or', 'or')}</b>.<br>"
                        )
                        tooltip_text_list += [tooltip_text]
                    showlegend = category not in legend_added
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
                layout = go.Layout(
                    title='',
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
        # Apply filters for Chart 1B
        if selected_start_date and selected_end_date:
            start_date = pd.to_datetime(selected_start_date)
            end_date = pd.to_datetime(selected_end_date)
            filtered_df = filtered_df[(filtered_df['date_published'] >= start_date) & (filtered_df['date_published'] <= end_date)]
        if selected_publishers:
            filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
        if selected_bias_ratings:
            filtered_df = filtered_df[filtered_df['bias_rating'].isin(selected_bias_ratings)]
        if selected_bias_categories:
            filtered_df = filtered_df[filtered_df[selected_bias_categories].sum(axis=1) > 0]
        if selected_topics:
            filtered_df = filtered_df[filtered_df['topic'].str.contains('|'.join(selected_topics))]

        # Logic to update the chart
        if filtered_df.shape[0] == 0:
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
            publisher_totals = filtered_df[filtered_df['bias_rating']>=1].groupby('publisher', observed=True).size()
            top_publishers = publisher_totals.sort_values(ascending=False).head(10).index[::-1]
            filtered_df = filtered_df[filtered_df['publisher'].isin(top_publishers)]
            filtered_df['publisher'] = pd.Categorical(filtered_df['publisher'], ordered=True, categories=top_publishers)
            filtered_df = filtered_df.sort_values('publisher')

            if color_by == 'bias_ratings':
                # If chart is empty, show text instead
                if filtered_df.shape[0]==0:
                    data = []
                    layout = {
                        'xaxis': {'visible': False},
                        'yaxis': {'visible': False},
                        'template': 'simple_white',
                        'height': 400,
                        'annotations': [{
                            'text': 'No biased articles found in the current selection.',
                            'showarrow': False,
                            'xref': 'paper',
                            'yref': 'paper',
                            'x': 0.5,
                            'y': 0.5,
                            'font': {'size': 20, 'color': '#2E2C2B'}
                        }]
                    }
                else:
                    color_map = {
                        2: ('#C22625', 'Very Biased'),
                        1: ('#eb8483', 'Biased'),
                        0: ('#f2eadf', 'Not Biased'), # #FFE5DC
                        -1: ('#CAC6C2', 'Inconclusive')
                    }
                    legend_added = set()
                    data = []
                    for publisher in top_publishers:
                        total_biased_articles = filtered_df[filtered_df['publisher'] == publisher]['bias_rating'].count()
                        for rating, (color, name) in color_map.items():
                            articles = filtered_df[(filtered_df['publisher'] == publisher) & (filtered_df['bias_rating'] == rating)]['bias_rating'].count()
                            percentage_of_total = (articles / total_biased_articles) * 100 if total_biased_articles > 0 else 0
                            tooltip_text = (
                                # f"<b>Publisher: </b>{publisher}<br>"
                                f"<b>Overall Bias Score:</b> {name}<br>"
                                f"<b>Count: </b> {articles}<br>"
                                f"<b>Proportion: {percentage_of_total:.1f}%"
                                # f"This accounts for <b>{percentage_of_total:.2f}%</b> of the total available articles in the current selection.<br>"
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
                    layout = go.Layout(
                        title='',
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
                # If chart is empty, show text instead
                if filtered_df.shape[0]==0:
                    data = []
                    layout = {
                        'xaxis': {'visible': False},
                        'yaxis': {'visible': False},
                        'template': 'simple_white',
                        'height': 400,
                        'annotations': [{
                            'text': 'No biased articles found in the current selection.',
                            'showarrow': False,
                            'xref': 'paper',
                            'yref': 'paper',
                            'x': 0.5,
                            'y': 0.5,
                            'font': {'size': 20, 'color': '#2E2C2B'}
                        }]
                    }
                else:
                    categories = ['generalisation', 'prominence', 'negative_behaviour', 'misrepresentation', 'headline_or_imagery']
                    category_colors = ['#4185A0', '#AA4D71', '#B85C3B', '#C5BE71', '#7658A0']
                    legend_added = set()
                    data = []
                    filtered_df['total_bias_category'] = filtered_df[categories].sum(axis=1)
                    for i, category in enumerate(categories):
                        articles_list = []
                        tooltip_text_list = []
                        for publisher in filtered_df['publisher'].unique():
                            total_biased_articles = filtered_df[filtered_df['publisher'] == publisher]['article_url'].nunique()
                            articles = filtered_df[(filtered_df['publisher'] == publisher) & (filtered_df[category] == 1)].shape[0]
                            articles_list += [articles]
                            percentage_of_total = (articles / total_biased_articles * 100) if total_biased_articles > 0 else 0
                            tooltip_text = (
                                # f"<b>Publisher: </b>{publisher}<br>"
                                f"<b>Category of Bias: </b>{category.replace('_', ' ').title().replace('Or', 'or')}<br>"
                                f"<b>Count: </b>{articles}<br>"
                                f"<b>Proportion: </b>{percentage_of_total:.1f}%<br>"
                                # f"Of the {total_biased_articles} articles, <b>{articles}</b> of them committed <b>{category.replace('_', ' ').title().replace('Or', 'or')}</b>.<br>"
                                # f"This accounts for <b>{percentage_of_total:.2f}%</b> of the total available articles for <b>{category.replace('_', ' ').title().replace('Or', 'or')}</b>.<br>"
                            )
                            tooltip_text_list += [tooltip_text]
                        showlegend = category not in legend_added
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
                    layout = go.Layout(
                        title=f"""<b>Who are the top offending publishers during the selected period?</b>""",
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
                    filtered_df = filtered_df[(filtered_df['date_published'] >= start_date) & (filtered_df['date_published'] <= end_date)]
                if selected_publishers:
                    filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
                if selected_bias_ratings:
                    filtered_df = filtered_df[filtered_df['bias_rating'].isin(selected_bias_ratings)]
                if selected_bias_categories:
                    filtered_df = filtered_df[filtered_df[selected_bias_categories].sum(axis=1) > 0]
                if selected_topics:
                    filtered_df = filtered_df[filtered_df['topic'].str.contains('|'.join(selected_topics))]
                    topics = 'having any of the selected topics'

                if (clickData is not None) or (clickData is None & id == 'export-button1'):
                    publisher = str(clickData['points'][0]['label'])
                    filtered_df = filtered_df[filtered_df['publisher'] == publisher]
                    start_date = pd.to_datetime(str(selected_start_date)).strftime('%d %b %Y')
                    end_date = pd.to_datetime(str(selected_end_date)).strftime('%d %b %Y')

                    if color_by == 'bias_ratings':
                        # Table title
                        main_title = f'Showing all articles from <b>{start_date}</b> to <b>{end_date}</b> {topics}'
                        keys = '<b>Legend: G =</b> Generalisation, <b>O =</b> Omit Due Prominence, <b>N =</b> Negative Behaviour, <b>M =</b> Misrepresentationn, <b>H =</b> Headline'
                        title_html = f'{main_title}<br>{keys}'
                        title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(title_html)

                        # Apply formatting
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
                        #     default='#2E2C2B'
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
                        categories = ['generalisation', 'prominence', 'negative_behaviour', 'misrepresentation', 'headline_or_imagery']
                        for category in categories:
                            filtered_df[category] = np.where(filtered_df[category] == 1, 'Y', 'N')
                        filtered_df['date_published_label_(yyyy-mm-dd)'] = filtered_df['date_published'].dt.date
                        filtered_df['explore_further'] = "<a href='" + '' + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>"

                        # Save to csv
                        csv_df = filtered_df[['publisher', 'title', 'article_url', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label'] + categories]
                        csv_df.columns = ['Publisher', 'Title', 'Article URL', 'Date Published (YYYY-MM-DD)', 'Topic', 'Bias Rating'] + [c.upper() for c in categories]
                        csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_df.to_csv(index=False, encoding='utf-8'))

                        # Mapping for specific columns to their new names
                        column_name_map = {
                            'publisher': 'Publisher',
                            'title_label': 'Title Label',
                            'date_published_label_(yyyy-mm-dd)': 'Date Published (YYYY-MM-DD)',
                            'topic': 'Topic',
                            'bias_rating_label': 'Bias Rating',
                            'generalisation': 'G',
                            'prominence': 'O',
                            'negative_behaviour': 'N',
                            'misrepresentation': 'M',
                            'headline_or_imagery': 'H',
                            'explore_further': 'Explore Further'
                        }

                        # Dash
                        filtered_df = filtered_df.sort_values('date_published_label_(yyyy-mm-dd)', ascending=False)[['publisher', 'title_label', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label'] + categories + ['explore_further']]
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
                            # style_data={'textAlign': 'left', 'padding': '5px', 'font-family': 'sans-serif', 'whiteSpace': 'normal', 'height': 'auto'},
                            # style_data_conditional=[
                            #     {
                            #         'if': {
                            #             'filter_query': '{bias_rating_label}="Very Biased"',
                            #             'column_id': ['title_label', 'bias_rating_label']
                            #             },
                            #         'backgroundColor': '#C22625',
                            #         'color': 'white'
                            #     },
                            #     {
                            #         'if': {
                            #             'filter_query': '{bias_rating_label}="Biased"',
                            #             'column_id': ['title_label', 'bias_rating_label']
                            #             },
                            #         'backgroundColor': '#eb8483',
                            #         'color': '#2E2C2B'
                            #     }
                            # ],
                            style_cell={'textAlign': 'center', 'padding': '5px', 'font-family':'sans-serif', 'whiteSpace': 'normal', 'height': 'auto'},
                            style_cell_conditional=[
                                {'if': {'column_id': 'publisher'}, 'width': '150px'},
                                {'if': {'column_id': 'title_label'}, 'width': '300px', 'textAlign': 'left'},
                                {'if': {'column_id': 'date_published_label_(yyyy-mm-dd)'}, 'width': '150px'},
                                {'if': {'column_id': 'topic'}, 'width': '200px', 'textAlign': 'left'},
                                {'if': {'column_id': 'bias_rating_label'}, 'width': '150px'},
                                {'if': {'column_id': 'generalisation'}, 'width': '50px'},
                                {'if': {'column_id': 'prominence'}, 'width': '50px'},
                                {'if': {'column_id': 'negative_behaviour'}, 'width': '50px'},
                                {'if': {'column_id': 'misrepresentation'}, 'width': '50px'},
                                {'if': {'column_id': 'headline_or_imagery'}, 'width': '50px'},
                                {'if': {'column_id': 'explore_further'}, 'width': '200px'}
                            ]
                        )

                    else:
                        main_title = f'Showing biased/very biased articles from <b>{publisher}</b> published <b>{start_date}</b> to <b>{end_date}</b> {topics}'
                        keys = '<b>Legend: G =</b> Generalisation, <b>O =</b> Omit Due Prominence, <b>N =</b> Negative Behaviour, <b>M =</b> Misrepresentation, <b>H =</b> Headline'
                        title_html = f'{main_title}<br>{keys}'
                        title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(title_html)
                        filtered_df['color'] = '#0066CB'
                        filtered_df['color'] = np.select(
                            [
                                filtered_df['bias_rating'] == 2,
                                filtered_df['bias_rating'] == 1
                            ],
                            [
                                'white',
                                '#2E2C2B'
                            ],
                            default='#2E2C2B'
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
                        filtered_df['explore_further'] = "<a href='" + '' + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>"

                        # Save to csv
                        csv_df = filtered_df[['publisher', 'title', 'article_url', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label'] + categories]
                        csv_df.columns = ['Publisher', 'Title', 'Article URL', 'Date Published (YYYY-MM-DD)', 'Topic', 'Bias Rating'] + [c.upper() for c in categories]
                        csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_df.to_csv(index=False, encoding='utf-8'))

                        # Mapping for specific columns to their new names
                        column_name_map = {
                            'publisher': 'Publisher',
                            'title_label': 'Title Label',
                            'date_published_label_(yyyy-mm-dd)': 'Date Published (YYYY-MM-DD)',
                            'topic': 'Topic',
                            'bias_rating_label': 'Bias Rating',
                            'generalisation': 'G',
                            'prominence': 'O',
                            'negative_behaviour': 'N',
                            'misrepresentation': 'M',
                            'headline_or_imagery': 'H',
                            'explore_further': 'Explore Further'
                        }

                        # Dash
                        filtered_df = filtered_df.sort_values('date_published_label_(yyyy-mm-dd)', ascending=False)[['publisher', 'title_label', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label'] + categories + ['explore_further']]
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
                            # style_data={'textAlign': 'left', 'padding': '5px', 'font-family': 'sans-serif', 'whiteSpace': 'normal', 'height': 'auto'},
                            # style_data_conditional=[
                            #     {
                            #         'if': {
                            #             'filter_query': '{bias_rating_label}="Very Biased"',
                            #             'column_id': ['title_label', 'bias_rating_label']
                            #             },
                            #         'backgroundColor': '#C22625',
                            #         'color': 'white'
                            #     },
                            #     {
                            #         'if': {
                            #             'filter_query': '{bias_rating_label}="Biased"',
                            #             'column_id': ['title_label', 'bias_rating_label']
                            #             },
                            #         'backgroundColor': '#eb8483',
                            #         'color': '#2E2C2B'
                            #     }
                            # ],
                            style_cell={'textAlign': 'center', 'padding': '5px', 'font-family':'sans-serif', 'whiteSpace': 'normal', 'height': 'auto'},
                            style_cell_conditional=[
                                {'if': {'column_id': 'publisher'}, 'width': '150px'},
                                {'if': {'column_id': 'title_label'}, 'width': '300px', 'textAlign': 'left'},
                                {'if': {'column_id': 'date_published_label_(yyyy-mm-dd)'}, 'width': '150px'},
                                {'if': {'column_id': 'topic'}, 'width': '200px', 'textAlign': 'left'},
                                {'if': {'column_id': 'bias_rating_label'}, 'width': '150px'},
                                {'if': {'column_id': 'generalisation'}, 'width': '50px'},
                                {'if': {'column_id': 'prominence'}, 'width': '50px'},
                                {'if': {'column_id': 'negative_behaviour'}, 'width': '50px'},
                                {'if': {'column_id': 'misrepresentation'}, 'width': '50px'},
                                {'if': {'column_id': 'headline_or_imagery'}, 'width': '50px'},
                                {'if': {'column_id': 'explore_further'}, 'width': '200px'}
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
                    filtered_df = filtered_df[(filtered_df['date_published'] >= start_date) & (filtered_df['date_published'] <= end_date)]
                if selected_publishers:
                    filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
                if selected_bias_ratings:
                    filtered_df = filtered_df[filtered_df['bias_rating'].isin(selected_bias_ratings)]
                if selected_bias_categories:
                    filtered_df = filtered_df[filtered_df[selected_bias_categories].sum(axis=1) > 0]
                if selected_topics:
                    filtered_df = filtered_df[filtered_df['topic'].str.contains('|'.join(selected_topics))]
                    topics = 'having any of the selected topics'

                if (clickData is not None) or (clickData is None & id == 'export-button1'):
                    publisher = str(clickData['points'][0]['label'])
                    filtered_df = filtered_df[filtered_df['publisher'] == publisher]
                    start_date = pd.to_datetime(str(selected_start_date)).strftime('%d %b %Y')
                    end_date = pd.to_datetime(str(selected_end_date)).strftime('%d %b %Y')

                    if color_by == 'bias_ratings':
                        # Table title
                        main_title = f'Showing all articles from <b>{start_date}</b> to <b>{end_date}</b> {topics}'
                        keys = '<b>Legend: G =</b> Generalisation, <b>O =</b> Omit Due Prominence, <b>N =</b> Negative Behaviour, <b>M =</b> Misrepresentation, <b>H =</b> Headline'
                        title_html = f'{main_title}<br>{keys}'
                        title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(title_html)

                        # Apply formatting
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
                        #     default='#2E2C2B'
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
                        categories = ['generalisation', 'prominence', 'negative_behaviour', 'misrepresentation', 'headline_or_imagery']
                        for category in categories:
                            filtered_df[category] = np.where(filtered_df[category] == 1, 'Y', 'N')
                        filtered_df['date_published_label_(yyyy-mm-dd)'] = filtered_df['date_published'].dt.date
                        filtered_df['explore_further'] = "<a href='" + '' + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>"

                        # Save to csv
                        csv_df = filtered_df[['publisher', 'title', 'article_url', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label'] + categories]
                        csv_df.columns = ['Publisher', 'Title', 'Article URL', 'Date Published (YYYY-MM-DD)', 'Topic', 'Bias Rating'] + [c.upper() for c in categories]
                        csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_df.to_csv(index=False, encoding='utf-8'))

                        # Mapping for specific columns to their new names
                        column_name_map = {
                            'publisher': 'Publisher',
                            'title_label': 'Title Label',
                            'date_published_label_(yyyy-mm-dd)': 'Date Published (YYYY-MM-DD)',
                            'topic': 'Topic',
                            'bias_rating_label': 'Bias Rating',
                            'generalisation': 'G',
                            'prominence': 'O',
                            'negative_behaviour': 'N',
                            'misrepresentation': 'M',
                            'headline_or_imagery': 'H',
                            'explore_further': 'Explore Further'
                        }

                        # Dash
                        filtered_df = filtered_df.sort_values('date_published_label_(yyyy-mm-dd)', ascending=False)[['publisher', 'title_label', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label'] + categories + ['explore_further']]
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
                            # style_data={'textAlign': 'left', 'padding': '5px', 'font-family': 'sans-serif', 'whiteSpace': 'normal', 'height': 'auto'},
                            # style_data_conditional=[
                            #     {
                            #         'if': {
                            #             'filter_query': '{bias_rating_label}="Very Biased"',
                            #             'column_id': ['title_label', 'bias_rating_label']
                            #             },
                            #         'backgroundColor': '#C22625',
                            #         'color': 'white'
                            #     },
                            #     {
                            #         'if': {
                            #             'filter_query': '{bias_rating_label}="Biased"',
                            #             'column_id': ['title_label', 'bias_rating_label']
                            #             },
                            #         'backgroundColor': '#eb8483',
                            #         'color': '#2E2C2B'
                            #     }
                            # ],
                            style_cell={'textAlign': 'center', 'padding': '5px', 'font-family':'sans-serif', 'whiteSpace': 'normal', 'height': 'auto'},
                            style_cell_conditional=[
                                {'if': {'column_id': 'publisher'}, 'width': '150px'},
                                {'if': {'column_id': 'title_label'}, 'width': '300px', 'textAlign': 'left'},
                                {'if': {'column_id': 'date_published_label_(yyyy-mm-dd)'}, 'width': '150px'},
                                {'if': {'column_id': 'topic'}, 'width': '200px', 'textAlign': 'left'},
                                {'if': {'column_id': 'bias_rating_label'}, 'width': '150px'},
                                {'if': {'column_id': 'generalisation'}, 'width': '50px'},
                                {'if': {'column_id': 'prominence'}, 'width': '50px'},
                                {'if': {'column_id': 'negative_behaviour'}, 'width': '50px'},
                                {'if': {'column_id': 'misrepresentation'}, 'width': '50px'},
                                {'if': {'column_id': 'headline_or_imagery'}, 'width': '50px'},
                                {'if': {'column_id': 'explore_further'}, 'width': '200px'}
                            ]
                        )

                    else:
                        main_title = f'Showing biased/very biased articles from <b>{publisher}</b> published <b>{start_date}</b> to <b>{end_date}</b> {topics}'
                        keys = '<b>Legend: G =</b> Generalisation, <b>O =</b> Omit Due Prominence, <b>N =</b> Negative Behaviour, <b>M =</b> Misrepresentation, <b>H =</b> Headline'
                        title_html = f'{main_title}<br>{keys}'
                        title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(title_html)
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
                        #     default='#2E2C2B'
                        # )
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
                        filtered_df['explore_further'] = "<a href='" + '' + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>"

                        # Save to csv
                        csv_df = filtered_df[['publisher', 'title', 'article_url', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label'] + categories]
                        csv_df.columns = ['Publisher', 'Title', 'Article URL', 'Date Published (YYYY-MM-DD)', 'Topic', 'Bias Rating'] + [c.upper() for c in categories]
                        csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_df.to_csv(index=False, encoding='utf-8'))

                        # Mapping for specific columns to their new names
                        column_name_map = {
                            'publisher': 'Publisher',
                            'title_label': 'Title Label',
                            'date_published_label_(yyyy-mm-dd)': 'Date Published (YYYY-MM-DD)',
                            'topic': 'Topic',
                            'bias_rating_label': 'Bias Rating',
                            'generalisation': 'G',
                            'prominence': 'O',
                            'negative_behaviour': 'N',
                            'misrepresentation': 'M',
                            'headline_or_imagery': 'H',
                            'explore_further': 'Explore Further'
                        }

                        # Dash
                        filtered_df = filtered_df.sort_values('date_published_label_(yyyy-mm-dd)', ascending=False)[['publisher', 'title_label', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label'] + categories + ['explore_further']]
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
                            # style_data={'textAlign': 'left', 'padding': '5px', 'font-family': 'sans-serif', 'whiteSpace': 'normal', 'height': 'auto'},
                            # style_data_conditional=[
                            #     {
                            #         'if': {
                            #             'filter_query': '{bias_rating_label}="Very Biased"',
                            #             'column_id': ['title_label', 'bias_rating_label']
                            #             },
                            #         'backgroundColor': '#C22625',
                            #         'color': 'white'
                            #     },
                            #     {
                            #         'if': {
                            #             'filter_query': '{bias_rating_label}="Biased"',
                            #             'column_id': ['title_label', 'bias_rating_label']
                            #             },
                            #         'backgroundColor': '#eb8483',
                            #         'color': '#2E2C2B'
                            #     }
                            # ],
                            style_cell={'textAlign': 'center', 'padding': '5px', 'font-family':'sans-serif', 'whiteSpace': 'normal', 'height': 'auto'},
                            style_cell_conditional=[
                                {'if': {'column_id': 'publisher'}, 'width': '150px'},
                                {'if': {'column_id': 'title_label'}, 'width': '300px', 'textAlign': 'left'},
                                {'if': {'column_id': 'date_published_label_(yyyy-mm-dd)'}, 'width': '150px'},
                                {'if': {'column_id': 'topic'}, 'width': '200px', 'textAlign': 'left'},
                                {'if': {'column_id': 'bias_rating_label'}, 'width': '150px'},
                                {'if': {'column_id': 'generalisation'}, 'width': '50px'},
                                {'if': {'column_id': 'prominence'}, 'width': '50px'},
                                {'if': {'column_id': 'negative_behaviour'}, 'width': '50px'},
                                {'if': {'column_id': 'misrepresentation'}, 'width': '50px'},
                                {'if': {'column_id': 'headline_or_imagery'}, 'width': '50px'},
                                {'if': {'column_id': 'explore_further'}, 'width': '200px'}
                            ]
                        )

                if id == 'export-button1b':
                    return [title], table, {'fontSize':14, 'display': 'block'}, {'fontSize':14, 'display': 'block', 'margin-left': '10px'}, csv_string

                return [title], table, {'fontSize':14, 'display': 'block'}, {'fontSize':14, 'display': 'block', 'margin-left': '10px'}, csv_string

            elif id in ['chart1b-datepickerrange', 'chart1b-topic-dropdown', 'chart1b-publisher-dropdown', 'chart1b-bias-rating-dropdown', 'chart1b-bias-category-dropdown', 'chart1b-color-toggle', 'clear-button1b']:
                return [], None, {'display': 'none'}, {'display': 'none'}, ''

        else:
            return [], None, {'display': 'none'}, {'display': 'none'}, ''
        