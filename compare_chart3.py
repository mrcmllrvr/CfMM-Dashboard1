# compare_chart3.py
# This file contains only the comparison logic for Charts 3A and 3B.

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
    '/assets/custom_compare_chart.css'
]
app = dash.Dash(__name__, external_stylesheets=stylesheets)

# Define the comparison layout for Chart 3A and Chart 3B
def create_layout():
    layout = html.Div(style={'justify-content': 'center', 'backgroundColor': '#ffffff'}, className='row', children=[
        html.H3(children="Which overall bias score is highest during the selected period?", style={'textAlign': 'center', 'font-weight':'bold', 'margin-bottom': '30px'}),

        # Chart 3A vs Chart 3B
        html.Div([

            # All elements for Chart 3A
            html.H4("Scenario A", style={'textAlign': 'center', 'margin-bottom':'30px', 'margin-top':'30px'}),

            html.Div([
                html.Label(
                    [
                        html.I(className="bi-calendar-week", style={'vertical-align': 'middle', 'font-size': '1.5em'}),
                        html.Span(' Date Published:', style={'vertical-align': 'middle'})
                    ],
                    style={'font-weight': 'bold', 'width': '40%'}
                ),
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
                    style = {'font-size':'13px', 'width': '70%'}
                )
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(
                    [
                        html.I(className="bi-person-fill", style={'vertical-align': 'middle', 'font-size': '1.5em'}),
                        html.Span(' Publishers:', style={'vertical-align': 'middle'})
                    ],
                    style={'font-weight': 'bold', 'width': '40%'}
                ),
                dcc.Dropdown(
                id='chart3a-publisher-dropdown',
                options=[{'label': publisher, 'value': publisher} for publisher in unique_publishers],
                placeholder='Select Publisher',
                multi=True,
                clearable=True,
                style = {'width': '70%'}
                )
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(
                    [
                        html.I(className="bi-speedometer2", style={'vertical-align': 'middle', 'font-size': '1.5em'}),
                        html.Span(' Overall Bias Score:', style={'vertical-align': 'middle'})
                    ],
                    style={'font-weight': 'bold', 'width': '40%'}
                ),
                dcc.Dropdown(
                    id='chart3a-bias-rating-dropdown',
                    options=[
                        {'label': 'Inconclusive', 'value':-1},
                        {'label': 'Biased', 'value': 1},
                        {'label': 'Very Biased', 'value': 2},
                        {'label': 'Not Biased', 'value': 0},
                    ],
                    value=[1, 2],
                    placeholder='Select Overall Bias Score',
                    multi=True,
                    clearable=True,
                    style = {'width': '70%'}
                )
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(
                    [
                        html.I(className="bi-chat-dots", style={'vertical-align': 'middle', 'font-size': '1.5em'}),
                        html.Span(' Topics:', style={'vertical-align': 'middle'})
                    ],
                    style={'font-weight': 'bold', 'width': '40%'}
                ),
                dcc.Dropdown(
                id='chart3a-topic-dropdown',
                options=[{'label': topic, 'value': topic} for topic in unique_topics],
                placeholder='Select Topic',
                multi=True,
                clearable=True,
                style = {'width': '70%'}
                )
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            # Graph for displaying the top topics
            dcc.Graph(id='top-offending-articles-bar-chart-3a', style = {'margin-bottom': '30px'}),

            # Table for displaying the top topics
            html.Div(id='table3a-title', style={'fontColor': '#2E2C2B', 'margin-bottom': '20px'}),
            html.Div(id='table3a'),
            html.Div([
                dbc.Button('Clear Table', id='clear-button3a', style = {'display': 'none', 'white-space': 'nowrap', 'margin-left': '2%', 'width': '40%', 'background-color': '#C22625', 'border-radius': '8px', 'border': 'none'}),
                dbc.Button('Export to CSV', id='export-button3a', style = {'display': 'none', 'white-space': 'nowrap', 'margin-left': '2%', 'width': '40%', 'background-color': '#C22625', 'border-radius': '8px', 'border': 'none'}
                            )
            ], style={'display':'flex', 'margin-top': '10px', 'align-items': 'center'}),
        ],
        style={'width': '48%', 'display': 'inline-block', 'backgroundColor': 'white', 'border': '2px solid #d3d3d3', 'border-radius': '8px', 'padding': '10px', 'margin': '5px'}),


        # All elements for Chart 3B
        html.Div([
            html.H4("Scenario B", style={'textAlign': 'center', 'margin-bottom':'30px', 'margin-top':'30px'}),

            html.Div([
                html.Label(
                    [
                        html.I(className="bi-calendar-week", style={'vertical-align': 'middle', 'font-size': '1.5em'}),
                        html.Span(' Date Published:', style={'vertical-align': 'middle'})
                    ],
                    style={'font-weight': 'bold', 'width': '40%'}
                ),
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
                    style = {'font-size':'13px', 'width': '70%'}
                )
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(
                    [
                        html.I(className="bi-person-fill", style={'vertical-align': 'middle', 'font-size': '1.5em'}),
                        html.Span(' Publishers:', style={'vertical-align': 'middle'})
                    ],
                    style={'font-weight': 'bold', 'width': '40%'}
                ),
                dcc.Dropdown(
                id='chart3b-publisher-dropdown',
                options=[{'label': publisher, 'value': publisher} for publisher in unique_publishers],
                placeholder='Select Publisher',
                multi=True,
                clearable=True,
                style = {'width': '70%'})
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(
                    [
                        html.I(className="bi-speedometer2", style={'vertical-align': 'middle', 'font-size': '1.5em'}),
                        html.Span(' Overall Bias Score:', style={'vertical-align': 'middle'})
                    ],
                    style={'font-weight': 'bold', 'width': '40%'}
                ),
                dcc.Dropdown(
                    id='chart3b-bias-rating-dropdown',
                    options=[
                        {'label': 'Inconclusive', 'value':-1},
                        {'label': 'Biased', 'value': 1},
                        {'label': 'Very Biased', 'value': 2},
                        {'label': 'Not Biased', 'value': 0},
                    ],
                    value=[1, 2],
                    placeholder='Select Overall Bias Score',
                    multi=True,
                    clearable=True,
                    style = {'width': '70%'}
                )
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(
                    [
                        html.I(className="bi-chat-dots", style={'vertical-align': 'middle', 'font-size': '1.5em'}),
                        html.Span(' Topics:', style={'vertical-align': 'middle'})
                    ],
                    style={'font-weight': 'bold', 'width': '40%'}
                ),
                dcc.Dropdown(
                id='chart3b-topic-dropdown',
                options=[{'label': topic, 'value': topic} for topic in unique_topics],
                placeholder='Select Topic',
                multi=True,
                clearable=True,
                style = {'width': '70%'})
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            # Graph for displaying the top topics
            dcc.Graph(id='top-offending-articles-bar-chart-3b', style = {'margin-bottom': '30px'}),

            # Table for displaying the top topics
            html.Div(id='table3b-title', style={'fontColor': '#2E2C2B', 'margin-bottom': '20px'}),
            html.Div(id='table3b'),
            html.Div([
                dbc.Button('Clear Table', id='clear-button3b', style = {'display': 'none', 'white-space': 'nowrap', 'margin-left': '2%', 'width': '40%', 'background-color': '#C22625', 'border-radius': '8px', 'border': 'none'}),
                dbc.Button('Export to CSV', id='export-button3b', style = {'display': 'none', 'white-space': 'nowrap', 'margin-left': '2%', 'width': '40%', 'background-color': '#C22625', 'border-radius': '8px', 'border': 'none'})
            ], style={'display':'flex', 'margin-top': '10px', 'align-items': 'center'}),
        ],
        style={'width': '48%', 'display': 'inline-block', 'backgroundColor': 'white', 'border': '2px solid #d3d3d3', 'border-radius': '8px', 'padding': '10px', 'margin': '5px'}),

    ])

    return layout


def register_callbacks(app):
    # Callback for Chart 3
    @app.callback(
        Output('top-offending-articles-bar-chart-3a', 'figure'),
        [
            Input('chart3a-datepickerrange', 'start_date'),
            Input('chart3a-datepickerrange', 'end_date'),
            Input('chart3a-publisher-dropdown', 'value'),
            Input('chart3a-bias-rating-dropdown', 'value'),
            Input('chart3a-topic-dropdown', 'value')
        ]
    )

    def update_chart3a(selected_start_date, selected_end_date, selected_publishers, selected_bias_ratings, selected_topics):
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
            categories = ['generalisation', 'prominence', 'negative_behaviour', 'misrepresentation', 'headline_or_imagery']
            labels = ['Generalisation', 'Omit Due Prominence', 'Negative Behaviour', 'Misrepresentation', 'Headline']
            label_map = dict(zip(categories, labels))

            filtered_df = filtered_df[['article_url']+categories].melt(id_vars='article_url')
            filtered_df = filtered_df.sort_values(['article_url', 'variable'])
            filtered_df.columns = ['article_url', 'bias_rating', 'count']
            filtered_df['bias_rating'] = filtered_df['bias_rating'].map(label_map)
            filtered_df['bias_rating'] = pd.Categorical(filtered_df['bias_rating'], labels, ordered=True)
            bias_counts = filtered_df.groupby('bias_rating', observed=True)['count'].sum()
            total_articles = filtered_df[filtered_df['count']>=1]['article_url'].nunique()
            
            # Predefine colors for the top 5 topics
            colors = ['#4185A0', '#AA4D71', '#B85C3B', '#C5BE71', '#7658A0']
            color_map = dict(zip(labels, colors))

            # Create bars for the bar chart
            data = []
            for (bias, count) in bias_counts.items():
                tooltip_text = (
                    # f"<b>Overall Bias Score: </b>{bias}<br>"
                    f"<b>Count:</b> {count}<br>"
                    f"<b>Proportion:</b> {count/total_articles:.1%} (Among {total_articles} articles that committed<br>at least 1 category of bias, {count/total_articles:.1%} are {bias}.)"
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
                title='',
                xaxis=dict(title='Number of Articles'),
                yaxis=dict(title='Overall Bias Score', tickmode='array', tickvals=list(range(len(bias_counts))), ticktext=bias_counts.index.tolist()),
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


    # Callback for Chart 3B
    @app.callback(
        Output('top-offending-articles-bar-chart-3b', 'figure'),
        [
            Input('chart3b-datepickerrange', 'start_date'),
            Input('chart3b-datepickerrange', 'end_date'),
            Input('chart3b-publisher-dropdown', 'value'),
            Input('chart3b-bias-rating-dropdown', 'value'),
            Input('chart3b-topic-dropdown', 'value')
        ]
    )

    def update_chart3b(selected_start_date, selected_end_date, selected_publishers, selected_bias_ratings, selected_topics):
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
            categories = ['generalisation', 'prominence', 'negative_behaviour', 'misrepresentation', 'headline_or_imagery']
            labels = ['Generalisation', 'Omit Due Prominence', 'Negative Behaviour', 'Misrepresentation', 'Headline']
            label_map = dict(zip(categories, labels))

            filtered_df = filtered_df[['article_url']+categories].melt(id_vars='article_url')
            filtered_df = filtered_df.sort_values(['article_url', 'variable'])
            filtered_df.columns = ['article_url', 'bias_rating', 'count']
            filtered_df['bias_rating'] = filtered_df['bias_rating'].map(label_map)
            filtered_df['bias_rating'] = pd.Categorical(filtered_df['bias_rating'], labels, ordered=True)
            bias_counts = filtered_df.groupby('bias_rating', observed=True)['count'].sum()
            total_articles = filtered_df[filtered_df['count']>=1]['article_url'].nunique()
            
            # Predefine colors for the top 5 topics
            colors = ['#4185A0', '#AA4D71', '#B85C3B', '#C5BE71', '#7658A0']
            color_map = dict(zip(labels, colors))

            # Create bars for the bar chart
            data = []
            for (bias, count) in bias_counts.items():
                tooltip_text = (
                    # f"<b>Overall Bias Score: </b>{bias}<br>"
                    f"<b>Count:</b> {count}<br>"
                    f"<b>Proportion:</b> {count/total_articles:.1%} (Among {total_articles} articles that committed<br>at least 1 category of bias, {count/total_articles:.1%} are {bias}.)"
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
                title='',
                xaxis=dict(title='Number of Articles'),
                yaxis=dict(title='Overall Bias Score', tickmode='array', tickvals=list(range(len(bias_counts))), ticktext=bias_counts.index.tolist()),
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
            Input('chart3a-bias-rating-dropdown', 'value'),
            Input('chart3a-topic-dropdown', 'value'),
            Input('top-offending-articles-bar-chart-3a', 'clickData'),
            Input('clear-button3a', 'n_clicks')
        ]
    )

    def update_table3a(selected_start_date, selected_end_date, selected_publishers, selected_bias_ratings, selected_topics, clickData, n_clicks):
        triggered = dash.callback_context.triggered
        topics = ''

        if triggered:
            id = triggered[0]['prop_id'].split('.')[0]

            if id in ['top-offending-articles-bar-chart-3a', 'export-button3a']:
                filtered_df = df_corpus.copy()

                # Apply filters for dates, publishers, and other criteria
                if (selected_start_date is not None) & (selected_end_date is not None):
                    start_date = pd.to_datetime(str(selected_start_date))
                    end_date = pd.to_datetime(str(selected_end_date))
                    filtered_df = filtered_df[(filtered_df['date_published'] >= start_date) & (filtered_df['date_published'] <= end_date)]
                if selected_publishers:
                    filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
                if selected_bias_ratings:
                    filtered_df = filtered_df[filtered_df['bias_rating'].isin(selected_bias_ratings)]
                if selected_topics:
                    filtered_df = filtered_df[filtered_df['topic'].str.contains('|'.join(selected_topics))]
                    topics = 'having any of the selected topics'

                # label_map = {
                #     -1: 'Inconclusive',
                #     0: 'Not Biased',
                #     1: 'Biased',
                #     2: 'Very Biased'
                # }
                # filtered_df['bias_rating_label'] = filtered_df['bias_rating'].map(label_map)
                # filtered_df['bias_rating_label'] = pd.Categorical(filtered_df['bias_rating_label'], categories=['Inconclusive', 'Not Biased', 'Biased', 'Very Biased'], ordered=True)

                if (clickData is not None) or (clickData is None and id == 'export-button3a'):
                    bias = str(clickData['points'][0]['label'])

                    # Ensure this category is correctly mapped
                    category_map = {
                        'Generalisation': 'generalisation',
                        'Omit Due Prominence': 'prominence',
                        'Negative Behaviour': 'negative_behaviour',
                        'Misrepresentation': 'misrepresentation',
                        'Headline': 'headline_or_imagery'
                    }

                    # Apply the bias category filter
                    filtered_df = filtered_df[filtered_df[category_map[bias]] > 0]

                    # Table title
                    main_title = f'Showing all articles that were rated <b>{bias}</b> by the model.'
                    keys = '<b>Legend: G =</b> Generalisation, <b>O =</b> Omit Due Prominence, <b>N =</b> Negative Behaviour, <b>M =</b> Misrepresentation, <b>H =</b> Headline'
                    title_html = f'{main_title}<br>{keys}'
                    
                    title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(title_html)

                    # Apply formatting
                    # filtered_df = filtered_df[filtered_df['bias_rating_label'] == bias]
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
                        'title_label': 'Title',
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

                    # Dash Table
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

                if id == 'export-button3a':
                    return [title], table, {'display': 'block', 'white-space': 'nowrap', 'width': '10%', 'background-color': '#C22625', 'border-radius': '8px', 'border': 'none'},  {'display': 'block', 'white-space': 'nowrap', 'margin-left': '1%', 'width': '10%', 'background-color': '#C22625', 'border-radius': '8px', 'border': 'none'}, csv_string

                return [title], table, {'display': 'block', 'white-space': 'nowrap', 'width': '10%', 'background-color': '#C22625', 'border-radius': '8px', 'border': 'none'},  {'display': 'block', 'white-space': 'nowrap', 'margin-left': '1%', 'width': '10%', 'background-color': '#C22625', 'border-radius': '8px', 'border': 'none'}, csv_string

            elif id in ['chart3a-datepickerrange', 'chart3a-publisher-dropdown', 'chart3a-bias-rating-dropdown', 'chart3a-topic-dropdown', 'clear-button3a']:
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
            Input('chart3b-bias-rating-dropdown', 'value'),
            Input('chart3b-topic-dropdown', 'value'),
            Input('top-offending-articles-bar-chart-3b', 'clickData'),
            Input('clear-button3b', 'n_clicks')
        ]
    )

    def update_table3b(selected_start_date, selected_end_date, selected_publishers, selected_bias_ratings, selected_topics, clickData, n_clicks):
        triggered = dash.callback_context.triggered
        topics = ''

        if triggered:
            id = triggered[0]['prop_id'].split('.')[0]

            if id in ['top-offending-articles-bar-chart-3b', 'export-button3b']:
                filtered_df = df_corpus.copy()

                # Apply filters for dates, publishers, and other criteria
                if (selected_start_date is not None) & (selected_end_date is not None):
                    start_date = pd.to_datetime(str(selected_start_date))
                    end_date = pd.to_datetime(str(selected_end_date))
                    filtered_df = filtered_df[(filtered_df['date_published'] >= start_date) & (filtered_df['date_published'] <= end_date)]
                if selected_publishers:
                    filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
                if selected_bias_ratings:
                    filtered_df = filtered_df[filtered_df['bias_rating'].isin(selected_bias_ratings)]
                if selected_topics:
                    filtered_df = filtered_df[filtered_df['topic'].str.contains('|'.join(selected_topics))]
                    topics = 'having any of the selected topics'

                # label_map = {
                #     -1: 'Inconclusive',
                #     0: 'Not Biased',
                #     1: 'Biased',
                #     2: 'Very Biased'
                # }
                # filtered_df['bias_rating_label'] = filtered_df['bias_rating'].map(label_map)
                # filtered_df['bias_rating_label'] = pd.Categorical(filtered_df['bias_rating_label'], categories=['Inconclusive', 'Not Biased', 'Biased', 'Very Biased'], ordered=True)

                if (clickData is not None) or (clickData is None and id == 'export-button3b'):
                    bias = str(clickData['points'][0]['label'])

                    # Ensure this category is correctly mapped
                    category_map = {
                        'Generalisation': 'generalisation',
                        'Omit Due Prominence': 'prominence',
                        'Negative Behaviour': 'negative_behaviour',
                        'Misrepresentation': 'misrepresentation',
                        'Headline': 'headline_or_imagery'
                    }

                    # Apply the bias category filter
                    filtered_df = filtered_df[filtered_df[category_map[bias]] > 0]

                    # Table title
                    main_title = f'Showing all articles that were rated <b>{bias}</b> by the model.'
                    keys = '<b>Legend: G =</b> Generalisation, <b>O =</b> Omit Due Prominence, <b>N =</b> Negative Behaviour, <b>M =</b> Misrepresentation, <b>H =</b> Headline'
                    title_html = f'{main_title}<br>{keys}'
                    
                    title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(title_html)

                    # Apply formatting
                    # filtered_df = filtered_df[filtered_df['bias_rating_label'] == bias]
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
                        'title_label': 'Title',
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

                    # Dash Table
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

                if id == 'export-button3b':
                    return [title], table, {'display': 'block', 'white-space': 'nowrap', 'width': '10%', 'background-color': '#C22625', 'border-radius': '8px', 'border': 'none'},  {'display': 'block', 'white-space': 'nowrap', 'margin-left': '1%', 'width': '10%', 'background-color': '#C22625', 'border-radius': '8px', 'border': 'none'}, csv_string

                return [title], table, {'display': 'block', 'white-space': 'nowrap', 'width': '10%', 'background-color': '#C22625', 'border-radius': '8px', 'border': 'none'},  {'display': 'block', 'white-space': 'nowrap', 'margin-left': '1%', 'width': '10%', 'background-color': '#C22625', 'border-radius': '8px', 'border': 'none'}, csv_string

            elif id in ['chart3b-datepickerrange', 'chart3b-publisher-dropdown', 'chart3b-bias-rating-dropdown', 'chart3b-topic-dropdown', 'clear-button3b']:
                return [], None, {'display': 'none'}, {'display': 'none'}, ''

        else:
            return [], None, {'display': 'none'}, {'display': 'none'}, ''
