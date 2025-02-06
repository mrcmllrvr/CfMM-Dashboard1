# compare_chart2.py
# This file contains only the comparison logic for Charts 2A and 2B.

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

# # (1) Import results of bias model and topic/location model
# # drive.mount('/content/gdrive', force_remount=True)
# # dir = 'gdrive/MyDrive/CfMM/data/'

# # with open(dir+f'df_topic_and_loc.pkl', 'rb') as pickle_file:
# #   df_topic_and_loc = pd.compat.pickle_compat.load(pickle_file)

# # with open(dir+f'df_bias.pkl', 'rb') as pickle_file:
# #   df_bias = pd.compat.pickle_compat.load(pickle_file)

# # Import datasets if from local
# # df_dummy = pd.read_pickle(r"df_dummy.pkl")
# df_topic_and_loc = pd.read_pickle(r"df_topic_and_loc.pkl")
# df_bias = pd.read_pickle(r"df_bias.pkl")

# # (2) Join
# df = df_topic_and_loc.merge(df_bias, on='url')

from google.cloud import bigquery
import pandas as pd

# Initialize BigQuery client
client = bigquery.Client()

# Define full table IDs (format: project.dataset.table)
articles_table = "snappy-cosine-449202-k9.cfmm1.articles"
topics_table = "snappy-cosine-449202-k9.cfmm1.topic_list"
article_analyses_table = "snappy-cosine-449202-k9.cfmm1.article_analyses"


# SQL query to join both tables using article_id
query = """
    SELECT *
    FROM `snappy-cosine-449202-k9.cfmm1.articles` a
    LEFT JOIN `snappy-cosine-449202-k9.cfmm1.topic_list` t ON a.article_id = t.article_id
    LEFT JOIN `snappy-cosine-449202-k9.cfmm1.article_analyses` aa ON a.article_id = aa.article_id
"""

# Run the query and convert results to a Pandas DataFrame
df_corpus = client.query(query).to_dataframe()
df_corpus['publish_date'] = pd.to_datetime(df_corpus['publish_date'], errors='coerce')
df_corpus['bias_rating'] = pd.to_numeric(df_corpus['bias_rating'], errors='coerce')
df_corpus['url'] = pd.to_numeric(df_corpus['url'], errors='coerce')


# (3) Get relevant parameters
# # If year to date:
start_date = df_corpus['publish_date'].min()
end_date = df_corpus['publish_date'].max()

# # If today only:
# start_date = df_corpus['publish_date'].max()
# end_date = df_corpus['publish_date'].max()

unique_publishers = sorted(df_corpus['publisher'].unique())
unique_topics = df_corpus['topic_name'].explode().dropna().unique()





# Initialize the Dash application
stylesheets = [
    dbc.themes.BOOTSTRAP,
    dbc.icons.BOOTSTRAP,
    '/assets/custom_compare_chart.css'
]
app = dash.Dash(__name__, external_stylesheets=stylesheets)

# Define the comparison layout for Chart 2A and Chart 2B
def create_layout():
    layout = html.Div(style={'justify-content': 'center', 'backgroundColor': '#ffffff'}, className='row', children=[
        html.H3(children="What are the topics of the biased/very biased article during the selected period?", style={'textAlign': 'center', 'font-weight':'bold', 'margin-bottom': '30px'}),

        # Chart 2A vs Chart 2B
        html.Div([

            # All elements for Chart 2A
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
                    id='chart2a-datepickerrange',
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
                id='chart2a-publisher-dropdown',
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
                id='chart2a-bias-rating-dropdown',
                options=[
                    {'label': 'Not Biased', 'value': 0},
                    {'label': 'Inconclusive', 'value':-1},
                    {'label': 'Biased', 'value': 1},
                    {'label': 'Very Biased', 'value': 2},
                    
                ],
                value = [1, 2],
                placeholder='Select Overall Bias Score',
                multi=True,
                clearable=True,
                style = {'width': '70%'}
                )
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(
                    [
                        html.I(className="bi-boxes", style={'vertical-align': 'middle', 'font-size': '1.5em'}),
                        html.Span(' Category of Bias:', style={'vertical-align': 'middle'})
                    ],
                    style={'font-weight': 'bold', 'width': '40%'}
                ),
                dcc.Dropdown(
                id='chart2a-bias-category-dropdown',
                options=[
                    {'label': 'Generalisation', 'value': 'generalization_tag'},
                    {'label': 'Prominence', 'value': 'omit_due_prominence_tag'},
                    {'label': 'Negative Behaviour', 'value': 'negative_aspects_tag'},
                    {'label': 'Misrepresentation', 'value': 'misrepresentation_tag'},
                    {'label': 'Headline or Imagery', 'value': 'headline_bias_tag'},
                ],
                placeholder='Select Category of Bias',
                multi=True,
                clearable=True,
                style = {'width': '70%'}
                )
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            # Graph for displaying the top topics
            dcc.Graph(id='top-topics-bar-chart-2a', style = {'margin-bottom': '30px'}),

            # Table for displaying the top topics
            html.Div(id='table2a-title', style={'fontColor': '#2E2C2B', 'margin-bottom': '20px'}),
            html.Div(id='table2a'),
            html.Div([
                dbc.Button('Clear Table', id='clear-button2a', style = {'display': 'none', 'white-space': 'nowrap', 'margin-left': '2%', 'width': '40%', 'background-color': '#C22625', 'border-radius': '8px', 'border': 'none'}),
                dbc.Button('Export to CSV', id='export-button2a', style = {'display': 'none', 'white-space': 'nowrap', 'margin-left': '2%', 'width': '40%', 'background-color': '#C22625', 'border-radius': '8px', 'border': 'none'}
                        )
            ], style={'display':'flex', 'margin-top': '10px', 'align-items': 'center'}),

        ],
        style={'width': '48%', 'display': 'inline-block', 'backgroundColor': 'white', 'border': '2px solid #d3d3d3', 'border-radius': '8px', 'padding': '10px', 'margin': '5px'}),


        # All elements for Chart 2B
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
                    id='chart2b-datepickerrange',
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
                id='chart2b-publisher-dropdown',
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
                id='chart2b-bias-rating-dropdown',
                options=[
                    {'label': 'Not Biased', 'value': 0},
                    {'label': 'Inconclusive', 'value':-1},
                    {'label': 'Biased', 'value': 1},
                    {'label': 'Very Biased', 'value': 2},
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
                        html.I(className="bi-boxes", style={'vertical-align': 'middle', 'font-size': '1.5em'}),
                        html.Span(' Category of Bias:', style={'vertical-align': 'middle'})
                    ],
                    style={'font-weight': 'bold', 'width': '40%'}
                ),                dcc.Dropdown(
                id='chart2b-bias-category-dropdown',
                options=[
                    {'label': 'Generalisation', 'value': 'generalization_tag'},
                    {'label': 'Prominence', 'value': 'omit_due_prominence_tag'},
                    {'label': 'Negative Behaviour', 'value': 'negative_aspects_tag'},
                    {'label': 'Misrepresentation', 'value': 'misrepresentation_tag'},
                    {'label': 'Headline or Imagery', 'value': 'headline_bias_tag'},
                ],
                placeholder='Select Category of Bias',
                multi=True,
                clearable=True,
                style = {'width': '70%'})
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            # Graph for displaying the top topics
            dcc.Graph(id='top-topics-bar-chart-2b', style = {'margin-bottom': '30px'}),

            # Table for displaying the top topics
            html.Div(id='table2b-title', style={'fontColor': '#2E2C2B', 'margin-bottom': '20px'}),
            html.Div(id='table2b'),
            html.Div([
                dbc.Button('Clear Table', id='clear-button2b', style = {'display': 'none', 'white-space': 'nowrap', 'margin-left': '2%', 'width': '40%', 'background-color': '#C22625', 'border-radius': '8px', 'border': 'none'}),
                dbc.Button('Export to CSV', id='export-button2b', style = {'display': 'none', 'white-space': 'nowrap', 'margin-left': '2%', 'width': '40%', 'background-color': '#C22625', 'border-radius': '8px', 'border': 'none'})
            ], style={'display':'flex', 'margin-top': '10px', 'align-items': 'center'}),

        ],
        style={'width': '48%', 'display': 'inline-block', 'backgroundColor': 'white', 'border': '2px solid #d3d3d3', 'border-radius': '8px', 'padding': '10px', 'margin': '5px'}),

    ])

    return layout


def register_callbacks(app):
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
            filtered_df = filtered_df[(filtered_df['publish_date']>=start_date) & (filtered_df['publish_date']<=end_date)]
        if selected_publishers:
            filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
        if selected_bias_ratings:
            filtered_df = filtered_df[filtered_df['bias_rating'].isin(selected_bias_ratings)]
        if selected_bias_categories:
            filtered_df[selected_bias_categories] = filtered_df[selected_bias_categories].apply(pd.to_numeric, errors='coerce')
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
            filtered_df_exploded = filtered_df[['url', 'topic_name']].explode('topic_name')
            topic_counts = filtered_df_exploded.groupby('topic_name', observed=True).size().sort_values(ascending=False)
            total_articles = filtered_df_exploded['url'].nunique()

            # Predefine colors for the top 5 topics
            top_colors = ['#4185A0', '#AA4D71', '#B85C3B', '#C5BE71', '#7658A0']
            gray_color = '#CAC6C2' # Add gray color for the remaining topics

            # Create bars for the bar chart
            data = []
            for i, (topic, count) in enumerate(topic_counts.items()):
                percentage_of_total = (count / total_articles) * 100 if total_articles > 0 else 0
                tooltip_text = (
                    # f"<b>Topic: </b>{topic}<br>"
                    f"<b>Count: </b>{count}<br>"
                    f"<b>Proportion: </b>{percentage_of_total:.1%}<br>"
                    # f"This accounts for <b>{count/total_articles:.2%}%</b> of the total available articles in the current selection.<br>"
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
                title="",
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
            filtered_df = filtered_df[(filtered_df['publish_date']>=start_date) & (filtered_df['publish_date']<=end_date)]
        if selected_publishers:
            filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
        if selected_bias_ratings:
            filtered_df = filtered_df[filtered_df['bias_rating'].isin(selected_bias_ratings)]
        if selected_bias_categories:
            filtered_df[selected_bias_categories] = filtered_df[selected_bias_categories].apply(pd.to_numeric, errors='coerce')
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
            filtered_df_exploded = filtered_df[['url', 'topic_name']].explode('topic_name')
            topic_counts = filtered_df_exploded.groupby('topic_name', observed=True).size().sort_values(ascending=False)
            total_articles = filtered_df_exploded['url'].nunique()

            # Predefine colors for the top 5 topics
            top_colors = ['#4185A0', '#AA4D71', '#B85C3B', '#C5BE71', '#7658A0']
            gray_color = '#CAC6C2' # Add gray color for the remaining topics

            # Create bars for the bar chart
            data = []
            for i, (topic, count) in enumerate(topic_counts.items()):
                percentage_of_total = (count / total_articles) * 100 if total_articles > 0 else 0
                tooltip_text = (
                    # f"<b>Topic: </b>{topic}<br>"
                    f"<b>Count: </b>{count}<br>"
                    f"<b>Proportion: </b>{percentage_of_total:.1%}<br>"
                    # f"This accounts for <b>{count/total_articles:.2%}%</b> of the total available articles in the current selection.<br>"
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
                title="",
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
                    filtered_df = filtered_df[(filtered_df['publish_date'] >= start_date) & (filtered_df['publish_date'] <= end_date)]
                if selected_publishers:
                    filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
                if selected_bias_ratings:
                    filtered_df = filtered_df[filtered_df['bias_rating'].isin(selected_bias_ratings)]
                if selected_bias_categories:
                    filtered_df[selected_bias_categories] = filtered_df[selected_bias_categories].apply(pd.to_numeric, errors='coerce')
                    filtered_df = filtered_df[filtered_df[selected_bias_categories].sum(axis=1) > 0]

                if (clickData is not None) or (clickData is None and id == 'export-button2a'):
                    topic = str(clickData['points'][0]['label'])

                    # Table title
                    main_title = f'Showing all articles about <b>{topic}</b>'
                    keys = '<b>Legend: G =</b> Generalisation, <b>O =</b> Omit Due Prominence, <b>N =</b> Negative Behaviour, <b>M =</b> Misrepresentation, <b>H =</b> Headline'
                    title_html = f'{main_title}<br>{keys}'
                        
                    title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(title_html)


                    # Apply formatting
                    filtered_df['topic_name'] = filtered_df['topic_name'].fillna("")
                    filtered_df = filtered_df[filtered_df['topic_name'].str.contains('|'.join([topic]))]
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
                    filtered_df['url'] = filtered_df['url'].astype(str)
                    filtered_df['color'] = filtered_df['color'].astype(str)
                    filtered_df['headline'] = filtered_df['headline'].astype(str)
                    
                    filtered_df['title_label'] = "<a href='" + filtered_df['url'] + "' target='_blank' style='color:" + filtered_df['color'] + ";'>" + filtered_df['headline'] + "</a>"
                    filtered_df['bias_rating_label'] = np.select(
                        [
                            filtered_df['bias_rating'] == 0,
                            filtered_df['bias_rating'] == -1,
                            filtered_df['bias_rating'] == 1,
                            filtered_df['bias_rating'] == 2
                        ],
                        [
                            'Not Biased',
                            'Inconclusive',
                            'Biased',
                            'Very Biased'
                        ],
                        default='Unknown'
                    )
                    categories = ['generalization_tag', 'omit_due_prominence_tag', 'negative_aspects_tag', 'misrepresentation_tag', 'headline_bias_tag']
                    for category in categories:
                        filtered_df[category] = np.where(filtered_df[category] == 1, 'Y', 'N')
                    filtered_df['publish_date'] = pd.to_datetime(filtered_df['publish_date'], errors='coerce') # Convert to datetime
                    filtered_df['publish_date_label_(yyyy-mm-dd)'] = filtered_df['publish_date'].dt.date
                    filtered_df['explore_further'] = "<a href='" + '' + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>"

                    # Save to csv
                    csv_df = filtered_df[['publisher', 'headline', 'url', 'publish_date_label_(yyyy-mm-dd)', 'topic_name', 'bias_rating_label'] + categories]
                    csv_df.columns = ['Publisher', 'headline', 'Article URL', 'Date Published (YYYY-MM-DD)', 'Topic', 'Bias Rating'] + [c.upper() for c in categories]
                    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_df.to_csv(index=False, encoding='utf-8'))

                    # Mapping for specific columns to their new names
                    column_name_map = {
                        'publisher': 'Publisher',
                        'title_label': 'Title Label',
                        'publish_date_label_(yyyy-mm-dd)': 'Date Published (YYYY-MM-DD)',
                        'topic_name': 'Topic',
                        'bias_rating_label': 'Bias Rating',
                        'generalization_tag': 'G',
                        'omit_due_prominence_tag': 'O',
                        'negative_aspects_tag': 'N',
                        'misrepresentation_tag': 'M',
                        'headline_bias_tag': 'H',
                        'explore_further': 'Explore Further'
                    }

                    # Dash
                    filtered_df = filtered_df.sort_values('publish_date_label_(yyyy-mm-dd)', ascending=False)[['publisher', 'title_label', 'publish_date_label_(yyyy-mm-dd)', 'topic_name', 'bias_rating_label'] + categories + ['explore_further']]
                    table = dash_table.DataTable(
                        css=[dict(selector="p", rule="margin:0; text-align:left")],
                        columns=[{'id': x, 'name': column_name_map.get(x, x.replace('_', ' ').title()), 'presentation': 'markdown'} if 'headline' in x or 'explore' in x else {'id': x, 'name': column_name_map.get(x, x.replace('_', ' ').replace('label', '').title().replace('Or', 'or').replace('Yyyy-Mm-Dd', 'yyyy-mm-dd'))} for x in filtered_df.columns],
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
                            {'if': {'column_id': 'publish_date_label_(yyyy-mm-dd)'}, 'width': '150px'},
                            {'if': {'column_id': 'topic_name'}, 'width': '200px', 'textAlign': 'left'},
                            {'if': {'column_id': 'bias_rating_label'}, 'width': '150px'},
                            {'if': {'column_id': 'generalization_tag'}, 'width': '50px'},
                            {'if': {'column_id': 'omit_due_prominence_tag'}, 'width': '50px'},
                            {'if': {'column_id': 'negative_aspects_tag'}, 'width': '50px'},
                            {'if': {'column_id': 'misrepresentation_tag'}, 'width': '50px'},
                            {'if': {'column_id': 'headline_bias_tag'}, 'width': '50px'},
                            {'if': {'column_id': 'explore_further'}, 'width': '200px'}
                        ]
                    )

                if id == 'export-button2a':
                    return [title], table, {'display': 'block', 'white-space': 'nowrap', 'width': '10%', 'background-color': '#C22625', 'border-radius': '8px', 'border': 'none'},  {'display': 'block', 'white-space': 'nowrap', 'margin-left': '1%', 'width': '10%', 'background-color': '#C22625', 'border-radius': '8px', 'border': 'none'}, csv_string

                return [title], table, {'display': 'block', 'white-space': 'nowrap', 'width': '10%', 'background-color': '#C22625', 'border-radius': '8px', 'border': 'none'},  {'display': 'block', 'white-space': 'nowrap', 'margin-left': '1%', 'width': '10%', 'background-color': '#C22625', 'border-radius': '8px', 'border': 'none'}, csv_string

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
                    filtered_df = filtered_df[(filtered_df['publish_date'] >= start_date) & (filtered_df['publish_date'] <= end_date)]
                if selected_publishers:
                    filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
                if selected_bias_ratings:
                    filtered_df = filtered_df[filtered_df['bias_rating'].isin(selected_bias_ratings)]
                if selected_bias_categories:
                    filtered_df[selected_bias_categories] = filtered_df[selected_bias_categories].apply(pd.to_numeric, errors='coerce')
                    filtered_df = filtered_df[filtered_df[selected_bias_categories].sum(axis=1) > 0]

                if (clickData is not None) or (clickData is None and id == 'export-button2'):
                    topic = str(clickData['points'][0]['label'])

                    # Table title
                    main_title = f'Showing all articles about <b>{topic}</b>'
                    keys = '<b>Legend: G =</b> Generalisation, <b>O =</b> Omit Due Prominence, <b>N =</b> Negative Behaviour, <b>M =</b> Misrepresentation, <b>H =</b> Headline'
                    title_html = f'{main_title}<br>{keys}'
                        
                    title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(title_html)


                    # Apply formatting
                    filtered_df['topic_name'] = filtered_df['topic_name'].fillna("")
                    filtered_df = filtered_df[filtered_df['topic_name'].str.contains('|'.join([topic]))]
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
                    filtered_df['url'] = filtered_df['url'].astype(str)
                    filtered_df['color'] = filtered_df['color'].astype(str)
                    filtered_df['headline'] = filtered_df['headline'].astype(str)
                    
                    filtered_df['title_label'] = "<a href='" + filtered_df['url'] + "' target='_blank' style='color:" + filtered_df['color'] + ";'>" + filtered_df['headline'] + "</a>"
                    filtered_df['bias_rating_label'] = np.select(
                        [
                            filtered_df['bias_rating'] == 0,
                            filtered_df['bias_rating'] == -1,
                            filtered_df['bias_rating'] == 1,
                            filtered_df['bias_rating'] == 2
                        ],
                        [
                            'Not Biased',
                            'Inconclusive',
                            'Biased',
                            'Very Biased'
                        ],
                        default='Unknown'
                    )
                    categories = ['generalization_tag', 'omit_due_prominence_tag', 'negative_aspects_tag', 'misrepresentation_tag', 'headline_bias_tag']
                    for category in categories:
                        filtered_df[category] = np.where(filtered_df[category] == 1, 'Y', 'N')
                    filtered_df['publish_date'] = pd.to_datetime(filtered_df['publish_date'], errors='coerce') # Convert to datetime
                    filtered_df['publish_date_label_(yyyy-mm-dd)'] = filtered_df['publish_date'].dt.date
                    filtered_df['explore_further'] = "<a href='" + '' + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>"

                    # Save to csv
                    csv_df = filtered_df[['publisher', 'headline', 'url', 'publish_date_label_(yyyy-mm-dd)', 'topic_name', 'bias_rating_label'] + categories]
                    csv_df.columns = ['Publisher', 'headline', 'Article URL', 'Date Published (YYYY-MM-DD)', 'Topic', 'Bias Rating'] + [c.upper() for c in categories]
                    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_df.to_csv(index=False, encoding='utf-8'))

                    # Mapping for specific columns to their new names
                    column_name_map = {
                        'publisher': 'Publisher',
                        'title_label': 'Title Label',
                        'publish_date_label_(yyyy-mm-dd)': 'Date Published (YYYY-MM-DD)',
                        'topic_name': 'Topic',
                        'bias_rating_label': 'Bias Rating',
                        'generalization_tag': 'G',
                        'omit_due_prominence_tag': 'O',
                        'negative_aspects_tag': 'N',
                        'misrepresentation_tag': 'M',
                        'headline_bias_tag': 'H',
                        'explore_further': 'Explore Further'
                    }

                    # Dash
                    filtered_df = filtered_df.sort_values('publish_date_label_(yyyy-mm-dd)', ascending=False)[['publisher', 'title_label', 'publish_date_label_(yyyy-mm-dd)', 'topic_name', 'bias_rating_label'] + categories + ['explore_further']]
                    table = dash_table.DataTable(
                        css=[dict(selector="p", rule="margin:0; text-align:left")],
                        columns=[{'id': x, 'name': column_name_map.get(x, x.replace('_', ' ').title()), 'presentation': 'markdown'} if 'headline' in x or 'explore' in x else {'id': x, 'name': column_name_map.get(x, x.replace('_', ' ').replace('label', '').title().replace('Or', 'or').replace('Yyyy-Mm-Dd', 'yyyy-mm-dd'))} for x in filtered_df.columns],
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
                            {'if': {'column_id': 'publish_date_label_(yyyy-mm-dd)'}, 'width': '150px'},
                            {'if': {'column_id': 'topic_name'}, 'width': '200px', 'textAlign': 'left'},
                            {'if': {'column_id': 'bias_rating_label'}, 'width': '150px'},
                            {'if': {'column_id': 'generalization_tag'}, 'width': '50px'},
                            {'if': {'column_id': 'omit_due_prominence_tag'}, 'width': '50px'},
                            {'if': {'column_id': 'negative_aspects_tag'}, 'width': '50px'},
                            {'if': {'column_id': 'misrepresentation_tag'}, 'width': '50px'},
                            {'if': {'column_id': 'headline_bias_tag'}, 'width': '50px'},
                            {'if': {'column_id': 'explore_further'}, 'width': '200px'}
                        ]
                    )

                if id == 'export-button2b':
                    return [title], table, {'display': 'block', 'white-space': 'nowrap', 'width': '10%', 'background-color': '#C22625', 'border-radius': '8px', 'border': 'none'},  {'display': 'block', 'white-space': 'nowrap', 'margin-left': '1%', 'width': '10%', 'background-color': '#C22625', 'border-radius': '8px', 'border': 'none'}, csv_string

                return [title], table, {'display': 'block', 'white-space': 'nowrap', 'width': '10%', 'background-color': '#C22625', 'border-radius': '8px', 'border': 'none'},  {'display': 'block', 'white-space': 'nowrap', 'margin-left': '1%', 'width': '10%', 'background-color': '#C22625', 'border-radius': '8px', 'border': 'none'}, csv_string

            elif id in ['chart2b-datepickerrange', 'chart2b-publisher-dropdown', 'chart2b-bias-rating-dropdown', 'chart2b-bias-category-dropdown', 'chart2b-color-toggle', 'clear-button2b']:
                return [], None, {'display': 'none'}, {'display': 'none'}, ''

        else:
            return [], None, {'display': 'none'}, {'display': 'none'}, ''
