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
stylesheets = [dbc.themes.FLATLY] # 'https://codepen.io/chriddyp/pen/bWLwgP.css'
app = dash.Dash(__name__, external_stylesheets=stylesheets)

# Define the comparison layout for Chart 3A and Chart 3B
def create_layout():
    layout = html.Div(style={'justify-content': 'center'}, className='row', children=[
        html.H1(children="Top Offending Articles", style={'textAlign': 'center'}),

        # Chart 3A vs Chart 3B
        html.Div([

            # All elements for Chart 3A
            html.H2("A", style={'textAlign': 'center'}),

            html.Div([
                html.Label(['Date Published:'], style={'font-weight': 'bold', 'width': '20%'}),
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
                html.Label(['Publishers:'], style={'font-weight': 'bold', 'width': '20%'}),
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
                html.Label(['Category of Bias:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart3a-bias-category-dropdown',
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
                style = {'width': '70%'}
                )
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Topics:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart3a-topic-dropdown',
                options=[{'label': topic, 'value': topic} for topic in unique_topics],
                placeholder='Select Topic',
                multi=True,
                clearable=True,
                style = {'width': '70%'}
                )
            ], style={'display':'flex', 'margin-bottom':'50px', 'align-items': 'center'}),

            # Graph for displaying the top topics
            dcc.Graph(id='top-offending-articles-bar-chart-3a', style = {'margin-bottom': '50px'}),

            # Table for displaying the top topics
            html.Div(id='table3a-title', style={'fontSize': 20, 'fontColor': '#2E2C2B', 'margin-bottom': '20px'}),
            html.Div(id='table3a'),
            html.Div([
                dbc.Button('Clear Table', id='clear-button3a', style = {'display': 'none'}),
                dbc.Button('Export to CSV', id='export-button3a', style = {'display': 'none'}
                            )
            ], style={'display':'flex', 'margin-top': '10px', 'align-items': 'center'}),
        ],
        style={'width': '48%', 'display': 'inline-block', 'border': '1px solid black'}),


        # All elements for Chart 3B
        html.Div([
            html.H2("B", style={'textAlign': 'center'}),

            html.Div([
                html.Label(['Date Published:'], style={'font-weight': 'bold', 'width': '20%'}),
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
                html.Label(['Publishers:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart3b-publisher-dropdown',
                options=[{'label': publisher, 'value': publisher} for publisher in unique_publishers],
                placeholder='Select Publisher',
                multi=True,
                clearable=True,
                style = {'width': '70%'})
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Category of Bias:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart3b-bias-category-dropdown',
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
                style = {'width': '70%'})
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Topics:'], style={'font-weight': 'bold', 'width': '20%'}),
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
        style={'width': '48%', 'display': 'inline-block', 'border': '1px solid black'}),
    ])

    return layout


def register_callbacks(app):
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
                    # f"<b>Bias Rating: </b>{bias}<br>"
                    f"<b>Count: </b>{count}<br>"
                    f"<b>Proportion: </b>{count/total_articles:.1%}"
                    # f"This accounts for <b>{count/total_articles:.2%}%</b> of the total available articles in the current selection.<br>"
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
                title='<b>Which category of bias is highest today?</b>',
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
                    # f"<b>Bias Rating: </b>{bias}<br>"
                    f"<b>Count: </b>{count}<br>"
                    f"<b>Proportion: </b>{count/total_articles:.1%}"
                    # f"This accounts for <b>{count/total_articles:.2%}%</b> of the total available articles in the current selection.<br>"
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
                title='<b>Which category of bias is highest today?</b>',
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
                    filtered_df = filtered_df[(filtered_df['date_published'] >= start_date) & (filtered_df['date_published'] <= end_date)]
                if selected_publishers:
                    filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
                if selected_bias_categories:
                    filtered_df = filtered_df[filtered_df[selected_bias_categories].sum(axis=1) > 0]
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

                if (clickData is not None) or (clickData is None and id == 'export-button3'):
                    bias = str(clickData['points'][0]['label'])

                    # Table title
                    main_title = f'Showing all articles that were rated <b>{bias}</b> by the model.'
                    keys = '<b>Legend: G =</b> Generalisation, <b>O =</b> Omit Due Prominence, <b>N =</b> Negative Behaviour, <b>M =</b> Misinterpretation, <b>H =</b> Headline'
                    title_html = f'{main_title}<br>{keys}'
                    
                    title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(title_html)

                    # Apply formatting
                    filtered_df = filtered_df[filtered_df['bias_rating_label'] == bias]
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

                if id == 'export-button3a':
                    return [title], table, {'fontSize': 14, 'display': 'block'}, {'fontSize': 14, 'display': 'block', 'margin-left': '10px'}, csv_string

                return [title], table, {'fontSize': 14, 'display': 'block'}, {'fontSize': 14, 'display': 'block', 'margin-left': '10px'}, csv_string

            elif id in ['chart3a-datepickerrange', 'chart3a-publisher-dropdown', 'chart3a-bias-category-dropdown', 'chart3a-topic-dropdown', 'clear-button3a']:
                return [], None, {'display': 'none'}, {'display': 'none'}, ''

        else:
            return [], None, {'display': 'none'}, {'display': 'none'}, ''
        
    
    # Callback for Chart 3B
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

            if id in ['top-offending-articles-bar-chart-3b', 'export-button3b']:
                filtered_df = df_corpus.copy()

                # Apply filters for quarters, publishers, and topics
                if (selected_start_date is not None) & (selected_end_date is not None):
                    start_date = pd.to_datetime(str(selected_start_date))
                    end_date = pd.to_datetime(str(selected_end_date))
                    filtered_df = filtered_df[(filtered_df['date_published'] >= start_date) & (filtered_df['date_published'] <= end_date)]
                if selected_publishers:
                    filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
                if selected_bias_categories:
                    filtered_df = filtered_df[filtered_df[selected_bias_categories].sum(axis=1) > 0]
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

                if (clickData is not None) or (clickData is None and id == 'export-button3'):
                    bias = str(clickData['points'][0]['label'])

                    # Table title
                    main_title = f'Showing all articles that were rated <b>{bias}</b> by the model.'
                    keys = '<b>Legend: G =</b> Generalisation, <b>O =</b> Omit Due Prominence, <b>N =</b> Negative Behaviour, <b>M =</b> Misinterpretation, <b>H =</b> Headline'
                    title_html = f'{main_title}<br>{keys}'
                    
                    title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(title_html)

                    # Apply formatting
                    filtered_df = filtered_df[filtered_df['bias_rating_label'] == bias]
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

                if id == 'export-button3b':
                    return [title], table, {'fontSize': 14, 'display': 'block'}, {'fontSize': 14, 'display': 'block', 'margin-left': '10px'}, csv_string

                return [title], table, {'fontSize': 14, 'display': 'block'}, {'fontSize': 14, 'display': 'block', 'margin-left': '10px'}, csv_string

            elif id in ['chart3b-datepickerrange', 'chart3b-publisher-dropdown', 'chart3b-bias-category-dropdown', 'chart3b-topic-dropdown', 'clear-button3b']:
                return [], None, {'display': 'none'}, {'display': 'none'}, ''

        else:
            return [], None, {'display': 'none'}, {'display': 'none'}, ''
