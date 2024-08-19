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

# Define the comparison layout for Chart 2A and Chart 2B
def create_layout():
    layout = html.Div(style={'justify-content': 'center', 'backgroundColor': '#ffffff'}, className='row', children=[
        html.H2(children="Top Topics", style={'textAlign': 'center'}),

        # Chart 2A vs Chart 2B
        html.Div([

            # All elements for Chart 2A
            html.H2("A", style={'textAlign': 'center'}),

            html.Div([
                html.Label(['Date Published:'], style={'font-weight': 'bold', 'width': '20%'}),
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
                html.Label(['Publishers:'], style={'font-weight': 'bold', 'width': '20%'}),
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
                html.Label(['Overall Bias Score'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart2a-bias-rating-dropdown',
                options=[
                    {'label': 'Inconclusive', 'value':-1},
                    {'label': 'Biased', 'value': 1},
                    {'label': 'Very Biased', 'value': 2},
                    {'label': 'Not Biased', 'value': 0},
                ],
                value = [1, 2],
                placeholder='Select Overall Bias Score',
                multi=True,
                clearable=True,
                style = {'width': '70%'}
                )
            ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

            html.Div([
                html.Label(['Category of Bias:'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart2a-bias-category-dropdown',
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
            ], style={'display':'flex', 'margin-bottom':'50px', 'align-items': 'center'}),

            # Graph for displaying the top topics
            dcc.Graph(id='top-topics-bar-chart-2a', style = {'margin-bottom': '50px'}),

            # Table for displaying the top topics
            html.Div(id='table2a-title', style={'fontSize': 20, 'fontColor': '#2E2C2B', 'margin-bottom': '20px'}),
            html.Div(id='table2a'),
            html.Div([
                dbc.Button('Clear Table', id='clear-button2a', style = {'display': 'none'}),
                dbc.Button('Export to CSV', id='export-button2a', style = {'display': 'none'}
                        )
            ], style={'display':'flex', 'margin-top': '10px', 'align-items': 'center'}),

        ],
        style={'width': '48%', 'display': 'inline-block', 'backgroundColor': 'white', 'box-shadow': '0 8px 15px rgba(0, 0, 0, 0.2)', 'border-radius': '0px', 'padding': '10px', 'margin': '5px'}),


        # All elements for Chart 2B
        html.Div([
            html.H2("B", style={'textAlign': 'center'}),

            html.Div([
                html.Label(['Date Published:'], style={'font-weight': 'bold', 'width': '20%'}),
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
                html.Label(['Publishers:'], style={'font-weight': 'bold', 'width': '20%'}),
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
                html.Label(['Overall Bias Score'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart2b-bias-rating-dropdown',
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
                html.Label(['Category of Bias'], style={'font-weight': 'bold', 'width': '20%'}),
                dcc.Dropdown(
                id='chart2b-bias-category-dropdown',
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
        style={'width': '48%', 'display': 'inline-block', 'backgroundColor': 'white', 'box-shadow': '0 8px 15px rgba(0, 0, 0, 0.2)', 'border-radius': '0px', 'padding': '10px', 'margin': '5px'}),

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
            total_articles = filtered_df_exploded['article_url'].nunique()

            # Predefine colors for the top 5 topics
            top_colors = ['#4185A0', '#AA4D71', '#B85C3B', '#C5BE71', '#7658A0']
            gray_color = '#CAC6C2' # Add gray color for the remaining topics

            # Create bars for the bar chart
            data = []
            for i, (topic, count) in enumerate(topic_counts.items()):
                tooltip_text = (
                    # f"<b>Topic: </b>{topic}<br>"
                    f"<b>Count: </b>{count}<br>"
                    f"<b>Proportion: </b>{count/total_articles:.1%}<br>"
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
                title="<b>WWhat are today's biased/very biased article topics?</b>",
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
            total_articles = filtered_df_exploded['article_url'].nunique()

            # Predefine colors for the top 5 topics
            top_colors = ['#4185A0', '#AA4D71', '#B85C3B', '#C5BE71', '#7658A0']
            gray_color = '#CAC6C2' # Add gray color for the remaining topics

            # Create bars for the bar chart
            data = []
            for i, (topic, count) in enumerate(topic_counts.items()):
                tooltip_text = (
                    # f"<b>Topic: </b>{topic}<br>"
                    f"<b>Count: </b>{count}<br>"
                    f"<b>Proportion: </b>{count/total_articles:.1%}<br>"
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
                title="<b>What are today's biased/very biased article topics?</b>",
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
                    filtered_df = filtered_df[(filtered_df['date_published'] >= start_date) & (filtered_df['date_published'] <= end_date)]
                if selected_publishers:
                    filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
                if selected_bias_ratings:
                    filtered_df = filtered_df[filtered_df['bias_rating'].isin(selected_bias_ratings)]
                if selected_bias_categories:
                    filtered_df = filtered_df[filtered_df[selected_bias_categories].sum(axis=1) > 0]

                if (clickData is not None) or (clickData is None and id == 'export-button2'):
                    topic = str(clickData['points'][0]['label'])

                    # Table title
                    main_title = f'Showing all articles about <b>{topic}</b>'
                    keys = '<b>Legend: G =</b> Generalisation, <b>O =</b> Omit Due Prominence, <b>N =</b> Negative Behaviour, <b>M =</b> Misrepresentation, <b>H =</b> Headline'
                    title_html = f'{main_title}<br>{keys}'
                        
                    title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(title_html)


                    # Apply formatting
                    filtered_df = filtered_df[filtered_df['topic'].str.contains('|'.join([topic]))]
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

                if id == 'export-button2a':
                    return [title], table, {'fontSize': 14, 'display': 'block'}, {'fontSize': 14, 'display': 'block', 'margin-left': '10px'}, csv_string

                return [title], table, {'fontSize': 14, 'display': 'block'}, {'fontSize': 14, 'display': 'block', 'margin-left': '10px'}, csv_string

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
                    filtered_df = filtered_df[(filtered_df['date_published'] >= start_date) & (filtered_df['date_published'] <= end_date)]
                if selected_publishers:
                    filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
                if selected_bias_ratings:
                    filtered_df = filtered_df[filtered_df['bias_rating'].isin(selected_bias_ratings)]
                if selected_bias_categories:
                    filtered_df = filtered_df[filtered_df[selected_bias_categories].sum(axis=1) > 0]

                if (clickData is not None) or (clickData is None and id == 'export-button2'):
                    topic = str(clickData['points'][0]['label'])

                    # Table title
                    main_title = f'Showing all articles about <b>{topic}</b>'
                    keys = '<b>Legend: G =</b> Generalisation, <b>O =</b> Omit Due Prominence, <b>N =</b> Negative Behaviour, <b>M =</b> Misrepresentation, <b>H =</b> Headline'
                    title_html = f'{main_title}<br>{keys}'
                        
                    title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(title_html)


                    # Apply formatting
                    filtered_df = filtered_df[filtered_df['topic'].str.contains('|'.join([topic]))]
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

                if id == 'export-button2b':
                    return [title], table, {'fontSize': 14, 'display': 'block'}, {'fontSize': 14, 'display': 'block', 'margin-left': '10px'}, csv_string

                return [title], table, {'fontSize': 14, 'display': 'block'}, {'fontSize': 14, 'display': 'block', 'margin-left': '10px'}, csv_string

            elif id in ['chart2b-datepickerrange', 'chart2b-publisher-dropdown', 'chart2b-bias-rating-dropdown', 'chart2b-bias-category-dropdown', 'chart2b-color-toggle', 'clear-button2b']:
                return [], None, {'display': 'none'}, {'display': 'none'}, ''

        else:
            return [], None, {'display': 'none'}, {'display': 'none'}, ''
