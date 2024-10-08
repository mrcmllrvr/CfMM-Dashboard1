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

def create_layout():
    layout = html.Div([
    html.Div(id='table-title-article', style={'fontSize': 20, 'color': '#2E2C2B', 'margin-bottom': '20px'}),
    html.Div(id='article-table'),
    html.Div([
        dbc.Button('Clear Table', id='clear-button-article', style={'display': 'none'}),
        dbc.Button('Export to CSV', id='export-button-article', style={'display': 'none'})
    ], style={'display': 'flex', 'margin-top': '10px', 'align-items': 'center'})
])

    return layout

# Register the callbacks
def register_callbacks(app):
    @app.callback(
        [
            Output('table-title-article', 'children'),
            Output(component_id='article-table', component_property='children'),
            Output('clear-button-article', 'style'),
            Output('export-button-article', 'style'),
            Output('export-button-article', 'href')
        ],
        [Input('clear-button-article', 'n_clicks')]
    )
    def update_table_article(n_clicks):
        filtered_df = df_corpus.copy()
        
        # Example title
        title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(f'Showing all articles')

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
        categories = ['generalisation', 'prominence', 'negative_behaviour', 'misrepresentation', 'headline_or_imagery']
        for category in categories:
            filtered_df[category] = np.where(filtered_df[category]==1, 'Y', 'N')
        filtered_df['date_published_label_(yyyy-mm-dd)'] = filtered_df['date_published'].dt.date
        filtered_df['explore_further'] = "<a href='" + '' + "' target='_blank' style='color:#2E2C2B;'>" + "<b>Explore model results ➡️</b>" + "</a>"

        # Save to csv
        csv_df = filtered_df[['publisher', 'title', 'article_url', 'date_published_label_(yyyy-mm-dd)', 'topic', 'bias_rating_label']+categories]
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
            style_data={'textAlign': 'left', 'padding': '5px', 'font-family': 'sans-serif', 'whiteSpace': 'normal', 'height': 'auto'},
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

        return [title], table, {'fontSize':14, 'display': 'block'}, {'fontSize':14, 'display': 'block', 'margin-left': '10px'}, csv_string