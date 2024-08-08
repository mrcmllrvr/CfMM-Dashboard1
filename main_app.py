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

import compare_chart1
import compare_chart2
import compare_chart3
import compare_chart4
import open_cards

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

# Calculate relevant parameters for the main page (boxes)
total_articles = len(df_corpus)
total_publishers = len(unique_publishers)
total_locations = len(df_corpus['location'].dropna().unique())

# Get today's date
date_today = datetime.today().strftime('%B %d, %Y')

# Initialize the Dash application
stylesheets = [dbc.themes.FLATLY] # 'https://codepen.io/chriddyp/pen/bWLwgP.css'
app = dash.Dash(__name__, external_stylesheets=stylesheets, suppress_callback_exceptions=True)
server = app.server

# Define the main layout of the application
app.layout = html.Div(children=[
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


# Callback for Homepage Chart 1
@app.callback(
    Output('homepage-top-offending-publishers-bar-chart', 'figure'),
    [
        Input('homepage-chart1-color-toggle', 'value')
    ]
)

def update_homepage_chart1(color_by):
    filtered_df = df_corpus.copy()

    # Filter latest scraped date for homepage
    start_date = pd.to_datetime(str(df_corpus['date_published'].max()))
    end_date = pd.to_datetime(str(df_corpus['date_published'].max()))
    filtered_df = filtered_df[(filtered_df['date_published']>=start_date) & (filtered_df['date_published']<=end_date)]

    # Calculate the total counts of very biased and biased articles for each publisher
    publisher_totals = filtered_df[filtered_df['bias_rating']>=1].groupby('publisher', observed=True).size()

    # Sort publishers by this count and get the top 10
    top_publishers = publisher_totals.sort_values(ascending=False).head(10).index[::-1]

    # Filter the dataframe to include only the top publishers
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
            # Color mapping for bias ratings
            color_map = {
                2: ('#C22625', 'Very Biased'),
                1: ('#eb8483', 'Biased'),
                0: ('#f2eadf', 'Not Biased'), # #FFE5DC
                -1: ('#CAC6C2', 'Inconclusive')
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
                        f"<b>Overall Bias Score: </b> {name}<br>"
                        f"<b>Count: </b>{articles}<br>"
                        f"<b>Proportion: </b>{percentage_of_total:.2f}%<br>"
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
                title=f"""<b>Who are today's top offending publishers?</b>""",
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
                            f"<b>Category of Bias: </b>{category.replace('_', ' ').title().replace('Or', 'or')}<br>"
                            f"<b>Count: </b>{articles}<br>"
                            f"<b>Proportion: </b>{percentage_of_total:.2f}%<br>"
                            # f"Of the {total_biased_articles} articles, <b>{articles}</b> of them committed <b>{category.replace('_', ' ').title().replace('Or', 'or')}</b>.<br>"
                            # f"This accounts for <b>{percentage_of_total:.2f}%</b> of the total available articles for <b>{category.replace('_', ' ').title().replace('Or', 'or')}</b>.<br>"
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
                title=f"""<b>Who are today's top offending publishers?</b>""",
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
    
# Function for Homepage Chart 2
def update_homepage_chart2():
    filtered_df = df_corpus.copy()

    # Focus on VB/B articles only
    filtered_df = filtered_df[filtered_df['bias_rating']>=1]

    # Filter latest scraped date for homepage
    start_date = pd.to_datetime(str(df_corpus['date_published'].max()))
    end_date = pd.to_datetime(str(df_corpus['date_published'].max()))
    filtered_df = filtered_df[(filtered_df['date_published']>=start_date) & (filtered_df['date_published']<=end_date)]

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
                f"<b>Count: </b>{count}<br>"
                f"<b>Proportion: </b>{count/total_articles:.2%}<br>"
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
            title="<b>What are the topics of today's biased/very biased articles?</b>",
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


# Function for Homepage Chart 3
def update_homepage_chart3():
    filtered_df = df_corpus.copy()

    # Filter latest scraped date for homepage
    start_date = pd.to_datetime(str(df_corpus['date_published'].max()))
    end_date = pd.to_datetime(str(df_corpus['date_published'].max()))
    filtered_df = filtered_df[(filtered_df['date_published']>=start_date) & (filtered_df['date_published']<=end_date)]

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
                f"<b>Overall Bias Score: </b>{bias}<br>"
                f"<b>Count: </b>{count}<br>"
                f"<b>Proportion: </b>{count/total_articles:.2%}<br>"
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


# Callback for Homepage Chart 4
@app.callback(
    Output('homepage-wordcloud-container', 'figure'),
    [
        Input('homepage-chart4-text-toggle', 'value'),
        Input('homepage-chart4-ngram-dropdown', 'value')
    ]
)
def update_homepage_chart4(text_by, ngram_value):
    filtered_df = df_corpus.copy()

    # Focus on B/VB articles only
    filtered_df = filtered_df[filtered_df['bias_rating']>=1]

    # Filter latest scraped date for homepage
    start_date = pd.to_datetime(str(df_corpus['date_published'].max()))
    end_date = pd.to_datetime(str(df_corpus['date_published'].max()))
    filtered_df = filtered_df[(filtered_df['date_published']>=start_date) & (filtered_df['date_published']<=end_date)]

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
                      scale=1.0,
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
                            f"<b>Count: </b>{raw_count}<br>"
                            f"<b>Proportion: </b>{percentage:.2f}%<br>"
                            #   f"The word <b>'{word}'</b> appeared <b>{raw_count}</b> times across all articles in the current selection.<br>"
                            #   f"This accounts for <b>{percentage:.2f}%</b> of the total available word/phrases.<br>"
                            )

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

        # Update the layout to remove axes and make the word cloud bigger
        fig.update_layout(
            title="<b>What are the trending words/phrases in today's biased/very biased articles?</b>",
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            template='simple_white',
            height=800,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='#2E2C2B',
            font_size=14,
            margin={'l': 150, 'r': 20, 'b': 40, 't': 40}
        )

        # Reverse the y-axis to match the word cloud orientation
        fig.update_yaxes(autorange="reversed")

    return fig

# Define the layout for the main page
main_layout = html.Div(children=[
    html.H4(date_today, style={'margin': '20px'}),
    html.H1(children="Today's Insight/Metrics", style={'margin': '20px', 'height': '100px', 'font-weight': 'bolder'}),

    # Modals
    # Modal for Chart 1
    dbc.Modal(
            [
                dbc.ModalHeader(),
                dbc.ModalBody(children=[
                    html.Div([
                        html.Label(['Date Published:'], style={'font-weight': 'bold', 'width': '20%'}),
                        dcc.DatePickerRange(
                            id='chart1-datepickerrange',
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
                        id='chart1-publisher-dropdown',
                        options=[{'label': publisher, 'value': publisher} for publisher in unique_publishers],
                        placeholder='Select Publisher',
                        multi=True,
                        clearable=True,
                        style = {'width': '70%'})
                    ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

                    html.Div([
                        html.Label(['Overall Bias Score:'], style={'font-weight': 'bold', 'width': '20%'}),
                        dcc.Dropdown(
                        id='chart1-bias-rating-dropdown',
                        options=[
                            {'label': 'Inconclusive', 'value':-1},
                            {'label': 'Biased', 'value': 1},
                            {'label': 'Very Biased', 'value': 2},
                            {'label': 'Not Biased', 'value': 0},
                        ],
                        placeholder='Select Overall Bias Score',
                        multi=True,
                        clearable=True,
                        style = {'width': '70%'})
                    ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

                    html.Div([
                        html.Label(['Category of Bias:'], style={'font-weight': 'bold', 'width': '20%'}),
                        dcc.Dropdown(
                        id='chart1-bias-category-dropdown',
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
                        id='chart1-topic-dropdown',
                        options=[{'label': topic, 'value': topic} for topic in unique_topics],
                        placeholder='Select Topic',
                        multi=True,
                        clearable=True,
                        style = {'width': '70%'})
                    ], style={'display':'flex', 'margin-bottom':'30px', 'align-items': 'center'}),

                    # Toggle for color by bias ratings or bias categories
                    dcc.RadioItems(
                        id='chart1-color-toggle',
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
                    dcc.Graph(id='top-offending-publishers-bar-chart', style = {'margin-bottom': '50px'}),

                    # Table for displaying the top offending publishers
                    html.Div(id='table1-title', style={'fontSize': 20, 'fontColor': '#2E2C2B', 'margin-bottom': '20px'}),
                    html.Div(id='table1'),
                    html.Div([
                        dbc.Button('Clear Table', id='clear-button1', style = {'display': 'none'}),
                        dbc.Button('Export to CSV', id='export-button1', style = {'display': 'none'})
                    ], style={'display':'flex', 'margin-top': '10px', 'align-items': 'center'}),
                ])
            ],
            id="modal_1",
            centered=True,
            scrollable=True,
            backdrop="static",
            fullscreen=True,
            is_open=False,
    ),

    # Modal for Chart 2
    dbc.Modal(
            [
                dbc.ModalHeader(),
                dbc.ModalBody(children=[
                    html.Div([
                        html.Label(['Date Published:'], style={'font-weight': 'bold', 'width': '20%'}),
                        dcc.DatePickerRange(
                            id='chart2-datepickerrange',
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
                        id='chart2-publisher-dropdown',
                        options=[{'label': publisher, 'value': publisher} for publisher in unique_publishers],
                        placeholder='Select Publisher',
                        multi=True,
                        clearable=True,
                        style = {'width': '70%'})
                    ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

                    html.Div([
                        html.Label(['Overall Bias Score:'], style={'font-weight': 'bold', 'width': '20%'}),
                        dcc.Dropdown(
                        id='chart2-bias-rating-dropdown',
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
                        style = {'width': '70%'})
                    ], style={'display':'flex', 'margin-bottom':'10px', 'align-items': 'center'}),

                    html.Div([
                        html.Label(['Category of Bias:'], style={'font-weight': 'bold', 'width': '20%'}),
                        dcc.Dropdown(
                        id='chart2-bias-category-dropdown',
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
                    dcc.Graph(id='top-topics-bar-chart', style = {'margin-bottom': '50px'}),

                    # Table for displaying the top topics
                    html.Div(id='table2-title', style={'fontSize': 20, 'fontColor': '#2E2C2B', 'margin-bottom': '20px'}),
                    html.Div(id='table2'),
                    html.Div([
                        dbc.Button('Clear Table', id='clear-button2', style = {'display': 'none'}),
                        dbc.Button('Export to CSV', id='export-button2', style = {'display': 'none'})
                    ], style={'display':'flex', 'margin-top': '10px', 'align-items': 'center'}),
                ])
            ],
            id="modal_2",
            centered=True,
            scrollable=True,
            backdrop="static",
            fullscreen=True,
            is_open=False,
    ),

    # Modal for Chart 3
    dbc.Modal(
            [
                dbc.ModalHeader(),
                dbc.ModalBody(children=[
                    html.Div([
                        html.Label(['Date Published:'], style={'font-weight': 'bold', 'width': '20%'}),
                        dcc.DatePickerRange(
                            id='chart3-datepickerrange',
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
                        id='chart3-publisher-dropdown',
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
                        id='chart3-bias-category-dropdown',
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
                        id='chart3-topic-dropdown',
                        options=[{'label': topic, 'value': topic} for topic in unique_topics],
                        placeholder='Select Topic',
                        multi=True,
                        clearable=True,
                        style = {'width': '70%'}
                        )
                    ], style={'display':'flex', 'margin-bottom':'50px', 'align-items': 'center'}),

                    # Graph for displaying the top topics
                    dcc.Graph(id='top-offending-articles-bar-chart', style = {'margin-bottom': '50px'}),

                    # Table for displaying the top topics
                    html.Div(id='table3-title', style={'fontSize': 20, 'fontColor': '#2E2C2B', 'margin-bottom': '20px'}),
                    html.Div(id='table3'),
                    html.Div([
                        dbc.Button('Clear Table', id='clear-button3', style = {'display': 'none'}),
                        dbc.Button('Export to CSV', id='export-button3', style = {'display': 'none'}
                                )
                    ], style={'display':'flex', 'margin-top': '10px', 'align-items': 'center'}),
                ])
            ],
            id="modal_3",
            centered=True,
            scrollable=True,
            backdrop="static",
            fullscreen=True,
            is_open=False,
    ),

    # Modal for Chart 4
    dbc.Modal(
            [
                dbc.ModalHeader(),
                dbc.ModalBody(children=[
                    html.Div([
                        html.Label(['Date Published:'], style={'font-weight': 'bold', 'width': '20%'}),
                        dcc.DatePickerRange(
                            id='chart4-datepickerrange',
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
                            id='chart4-publisher-dropdown',
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
                            id='chart4-bias-rating-dropdown',
                            options=[
                                {'label': 'Biased', 'value': 2},
                                {'label': 'Very Biased', 'value': 1},
                                {'label': 'Not Biased', 'value': 0},
                                {'label': 'Inconclusive', 'value': -1},
                            ],
                            value=[1, 2],
                            placeholder='Select Overall Bias Score',
                            multi=True,
                            clearable=True,
                            style={'width': '70%'}
                        )
                    ], style={'display': 'flex', 'margin-bottom': '10px', 'align-items': 'center'}),

                    html.Div([
                        html.Label(['Category of Bias:'], style={'font-weight': 'bold', 'width': '20%'}),
                        dcc.Dropdown(
                            id='chart4-bias-category-dropdown',
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
                            id='chart4-topic-dropdown',
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
                            id='chart4-ngram-dropdown',
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
                        id='chart4-text-toggle',
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
                        dcc.Input(id='word-search', type='text', style={'width': '49%', 'display': 'block'}),
                        dbc.Button('Search', id='search-button4', style={'margin-left': '30px', 'width': '10%', 'display': 'block'})
                    ], style={'display': 'flex', 'margin-top': '30px', 'margin-bottom': '30px', 'align-items': 'center'}),

                    # Graph for displaying the word cloud
                    dcc.Graph(id='wordcloud-container'),

                    # Table for displaying the result for word search
                    html.Div(id='table4-title', style={'fontSize': 20, 'color': '#2E2C2B', 'margin-bottom': '20px'}),
                    html.Div(id='table4'),
                    html.Div([
                        dbc.Button('Clear Table', id='clear-button4', style={'display': 'none'}),
                        dbc.Button('Export to CSV', id='export-button4', style={'display': 'none'})
                    ], style={'display': 'flex', 'margin-top': '10px', 'align-items': 'center'}),
                ])
            ],
            id="modal_4",
            centered=True,
            scrollable=True,
            backdrop="static",
            fullscreen=True,
            is_open=False,
    ),

    # Cards
    html.Div(
        id='left-column',
        style={'width': '13%', 'float': 'left'},
        children=[
            html.Div(
                html.A(
                    dbc.Button(
                        html.Div([
                            html.Div(f'{total_articles}', style={'font-size': '55px', 'font-weight': 'bolder'}),
                            html.Div('Articles', style={'font-size': '18px', 'margin-top': '0px'})
                        ], style={'text-align': 'center'}),
                        id='total-articles-card',
                        style={
                            'width': '98%',
                            'height': '100%',
                            'background-color': '#E7E5E3',
                            'color': '#2E2C2B',
                            'border': 'none',
                            'display': 'flex',
                            'flex-direction': 'column',
                            'justify-content': 'center',
                            'align-items': 'center',
                            'padding': '20px',
                            'margin-top': '0px',
                            'margin-left': '10px'
                        }
                    ),
                    href='/total-articles-card',
                    target="_blank",
                    style={'text-decoration': 'none', 'width': '100%', 'height': '100%'}
                )
            ),

            html.Div(
                html.A(
                    dbc.Button(
                        html.Div([
                            html.Div(f'{total_publishers}', style={'font-size': '55px', 'font-weight': 'bolder'}),
                            html.Div('Publishers', style={'font-size': '18px', 'margin-top': '0px'})
                        ], style={'text-align': 'center'}),
                        id='total-publishers-card',
                        style={
                            'width': '98%',
                            'height': '100%',
                            'background-color': '#E7E5E3',
                            'color': '#2E2C2B',
                            'border': 'none',
                            'display': 'flex',
                            'flex-direction': 'column',
                            'justify-content': 'center',
                            'align-items': 'center',
                            'padding': '20px',
                            'margin-top': '10px',
                            'margin-left': '10px'
                        }
                    ),
                    href='/total-publishers-card',
                    target="_blank",
                    style={'text-decoration': 'none', 'width': '100%', 'height': '100%'}
                )
            ),
            html.Div(
                html.A(
                    dbc.Button(
                        html.Div([
                            html.Div(f'{total_locations}', style={'font-size': '55px', 'font-weight': 'bolder'}),
                            html.Div('Locations', style={'font-size': '18px', 'margin-top': '0px'})
                        ], style={'text-align': 'center'}),
                        id='total-locations-card',
                        style={
                            'width': '98%',
                            'height': '100%',
                            'background-color': '#E7E5E3',
                            'color': '#2E2C2B',
                            'border': 'none',
                            'display': 'flex',
                            'flex-direction': 'column',
                            'justify-content': 'center',
                            'align-items': 'center',
                            'padding': '20px',
                            'margin-top': '10px',
                            'margin-left': '10px'
                        }
                    ),
                    href='/total-locations-card',
                    target="_blank",
                    style={'text-decoration': 'none', 'width': '100%', 'height': '100%'}
                )
            )
        ]
    ),


    # Charts 1, 2, 3, and 4
    html.Div(style={'width': '87%', 'float': 'right', "display": "grid", "grid-template-columns": "110%"}, className='row', children=[

        # Charts 1 and 2 side by side
        html.Div(className='row',children=[

            # All elements for Chart 1
            html.Div([
                html.A(dbc.Button('Explore', id='explore-button1', style={'margin-left': '85%', 'width': '15%', 'display': 'grid', 'background-color': '#D90429'}),
                    target="_blank",
                    style={'text-decoration': 'none'},
                    n_clicks=0),

                html.A(dbc.Button('Compare', id='compare-button1', style={'margin-left': '85%', 'width': '15%', 'display': 'block', 'background-color': '#D90429'}),
                    href='/compare-chart-1', target="_blank",
                    style={'text-decoration': 'none'}),

                ## TODO: Place Homepage Chart 1 elements here
                # Toggle for color by bias ratings or bias categories
                dcc.RadioItems(
                    id='homepage-chart1-color-toggle',
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
                dcc.Graph(id='homepage-top-offending-publishers-bar-chart', style = {'margin-bottom': '50px'}),
                
            ],style={'width': '45%', 'display': 'inline-block','border': '1px solid black', 'border-radius': '5px', 'padding': '10px'},
            ),


            # All elements for Chart 2
            html.Div([
                html.A(dbc.Button('Explore', id='explore-button2', style={'margin-left': '85%', 'width': '15%', 'display': 'block', 'background-color': '#D90429'}),
                    target="_blank",
                    style={'text-decoration': 'none'},
                    n_clicks=0),

                html.A(dbc.Button('Compare', id='compare-button2', style={'margin-left': '85%', 'width': '15%', 'display': 'block', 'background-color': '#D90429'}),
                    href='/compare-chart-2', target="_blank",
                    style={'text-decoration': 'none'}),


                ## TODO: Place Homepage Chart 2 elements Here
                # Graph for displaying the top topics
                dcc.Graph(id='homepage-top-topics-bar-chart', figure=update_homepage_chart2(), style={'margin-bottom': '50px'}),

            ],style={'width': '45%', 'display': 'inline-block','border': '1px solid black', 'border-radius': '5px', 'padding': '10px'},
            ),

        ]),

        # Charts 3 and 4 side by side
        html.Div(className='row', children=[

            # All elements for Chart 3
            html.Div([
                html.A(dbc.Button('Explore', id='explore-button3', style={'margin-left': '85%', 'width': '15%', 'display': 'block', 'background-color': '#D90429'}),
                    target="_blank",
                    style={'text-decoration': 'none'},
                    n_clicks=0),

                html.A(dbc.Button('Compare', id='compare-button3', style={'margin-left': '85%', 'width': '15%', 'display': 'block', 'background-color': '#D90429'}),
                    href='/compare-chart-3', target="_blank",
                    style={'text-decoration': 'none'}),

                ## TODO: Place Homepage Chart 3 elements here
                # Graph for displaying the top topics
                dcc.Graph(id='homepage-top-offending-articles-bar-chart', figure=update_homepage_chart3(), style = {'margin-bottom': '50px'}),

            ],style={'width': '45%', 'display': 'inline-block','border': '1px solid black', 'border-radius': '5px', 'padding': '10px'},
            ),

            # All elements for Chart 4
            html.Div([
                html.A(dbc.Button('Explore', id='explore-button4', style={'margin-left': '85%', 'width': '15%', 'display': 'block', 'background-color': '#D90429'}),
                    target="_blank",
                    style={'text-decoration': 'none'},
                    n_clicks=0),

                html.A(dbc.Button('Compare', id='compare-button4', style={'margin-left': '85%', 'width': '15%', 'display': 'block', 'background-color': '#D90429'}),
                    href='/compare-chart-4', target="_blank",
                    style={'text-decoration': 'none'}),

                ## TODO: Place Homepage Chart 4 elements here
                # Dropdown for n-gram selection
                html.Div([
                    html.Label(['Select Word Grouping:'], style={'font-weight': 'bold', 'width': '20%'}),
                    dcc.Dropdown(
                        id='homepage-chart4-ngram-dropdown',
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
                    id='homepage-chart4-text-toggle',
                    options=[
                        {'label': '    Headline-only', 'value': 'title'},
                        {'label': '    Full-text', 'value': 'text'}
                    ],
                    value='title',  # default value on load
                    labelStyle={'display': 'inline-block'},
                    inputStyle={"margin-left": "10px"},
                    style={'margin-bottom': '50px'}
                ),

                # Graph for displaying the word cloud
                dcc.Graph(id='homepage-wordcloud-container'),



            ], 
                style={'width': '45%', 'display': 'inline-block','border': '1px solid black', 'border-radius': '5px', 'padding': '10px'},
            )

        ]),
    ])
])

# Callback for Modal 1
@app.callback(
    Output("modal_1", "is_open"),
    [Input("explore-button1", "n_clicks")],
    [State("modal_1", "is_open")],
)
def toggle_modal_1(n1, is_open):
    if n1:
        return not is_open
    return is_open

# Callback for Modal 2
@app.callback(
    Output("modal_2", "is_open"),
    [Input("explore-button2", "n_clicks")],
    [State("modal_2", "is_open")],
)
def toggle_modal_2(n1, is_open):
    if n1:
        return not is_open
    return is_open

# Callback for Modal 3
@app.callback(
    Output("modal_3", "is_open"),
    [Input("explore-button3", "n_clicks")],
    [State("modal_3", "is_open")],
)
def toggle_modal_3(n1, is_open):
    if n1:
        return not is_open
    return is_open

# Callback for Modal 4
@app.callback(
    Output("modal_4", "is_open"),
    [Input("explore-button4", "n_clicks")],
    [State("modal_4", "is_open")],
)
def toggle_modal_4(n1, is_open):
    if n1:
        return not is_open
    return is_open



# Callback for Chart 1
@app.callback(
    Output('top-offending-publishers-bar-chart', 'figure'),
    [
        Input('chart1-datepickerrange', 'start_date'),
        Input('chart1-datepickerrange', 'end_date'),
        Input('chart1-publisher-dropdown', 'value'),
        Input('chart1-bias-rating-dropdown', 'value'),
        Input('chart1-bias-category-dropdown', 'value'),
        Input('chart1-topic-dropdown', 'value'),
        Input('chart1-color-toggle', 'value')
    ],
    allow_duplicate=True,
)

def update_chart1(selected_start_date, selected_end_date, selected_publishers, selected_bias_ratings, selected_bias_categories, selected_topics, color_by):
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

    # Calculate the total counts of very biased and biased articles for each publisher
    publisher_totals = filtered_df[filtered_df['bias_rating']>=1].groupby('publisher', observed=True).size()

    # Sort publishers by this count and get the top 10
    top_publishers = publisher_totals.sort_values(ascending=False).head(10).index[::-1]

    # Filter the dataframe to include only the top publishers
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
            # Color mapping for bias ratings
            color_map = {
                2: ('#C22625', 'Very Biased'),
                1: ('#eb8483', 'Biased'),
                0: ('#f2eadf', 'Not Biased'), # #FFE5DC
                -1: ('#CAC6C2', 'Inconclusive')
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
                        f"<b>Overall Bias Score:</b> {name}<br>"
                        f"<b>Count:</b> {articles}<br>"
                        f"<b>Proportion:</b> {percentage_of_total:.2f}%<br>"
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
    
            # Update the layout
            layout = go.Layout(
                title=f"""<b>Who are today's top offending publishers?</b>""",
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
                            f"<b>Category of Bias: </b>{category.replace('_', ' ').title().replace('Or', 'or')}<br>"
                            f"<b>Count:</b> {articles}<br>"
                            f"<b>Proportion:</b> {percentage_of_total:.2f}%<br>"
                            # f"Of the {total_biased_articles} articles, <b>{articles}</b> of them committed <b>{category.replace('_', ' ').title().replace('Or', 'or')}</b>.<br>"
                            # f"This accounts for <b>{percentage_of_total:.2f}%</b> of the total available articles for <b>{category.replace('_', ' ').title().replace('Or', 'or')}</b>.<br>"
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
                title=f"""<b>Who are today's top offending publishers?</b>""",
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


# Callback for Chart 2
@app.callback(
    Output('top-topics-bar-chart', 'figure'),
    [
        Input('chart2-datepickerrange', 'start_date'),
        Input('chart2-datepickerrange', 'end_date'),
        Input('chart2-publisher-dropdown', 'value'),
        Input('chart2-bias-rating-dropdown', 'value'),
        Input('chart2-bias-category-dropdown', 'value')
    ]
)

def update_chart2(selected_start_date, selected_end_date, selected_publishers, selected_bias_ratings, selected_bias_categories):
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
                f"<b>Count: </b>{count}<br>"
                f"<b>Proportion: </b>{count/total_articles:.2%}"
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


# Callback for Chart 3
@app.callback(
    Output('top-offending-articles-bar-chart', 'figure'),
    [
        Input('chart3-datepickerrange', 'start_date'),
        Input('chart3-datepickerrange', 'end_date'),
        Input('chart3-publisher-dropdown', 'value'),
        Input('chart3-bias-category-dropdown', 'value'),
        Input('chart3-topic-dropdown', 'value')
    ]
)

def update_chart3(selected_start_date, selected_end_date, selected_publishers, selected_bias_categories, selected_topics):
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
                f"<b>Overall Bias Score: </b>{bias}<br>"
                f"<b>Count: </b>{count}<br>"
                f"<b>Proportion: </b>{count/total_articles:.2%}<br>"
                # f"This accounts for <b>{count/total_articles:.2%}%</b> of the total available articles in the current selection.<br>"
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


# Callback for Chart 4
@app.callback(
    Output('wordcloud-container', 'figure'),
    [
        Input('chart4-datepickerrange', 'start_date'),
        Input('chart4-datepickerrange', 'end_date'),
        Input('chart4-publisher-dropdown', 'value'),
        Input('chart4-topic-dropdown', 'value'),
        Input('chart4-bias-category-dropdown', 'value'),
        Input('chart4-bias-rating-dropdown', 'value'),
        Input('chart4-text-toggle', 'value'),
        Input('chart4-ngram-dropdown', 'value')
    ]
)
def update_chart4(selected_start_date, selected_end_date, selected_publishers, selected_topics, selected_bias_categories, selected_bias_ratings, text_by, ngram_value):
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
                              f"<b>Count: </b>{raw_count}<br>"
                              f"<b>Proportion: </b>{percentage:.2f}%<br>"
                            #   f"The word <b>'{word}'</b> appeared <b>{raw_count}</b> times across all articles in the current selection.<br>"
                            #   f"This accounts for <b>{percentage:.2f}%</b> of the total available word/phrases.<br>"
                              f"<br>"
                              f"Type <b>'{word}'</b> in the Word Search above to find out which articles used this word.")
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

        # Update the layout to remove axes and make the word cloud bigger
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


# Callback for Table 1
@app.callback(
    [
        Output('table1-title', 'children'),
        Output(component_id='table1', component_property='children'),
        Output('clear-button1', 'style'),
        Output('export-button1', 'style'),
        Output('export-button1', 'href')
    ],
    [
        Input('chart1-datepickerrange', 'start_date'),
        Input('chart1-datepickerrange', 'end_date'),
        Input('chart1-publisher-dropdown', 'value'),
        Input('chart1-bias-rating-dropdown', 'value'),
        Input('chart1-bias-category-dropdown', 'value'),
        Input('chart1-topic-dropdown', 'value'),
        Input('chart1-color-toggle', 'value'),
        Input('top-offending-publishers-bar-chart', 'clickData'),
        Input('clear-button1', 'n_clicks')
    ]
)

def update_table1(selected_start_date, selected_end_date, selected_publishers, selected_bias_ratings, selected_bias_categories, selected_topics, color_by, clickData, n_clicks):
    triggered = dash.callback_context.triggered
    topics = ''

    if triggered:
        id = triggered[0]['prop_id'].split('.')[0]

        if id in ['top-offending-publishers-bar-chart', 'export-button1']:
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
                    keys = '<b>Legend: G =</b> Generalization, <b>O =</b> Omit Due Prominence, <b>N =</b> Negative Behaviour, <b>M =</b> Misinterpretation, <b>H =</b> Headline'
                    title_html = f'{main_title}<br>{keys}'
                    title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(title_html)

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
                        default='#2E2C2B'
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

                else:
                    main_title = f'Showing biased/very biased articles from <b>{publisher}</b> published <b>{start_date}</b> to <b>{end_date}</b> {topics}'
                    keys = '<b>Legend: G =</b> Generalization, <b>O =</b> Omit Due Prominence, <b>N =</b> Negative Behaviour, <b>M =</b> Misinterpretation, <b>H =</b> Headline'
                    title_html = f'{main_title}<br>{keys}'
                    title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(title_html)

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

            if id == 'export-button1':
                return [title], table, {'fontSize':14, 'display': 'block'}, {'fontSize':14, 'display': 'block', 'margin-left': '10px'}, csv_string

            return [title], table, {'fontSize':14, 'display': 'block'}, {'fontSize':14, 'display': 'block', 'margin-left': '10px'}, csv_string

        elif id in ['chart1-datepickerrange', 'chart1-topic-dropdown', 'chart1-publisher-dropdown', 'chart1-bias-rating-dropdown', 'chart1-bias-category-dropdown', 'chart1-color-toggle', 'clear-button1']:
            return [], None, {'display': 'none'}, {'display': 'none'}, ''

    else:
        return [], None, {'display': 'none'}, {'display': 'none'}, ''
    


# Callback for Table 2
@app.callback(
    [
        Output('table2-title', 'children'),
        Output(component_id='table2', component_property='children'),
        Output('clear-button2', 'style'),
        Output('export-button2', 'style'),
        Output('export-button2', 'href')
    ],
    [
        Input('chart2-datepickerrange', 'start_date'),
        Input('chart2-datepickerrange', 'end_date'),
        Input('chart2-publisher-dropdown', 'value'),
        Input('chart2-bias-rating-dropdown', 'value'),
        Input('chart2-bias-category-dropdown', 'value'),
        Input('top-topics-bar-chart', 'clickData'),
        Input('clear-button2', 'n_clicks')
    ]
)

def update_table2(selected_start_date, selected_end_date, selected_publishers, selected_bias_ratings, selected_bias_categories, clickData, n_clicks):
    triggered = dash.callback_context.triggered

    if triggered:
        id = triggered[0]['prop_id'].split('.')[0]

        if id in ['top-topics-bar-chart', 'export-button2']:
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
                keys = '<b>Legend: G =</b> Generalization, <b>O =</b> Omit Due Prominence, <b>N =</b> Negative Behaviour, <b>M =</b> Misinterpretation, <b>H =</b> Headline'
                title_html = f'{main_title}<br>{keys}'
                    
                title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(title_html)


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

            if id == 'export-button2':
                return [title], table, {'fontSize': 14, 'display': 'block'}, {'fontSize': 14, 'display': 'block', 'margin-left': '10px'}, csv_string

            return [title], table, {'fontSize': 14, 'display': 'block'}, {'fontSize': 14, 'display': 'block', 'margin-left': '10px'}, csv_string

        elif id in ['chart2-datepickerrange', 'chart2-publisher-dropdown', 'chart2-bias-rating-dropdown', 'chart2-bias-category-dropdown', 'chart2-color-toggle', 'clear-button2']:
            return [], None, {'display': 'none'}, {'display': 'none'}, ''

    else:
        return [], None, {'display': 'none'}, {'display': 'none'}, ''
    

# Callback for Table 3
@app.callback(
    [
        Output('table3-title', 'children'),
        Output(component_id='table3', component_property='children'),
        Output('clear-button3', 'style'),
        Output('export-button3', 'style'),
        Output('export-button3', 'href')
    ],
    [
        Input('chart3-datepickerrange', 'start_date'),
        Input('chart3-datepickerrange', 'end_date'),
        Input('chart3-publisher-dropdown', 'value'),
        Input('chart3-bias-category-dropdown', 'value'),
        Input('chart3-topic-dropdown', 'value'),
        Input('top-offending-articles-bar-chart', 'clickData'),
        Input('clear-button3', 'n_clicks')
    ]
)

def update_table3(selected_start_date, selected_end_date, selected_publishers, selected_bias_categories, selected_topics, clickData, n_clicks):
    triggered = dash.callback_context.triggered
    topics = ''

    if triggered:
        id = triggered[0]['prop_id'].split('.')[0]

        if id in ['top-offending-articles-bar-chart', 'export-button3']:
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
                keys = '<b>Legend: G =</b> Generalization, <b>O =</b> Omit Due Prominence, <b>N =</b> Negative Behaviour, <b>M =</b> Misinterpretation, <b>H =</b> Headline'
                title_html = f'{main_title}<br>{keys}'
                
                title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(title_html)

                # Apply formatting
                filtered_df = filtered_df[filtered_df['bias_rating_label'] == bias]
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

            if id == 'export-button3':
                return [title], table, {'fontSize': 14, 'display': 'block'}, {'fontSize': 14, 'display': 'block', 'margin-left': '10px'}, csv_string

            return [title], table, {'fontSize': 14, 'display': 'block'}, {'fontSize': 14, 'display': 'block', 'margin-left': '10px'}, csv_string

        elif id in ['chart3-datepickerrange', 'chart3-publisher-dropdown', 'chart3-bias-category-dropdown', 'chart3-topic-dropdown', 'clear-button3']:
            return [], None, {'display': 'none'}, {'display': 'none'}, ''

    else:
        return [], None, {'display': 'none'}, {'display': 'none'}, ''



# Callback for Table 4
@app.callback(
    [
        Output('table4-title', 'children'),
        Output('table4', 'children'),
        Output('clear-button4', 'style'),
        Output('export-button4', 'style'),
        Output('export-button4', 'href')
    ],
    [
        Input('search-button4', 'n_clicks'),
        Input('clear-button4', 'n_clicks'),

        Input('chart4-datepickerrange', 'start_date'),
        Input('chart4-datepickerrange', 'end_date'),
        Input('chart4-publisher-dropdown', 'value'),
        Input('chart4-topic-dropdown', 'value'),
        Input('chart4-bias-category-dropdown', 'value'),
        Input('chart4-bias-rating-dropdown', 'value'),
        Input('chart4-ngram-dropdown', 'value'),
        Input('chart4-text-toggle', 'value'),

        Input('word-search', 'value')
    ]
)

def update_table4(n_clicks_search, n_clicks_clear, selected_start_date, selected_end_date, selected_publishers, selected_topics, selected_bias_categories, selected_bias_ratings, selected_ngrams, text_by, search_word):
    triggered = dash.callback_context.triggered
    topics = ''

    if triggered:
        id = triggered[0]['prop_id'].split('.')[0]

        if id == 'search-button4':
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
            keys = '<b>Legend: G =</b> Generalization, <b>O =</b> Omit Due Prominence, <b>N =</b> Negative Behaviour, <b>M =</b> Misinterpretation, <b>H =</b> Headline'
            title_html = f'{main_title}<br>{keys}'
            
            title = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(title_html)

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

            if id == 'export-button4':
                return [title], table, {'fontSize': 14, 'display': 'block'}, {'fontSize': 14, 'display': 'block', 'margin-left': '10px'}, csv_string

            return [title], table, {'fontSize': 14, 'display': 'block'}, {'fontSize': 14, 'display': 'block', 'margin-left': '10px'}, csv_string

        elif id in ['chart4-datepickerrange', 'chart4-publisher-dropdown', 'chart4-bias-rating-dropdown', 'chart4-bias-category-dropdown', 'chart4-topic-dropdown', 'chart4-ngram-dropdown', 'chart4-text-toggle', 'clear-button4']:
            return [], None, {'display': 'none'}, {'display': 'none'}, ''

    return [], None, {'display': 'none'}, {'display': 'none'}, ''


@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/compare-chart-1':
        return compare_chart1.create_layout()
    elif pathname == '/compare-chart-2':
        return compare_chart2.create_layout()
    elif pathname == '/compare-chart-3':
        return compare_chart3.create_layout()
    elif pathname == '/compare-chart-4':
        return compare_chart4.create_layout()
    elif pathname == '/total-articles-card':
        return open_cards.create_layout()
    elif pathname == '/total-publishers-card':
        return open_cards.create_layout()
    elif pathname == '/total-locations-card':
        return open_cards.create_layout()
    else:
        return main_layout

compare_chart1.register_callbacks(app)
compare_chart2.register_callbacks(app)
compare_chart3.register_callbacks(app)
compare_chart4.register_callbacks(app)
open_cards.register_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=True)
