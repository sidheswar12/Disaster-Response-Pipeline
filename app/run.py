import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, WhitespaceTokenizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine

import numpy as np
import re
import nltk

app = Flask(__name__)

def tokenize(text):    
    # get tokens from text
    tokens = WhitespaceTokenizer().tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # clean tokens
    processed_tokens = []
    for token in tokens:
        token = lemmatizer.lemmatize(token).lower().strip('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        token = re.sub(r'\[[^.,;:]]*\]', '', token)
        
        # add token to compiled list if not empty
        if token != '':
            processed_tokens.append(token)
        
    return processed_tokens

def find_text_length(data):
    return np.array([len(text) for text in data]).reshape(-1, 1)

# load data
#engine = create_engine('sqlite:///../data/disaster_message_categories.db')
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('labeled_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# compute length of texts
df['text_length'] = find_text_length(df['message'])

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():    
    # extract genre counts
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # extract categories
    category_map = df.iloc[:,4:].corr().values
    category_names = list(df.iloc[:,4:].columns)

    # extract length of texts
    length_direct = df.loc[df.genre=='direct','text_length']
    length_social = df.loc[df.genre=='social','text_length']
    length_news = df.loc[df.genre=='news','text_length']
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Heatmap(
                    x=category_names,
                    y=category_names[::-1],
                    z=category_map
                )    
            ],

            'layout': {
                'title': 'Heatmap of Categories'
            }
        },

        {
            'data': [
                Histogram(
                    y=length_direct,
                    name='Direct',
                    opacity=0.5
                ),
                Histogram(
                    y=length_social,
                    name='Social',
                    opacity=0.5
                ),
                Histogram(
                    y=length_news,
                    name='News',
                    opacity=0.5
                )
            ],

            'layout': {
                'title': 'Distribution of Text Length',
                'yaxis':{
                    'title':'Count'
                },
                'xaxis': {
                    'title':'Text Length'
                }
            }
        }

    ]
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# Web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
