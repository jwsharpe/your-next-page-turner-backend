# imports
import json
import codecs
import pandas as pd
from flask import request, url_for
from flask_api import FlaskAPI, status, exceptions
from flask_cors import CORS

from fuzzywuzzy import fuzz

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

app = FlaskAPI(__name__)
CORS(app)

# load in data and transform as needed

# data = pd.read_json('updating_df.json',orient='records') # later
data = pd.read_json('random_200.json')

# data = pd.read_json('goodreads_updated.json', orient='index')
# james_data = data[['id', 'authors', 'titles', 'description', 'img', 'genre']]
# james_data['genre'] = james_data['genre'].map(
#     lambda x: x.split() if type(x) == str else [])

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                     min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(data['bag_of_words'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# input: < string > search query
# output: < [<json>] > list of books. LIMIT = 50
@app.route("/query", methods=['POST'])
def handleSearch():
    query = request.json['query']
    return return_query_pull(query).to_json(orient='records')
    # input: < string > page number
    # output: < [<json>] > list of books. LIMIT = 50


@app.route("/books/<page_num>", methods=['GET'])
def getBooks(page_num):
    pageIndex = int(page_num)
    firstIndex = pageIndex * 50
    lastIndex = (pageIndex + 1) * 50
    return data.iloc[firstIndex:lastIndex].to_json(orient='records')


# output: < [<json>] > list of first 50 books.
@app.route("/books", methods=['GET'])
def notes_list():
    return data.iloc[0:50].to_json(orient='records')


@app.route('/books', methods=['POST'])
def returnTitle():
    text = request.json['text']
    text_json = {'data':  text}
    # here we could add an option to filter by length & popularity - "filter_args" respectively
    list_of_recs = recommendations(
        text, data, cosine_sim, filter_args=(None, None))
    return list_of_recs.to_json(orient='records')


def return_query_pull(query):
    ''' Takes in a string and returns a df with 50 most similar titles '''
    matching_books = []
    for k, v in data.iterrows():
        title = v.titles
        ratio_set = fuzz.token_set_ratio(title.lower(), query.lower())
        if ratio_set > 70:
            matching_books.append(k)

    return data.iloc[matching_books]


def recommendations(title, df, sim_matrix, filter_args=(None, None), list_length=11, suppress=True):
    recommended_books = []

    # creating a Series for the movie titles so they are associated to an ordered numerical list
    indices = pd.DataFrame(df.titles, index=df.index)

    # getting the index of the book that matches the title
    idx = indices[indices.titles == title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(sim_matrix[idx]).sort_values(ascending=False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:list_length+1].index)

    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_books.append(list(df.index)[i])

    if suppress == False:
        print(f"\n We recommend: ")
        for book_num in range(len(recommended_books)):
            print(book_num + 1, recommended_books[book_num])

    if filter_args != (None, None):
        return filter_df(filter_args[0], filter_args[1], df.loc[recommended_books])

    return df.loc[recommended_books]
