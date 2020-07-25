# -*- coding: utf-8 -*-
"""
Created on Fri May 28 08:39:30 2020

@author: TANISHQ
"""
import numpy as np
import pandas as pd
# Importing csv files
d1 = pd.read_csv('credits.csv')
d2 = pd.read_csv('movies_metadata.csv')
d3 = pd.read_csv('keywords.csv')
# Printing the first 5 values of the datasets
d1.head()
d2.head()
d3.head()
# Removing invalid id's of wrong format
d2 = d2[d2.id!= '1997-08-20']
d2 = d2[d2.id!= '2012-09-29']
d2 = d2[d2.id!= '2014-01-01']
# Converting id values in d2 from string type to int
d2.id = d2.id.astype(int)
# Merging the data from the csv files into one 
data = d2.merge(d1, on ='id')
data = data.merge(d3, on = 'id')
# Printing the final columns in the dataset
data.columns
unused = ['adult','budget','homepage','imdb_id', 'poster_path','production_countries','revenue','runtime','status','video']
data = data.drop(unused,axis=1)
data = data[:10000]
data.overview = data.overview.astype(str)
# Extracting features from the dictionary type data
from ast import literal_eval
features = ['cast', 'crew', 'keywords', 'genres']
for i in features:
    data[i] = data[i].apply(literal_eval)
# Printing the first 5 values of the final data
data.head()
# Function to return director name from the crew column
def director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
# Function to get genre, cast, keywords from the data
def feat(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
# Check if more than 3 elements exist. If yes, return only first three. If no, return entire list
        if len(names) > 3:
            names = names[:3]
        return names
# Return empty list in case of missing data
    return []
# Define director, cast, genres and keywords features
data['director'] = data['crew'].apply(lambda x:director(x))
features = ['cast', 'keywords', 'genres']
for feature in features:
    data[feature] = data[feature].apply(feat)
# Printing the data with the respective features
data[['title', 'cast', 'director', 'keywords', 'genres']].head()  
# Function to convert all strings to lower case and remove spaces
def clean(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
#Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
# Apply clean data function to the features
feature = ['cast', 'keywords', 'director', 'genres']
for i in feature:
    data[i] = data[i].apply(lambda x: clean(x))
# Concatinate all the important features
def create(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] +' ' + ' '.join(x['genres']) + ' ' + (x['overview'])
data['conca'] = data.apply(create, axis=1)
#Printing the new concatinated features
data['conca']
# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(data.conca)
# Calculating Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(count_matrix, count_matrix)
# Constructing a series of index number and title
ind = pd.Series(data.index, index=data['title'])
# Function to get movie recommendation
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = ind[title]
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return data['title'].iloc[movie_indices]
get_recommendations('Rocky')

