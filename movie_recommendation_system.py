import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Load the data
data = pd.read_csv("tmdb_5000_movies.csv")
data2 = pd.read_csv("tmdb_5000_credits.csv")

# Merge the datasets
data2 = data2.drop(['title', 'movie_id'], axis=1)
data = data.join(data2)
data = data[['genres', 'keywords', 'original_title', 'overview', 'crew']]

# Handle missing values
data['overview'] = data['overview'].fillna('')

# Function to get the top 3 genres or keywords
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []

from ast import literal_eval

# Convert the 'genres' and 'keywords' columns to lists
features = ['genres', 'keywords']
for feature in features:
    data[feature] = data[feature].apply(literal_eval)

for i in features:
    data[i] = data[i].apply(get_list)

# Function to clean and preprocess the strings
def get_string(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

for i in features:
    data[i] = data[i].apply(get_string)

# Extract the director from the crew column
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

data['crew'] = data['crew'].apply(literal_eval)
data['director'] = data['crew'].apply(get_director)
data = data.drop(['crew'], axis=1)

# Combine genres, keywords, overview, and director into a single tag column
data['tag'] = data['genres'].apply(lambda x: ' '.join(x)) + ' ' + \
              data['keywords'].apply(lambda x: ' '.join(x)) + ' ' + \
              data['overview'] + ' ' + \
              data['director']

data = data.drop(['genres', 'keywords', 'overview', 'director'], axis=1)

# Text processing function
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('punkt')

stemmer = SnowballStemmer(language="english")
stop_words = set(stopwords.words("english"))
punct = string.punctuation

def process(s):
    for p in punct:
        s = s.replace(p, '')
    s = s.lower()
    s = word_tokenize(s)
    s = [w for w in s if w not in stop_words]
    s = [stemmer.stem(word) for word in s]
    return s

from tqdm import tqdm
for i in tqdm(data['tag'].index):
    processed_text = process(str(data.at[i, "tag"]))
    data.at[i, "tag"] = " ".join(processed_text)

# Convert text data to numerical data using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=10000, analyzer='word', ngram_range=(1, 1), stop_words='english')
bow = vectorizer.fit_transform(data['tag'])

# Compute the Cosine Similarity matrix
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(bow, bow)

# Save the cosine similarity matrix and the indices
import pickle
pickle.dump((cosine_sim, data['original_title']), open('model.pkl', 'wb'))
