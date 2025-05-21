import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import sys
import io

# Ensure the environment uses UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = Flask(__name__)

# Load the model and other necessary data
with open('model/model.pkl', 'rb') as f:
    cosine_sim, titles = pickle.load(f)

indices = pd.Series(titles.index, index=titles).drop_duplicates()

# Function to get recommendations based on movie title
def get_recommendations(original_title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[original_title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 30 most similar movies
    sim_scores = sim_scores[1:51]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 30 most similar movies as a list
    return titles.iloc[movie_indices].tolist()

sorted_titles = titles.sort_values().tolist()

def final_recommendations(L1, L2):
    # Find the common movies in both lists
    common_movies = [movie for movie in L1 if movie in L2]
    return common_movies

# Home route
@app.route('/')
def home():
    return render_template('index.html', titles=sorted_titles)

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the movie titles from the form
    movie_title1 = request.form['movie_title1']
    recommendations1 = get_recommendations(movie_title1, cosine_sim=cosine_sim)
    movie_title2 = request.form['movie_title2']
    recommendations2 = get_recommendations(movie_title2, cosine_sim=cosine_sim)
    common_movies = final_recommendations(recommendations1, recommendations2)
    if len(common_movies) == 0 :
        return render_template('Predection.html', predection_text = 'there is no recommended movies')

    return render_template('Predection.html', common_movies=common_movies)

if __name__ == "__main__":
    app.run(debug=True)
