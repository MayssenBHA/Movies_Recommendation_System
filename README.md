---

# 🎬 TMDB Movie Recommendation System

This project is a **content-based movie recommendation system** built using data from the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata). It recommends similar movies based on metadata such as genres, keywords, overview, and director.

## 🚀 Features

* Preprocessing of genres, keywords, and director information
* Text normalization using stemming and tokenization
* Vectorization using CountVectorizer
* Similarity calculation using Cosine Similarity
* Export of the trained similarity matrix for fast recommendations

## 📁 Dataset

The project uses two CSV files:

* `tmdb_5000_movies.csv`
* `tmdb_5000_credits.csv`

These datasets include information such as movie genres, overview, keywords, cast, and crew.

## ⚙️ How It Works

1. **Merge Datasets**: Combines movie and credit data.
2. **Extract Features**: Keeps only relevant columns: genres, keywords, overview, and director.
3. **Text Preprocessing**:

   * Removes punctuation
   * Converts text to lowercase
   * Removes stopwords
   * Applies stemming
4. **Vectorization**: Uses Bag of Words (CountVectorizer) to convert text into vectors.
5. **Similarity Calculation**: Computes cosine similarity between movies.
6. **Model Saving**: Exports the similarity matrix and titles into a `model.pkl` file using `pickle`.

## 📦 Dependencies

Install the following Python packages before running the project:

```bash
pip install numpy pandas scikit-learn nltk tqdm
```

You also need to download NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 🧠 How to Use

After training and saving the model in `model.pkl`, you can load it in a separate script or application to recommend movies:

```python
import pickle

cosine_sim, titles = pickle.load(open('model.pkl', 'rb'))

def recommend(movie_title):
    idx = titles[titles == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    recommended_titles = [titles[i[0]] for i in sim_scores]
    return recommended_titles
```

## 📌 Example

```python
recommend("The Dark Knight")
# Output: List of 5 movies similar to The Dark Knight
```

## 📚 Credits

* Data Source: [TMDB on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
* Developed using: Python, Pandas, NLTK, Scikit-learn

## 📄 License

This project is for educational purposes.

---

