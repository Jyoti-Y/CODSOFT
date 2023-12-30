import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load movies data
movies = pd.read_csv('movies.csv')

# Preprocessing and feature extraction
movies['genre'] = movies['genre'].fillna('')
movies['description'] = movies['title'] + ' ' + movies['genre']

# TF-IDF Vectorization on movie descriptions (title + genres)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['description'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to recommend movies based on user input
def recommend_movies(input_title, cosine_sim=cosine_sim, movies=movies):
    try:
        idx = movies[movies['title'].str.lower() == input_title.lower()].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]  # Get top 5 similar movies

        movie_indices = [i[0] for i in sim_scores]

        return movies['title'].iloc[movie_indices]
    except IndexError:
        return "Movie not found or insufficient data for recommendations."
# Continuous loop for user input
while True:
    user_movie = input("Enter a movie title (or type 'exit' to quit): ")
    if user_movie.lower() == 'exit':
        print("Exiting the recommendation system. Goodbye!")
        break

    recommended_movies = recommend_movies(user_movie, cosine_sim, movies)
    print("Recommended Movies:")
    print(recommended_movies)

