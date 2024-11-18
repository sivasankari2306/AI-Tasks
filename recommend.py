import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample data
data = {
    "User": ["User1", "User2", "User3"],
    "Movie1": [5, 4, 1],
    "Movie2": [4, 0, 5],
    "Movie3": [1, 5, 4],
}
df = pd.DataFrame(data).set_index("User")

# Calculate cosine similarity
similarity = cosine_similarity(df)
similarity_df = pd.DataFrame(similarity, index=df.index, columns=df.index)

# Recommend based on similarity
def recommend(user):
    similar_users = similarity_df[user].sort_values(ascending=False).index[1:]
    return f"Recommendations for {user}: Movies liked by {similar_users[0]}."

print(recommend("User1"))

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Sample dataset of movies
data = {
    "MovieID": [1, 2, 3, 4, 5],
    "Title": ["The Matrix", "Inception", "Interstellar", "The Dark Knight", "The Avengers"],
    "Genre": ["Sci-Fi Action", "Sci-Fi Thriller", "Sci-Fi Adventure", "Action Thriller", "Action Superhero"],
}

# Create a DataFrame
movies_df = pd.DataFrame(data)

# Combine all text features into a single string for each movie
movies_df["Combined"] = movies_df["Title"] + " " + movies_df["Genre"]

# Convert textual data into numerical data using CountVectorizer
vectorizer = CountVectorizer()
feature_matrix = vectorizer.fit_transform(movies_df["Combined"])

# Calculate similarity scores using cosine similarity
similarity_scores = cosine_similarity(feature_matrix, feature_matrix)

# Function to recommend movies
def recommend_movies(movie_title, num_recommendations=3):
    # Find the index of the movie in the dataset
    movie_idx = movies_df[movies_df["Title"] == movie_title].index[0]

    # Get similarity scores for the selected movie
    movie_similarities = list(enumerate(similarity_scores[movie_idx]))

    # Sort movies by similarity scores in descending order
    sorted_movies = sorted(movie_similarities, key=lambda x: x[1], reverse=True)

    # Get top recommendations (excluding the selected movie itself)
    recommendations = [movies_df.iloc[i[0]]["Title"] for i in sorted_movies[1:num_recommendations + 1]]

    return recommendations

# Test the recommendation system
selected_movie = "The Matrix"
print(f"Movies similar to '{selected_movie}':")
print(recommend_movies(selected_movie))


