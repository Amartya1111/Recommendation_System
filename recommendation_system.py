import pandas as pd
import numpy as np

# Sample movie ratings data
data = {
    'User': ['User1', 'User2', 'User3', 'User4', 'User5'],
    'Movie1': [5, 4, 3, np.nan, 5],
    'Movie2': [4, 5, np.nan, 4, 3],
    'Movie3': [3, np.nan, 5, 2, 4],
    'Movie4': [np.nan, 3, 4, 5, 2],
    'Movie5': [2, 5, 4, 3, 1]
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

from sklearn.metrics.pairwise import cosine_similarity

def find_similar_users(user_ratings):
    # Drop rows with missing values (NaN) and transpose the DataFrame
    user_ratings_clean = user_ratings.dropna(axis=0)
    user_ratings_transposed = user_ratings_clean.T

    # Calculate cosine similarity between users
    similarity_matrix = cosine_similarity(user_ratings_transposed)
    
    # Convert the similarity matrix to a DataFrame for better readability
    similar_users_df = pd.DataFrame(similarity_matrix, index=user_ratings_transposed.index, columns=user_ratings_transposed.index)
    
    return similar_users_df

# Calculate similarity between users based on their ratings
similar_users = find_similar_users(df.set_index('User'))

# Display the similarity matrix
print(similar_users)

def recommend_movies(user_ratings, similar_users, user):
    # Get the ratings of the specified user
    user_ratings = user_ratings.set_index('User').loc[user]
    
    # Drop movies already rated by the user
    unrated_movies = user_ratings[user_ratings.isnull()]
    
    # Calculate the mean rating given by the user
    user_mean_rating = user_ratings.mean()
    
    # Weighted sum of ratings from similar users
    weighted_sum = np.zeros(len(unrated_movies))
    similarity_sum = np.zeros(len(unrated_movies))
    
    for similar_user, similarity_score in similar_users[user].items():
        if np.isnan(similarity_score):  # Skip NaN values
            continue
        
        # Get ratings of the similar user
        similar_user_ratings = user_ratings.reset_index().set_index('User').loc[similar_user]
        
        # Calculate mean rating of the similar user
        similar_user_mean_rating = similar_user_ratings.mean()
        
        for i, movie_rating in enumerate(similar_user_ratings):
            if pd.isnull(movie_rating) or movie_rating == 0:
                continue
            
            # Calculate weighted sum and similarity sum
            weighted_sum[i] += similarity_score * (movie_rating - similar_user_mean_rating)
            similarity_sum[i] += similarity_score
    
    # Calculate predicted ratings
    predicted_ratings = user_mean_rating + (weighted_sum / similarity_sum)
    
    # Sort movies based on predicted ratings
    recommended_movies = predicted_ratings.sort_values(ascending=False)
    
    return recommended_movies

# Example: Recommend movies for User1
user = 'User1'
recommended_movies = recommend_movies(df, similar_users, user)

# Display recommended movies
print(recommended_movies)
