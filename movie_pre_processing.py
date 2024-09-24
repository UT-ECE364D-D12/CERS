import pandas as pd

def movie_data_process(data):
    # Sort the data by user_id and timestamp
    data = data.sort_values(by=['user_id', 'timestamp'])

    # Initialize lists to store processed training and test data
    processed_data = []
    test_data = []

    # Group the data by user_id
    grouped = data.groupby('user_id')

    # Iterate over each user's group
    for user_id, group in grouped:
        # Create a list of (movie_id, rating) pairs for the user
        movies_watched = list(zip(group['movie_id'], group['rating']))
        
        if len(movies_watched) > 1:
            # Append all but the last movie for training data
            processed_data.extend([[user_id, int(movie_id), int(rating)] for movie_id, rating in movies_watched[:-1]])
            
            # Get the last movie watched by the user for the test data
            last_movie = movies_watched[-1]
            test_data.append([user_id, int(last_movie[0]), int(last_movie[1])])

    # Convert processed training data to DataFrame
    processed_df = pd.DataFrame(processed_data, columns=['user_id', 'movie_id', 'rating'])

    # Convert test data to DataFrame
    test_df = pd.DataFrame(test_data, columns=['user_id', 'movie_id', 'rating'])

    # Save to CSV with no header and no index
    processed_df.to_csv('data/ml-100k/trial_movie_training_data.csv', index=False, header=False, sep=' ')
    test_df.to_csv('data/ml-100k/trial_movie_test_data.csv', index=False, header=False, sep=' ')

if __name__ == "__main__":
    # Load the data
    data = pd.read_csv('data/ml-100k/u.data', sep='\t', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])

    # Process the data
    movie_data_process(data)
