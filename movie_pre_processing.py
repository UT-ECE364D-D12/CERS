import pandas as pd


def movie_data_process (data):
    # Sort the data by user_id and timestamp
    data = data.sort_values(by=['user_id', 'timestamp'])

    # Initialize lists to store training and test data
    training_data = []
    test_data = []

    # Group the data by user_id
    grouped = data.groupby('user_id')

    # Iterate over each user's group
    for user_id, group in grouped:
        # Sort by timestamp (in case the data wasn't already sorted)
        group = group.sort_values(by='timestamp', ascending=True)
        
        # Separate the last movie (for test data) from the rest (for training data)
        movies_watched = list(zip(group['movie_id'], group['rating']))
        
        if len(movies_watched) > 1:
            # Add to training data (all but the last movie)
            training_data.append([user_id] + [movies_watched[:-1]])
            
            # Add to test data (the last movie)
            latest_movie = movies_watched[-1]
            test_data.append([user_id, latest_movie[0], latest_movie[1]])
        elif len(movies_watched) == 1:
            # If there's only one movie, put it in test data and skip training data
            latest_movie = movies_watched[0]
            test_data.append([user_id, latest_movie[0], latest_movie[1]])

    # Convert training data to DataFrame
    training_df = pd.DataFrame({
        'user_id': [row[0] for row in training_data],
        'movies_ratings': [row[1] for row in training_data]  # The list of (movie_id, rating) pairs
    })

    # Convert test data to DataFrame
    test_df = pd.DataFrame(test_data, columns=['user_id', 'movie_id', 'rating'])

    # Optionally save to CSV
    training_df.to_csv('trial_movie_training_data.csv', index=False)
    test_df.to_csv('trial_movie_test_data.csv', index=False)

if __name__ == "__main__":
    # Load the data
    data = pd.read_csv('data/ml-100k/u.data', sep='\t', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])

    # Process the data
    movie_data_process(data)
