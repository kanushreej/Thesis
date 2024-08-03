import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the top users' versions data with distance to cluster center
top_users_versions_with_distance = pd.read_csv('Analyses/Research Questions/RQ1/Data/top_users_versions_with_distance.csv')

# Sort data by username and version_date to ensure correct order
top_users_versions_with_distance = top_users_versions_with_distance.sort_values(by=['username', 'version_date'])

# Add a 'version_number' column to indicate the sequential order of versions for each user
top_users_versions_with_distance['version_number'] = top_users_versions_with_distance.groupby('username').cumcount() + 1

# Get the unique usernames of the top 20 users
top_20_usernames = top_users_versions_with_distance['username'].unique()

# Define the number of users to plot at a time
users_per_plot = 5
num_plots = int(np.ceil(len(top_20_usernames) / users_per_plot))

# Plot distance to cluster center over sequential versions for users
for i in range(num_plots):
    plt.figure(figsize=(12, 8))
    
    for username in top_20_usernames[i * users_per_plot : (i + 1) * users_per_plot]:
        user_data = top_users_versions_with_distance[top_users_versions_with_distance['username'] == username]
        plt.plot(user_data['version_number'], user_data['distance_to_center'], label=username)
    
    plt.xlabel('Version Number')
    plt.ylabel('Distance to Cluster Center')
    plt.title(f'Distance to Cluster Center Over Sequential Versions for Users {i * users_per_plot + 1} to {(i + 1) * users_per_plot}')
    plt.legend(title='User')
    plt.grid(True)
    plt.show()