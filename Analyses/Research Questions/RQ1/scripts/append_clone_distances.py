import pandas as pd

# Load the clustered data
clustered_data = pd.read_csv('Analyses/Research Questions/RQ1/Data/UK_clones.csv')

# Load the original top 20 users' versions data
top_users_versions_df = pd.read_csv('Analyses/Research Questions/RQ1/Data/topUK_users_preprocessed.csv')

# Extract the necessary columns from the clustered data
clustered_data_relevant = clustered_data[['username', 'version_date', 'distance_to_center']]

# Merge the distance to cluster center with the top users' versions data
top_users_versions_with_distance = pd.merge(top_users_versions_df, clustered_data_relevant, 
                                            on=['username', 'version_date'], 
                                            how='left')
# Save the updated top users' versions data to a CSV file
output_file = 'Analyses/Research Questions/RQ1/Data/top_users_versions_with_distance.csv'
top_users_versions_with_distance.to_csv(output_file, index=False)

print(f"Top users' versions with distance to cluster center have been saved to {output_file}")
