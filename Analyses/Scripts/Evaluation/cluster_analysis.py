import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import cdist
from umap import UMAP

# Load the clustered user data
clustered_data_file_path = r'C:\Users\vshap\OneDrive\Desktop\work\code\Thesis\Thesis\Analyses\User Data\Clustered\usersUK_nr8.csv'
data = pd.read_csv(clustered_data_file_path)

region = 'UK'

if region == 'UK':
    opinion_columns = [
        'Brexit',
        'ClimateChangeUK',
        'HealthcareUK',
        'IsraelPalestineUK',
        'TaxationUK',
    ]
if region == 'US':
    opinion_columns = [
        'ImmigrationUS',
        'ClimateChangeUS',
        'HealthcareUS',
        'IsraelPalestineUS',
        'TaxationUS',
    ]

# Ensure the cluster column is numeric
data['cluster'] = pd.to_numeric(data['topic'])

# Calculate the cluster centers
cluster_centers = data.groupby('cluster')[opinion_columns].mean().values

# Calculate distances between cluster centers
distances_between_centers = cdist(cluster_centers, cluster_centers)

# Calculate distances from each opinion to the nearest cluster center
_, distances = pairwise_distances_argmin_min(cluster_centers, data[opinion_columns].values)

# Most popular ideologies in each cluster
popular_ideologies = data.groupby('cluster')[opinion_columns].mean()

# Collect cluster analysis data
cluster_analysis = []

for cluster in data['cluster'].unique():
    cluster_data = popular_ideologies.loc[cluster]
    absolute_values = cluster_data.abs()
    sorted_abs_values = absolute_values.sort_values(ascending=False)
    
    # Rank the opinions by their absolute values and print them with their original signs
    print(f"Cluster {cluster} ranked opinions (absolute values):")
    for topic in sorted_abs_values.index:
        abs_value = sorted_abs_values[topic]
        original_value = cluster_data[topic]
        print(f"{topic}: {abs_value} ({original_value})")
    print("\n")
    
    # Retrieve the original signs for the top 2 absolute values
    top_2_opinions = sorted_abs_values.index[:2]
    top_2_signed_values = cluster_data[top_2_opinions]

    cluster_size = len(data[data['cluster'] == cluster])
    center = cluster_centers[cluster]
    
    # Find nearest cluster
    nearest_cluster = distances_between_centers[cluster].argsort()[1]  # 0 is the cluster itself, so take the next one
    nearest_distance = distances_between_centers[cluster][nearest_cluster]
    
    cluster_analysis.append({
        'cluster': cluster,
        '1st_highest_opinion': top_2_signed_values.index[0],
        '1st_highest_value': top_2_signed_values.iloc[0],
        '2nd_highest_opinion': top_2_signed_values.index[1],
        '2nd_highest_value': top_2_signed_values.iloc[1],
        'cluster_size': cluster_size,
        'cluster_center': center,
        'nearest_cluster': nearest_cluster,
        'nearest_distance': nearest_distance
    })

# Create a DataFrame from the cluster analysis data
cluster_analysis_df = pd.DataFrame(cluster_analysis)

# Save the clustered data (though this should already be saved, this is just for confirmation)
clustered_data_output_path = 'Analyses/User Data/Clustered Data/UK/usersUK_nr8_updatedOpinions.csv'
data.to_csv(clustered_data_output_path, index=False)
print(f"Clustered data saved to {clustered_data_output_path}")
