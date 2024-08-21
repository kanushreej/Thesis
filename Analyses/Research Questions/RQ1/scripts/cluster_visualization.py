from datetime import datetime
import pandas as pd
import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the pre-clustered data
df = pd.read_csv('Analyses/User Data/Clustered/usersUS_with_clusters.csv')

# Convert UNIX timestamp to datetime
df['account_creation_date'] = pd.to_datetime(df['account_creation_date'], unit='s')

# Calculate current date
current_date = datetime.now()

# Calculate the number of days since account creation
df['days_since_creation'] = (current_date - df['account_creation_date']).dt.days

# Convert days to years
df['years_since_creation'] = df['days_since_creation'] / 365.25

# Print summary statistics
print("Summary statistics for account creation dates:")
print(df['years_since_creation'].describe())

# Create quantiles
quantile_labels = ['Quantile 1', 'Quantile 2', 'Quantile 3', 'Quantile 4']
df['quantile_group'] = pd.qcut(df['years_since_creation'], 
                               q=4, 
                               labels=quantile_labels)

# Define the opinion columns
# opinion_columns = [
#     'Brexit',
#     'ClimateChangeUK',
#     'HealthcareUK',
#     'IsraelPalestineUK',
#     'TaxationUK',
# ]

opinion_columns = [
    'ImmigrationUS',
    'ClimateChangeUS',
    'HealthcareUS',
    'IsraelPalestineUS',
    'TaxationUS',
]

# Normalize the opinion columns (if not already normalized)
scaler = StandardScaler()
df[opinion_columns] = scaler.fit_transform(df[opinion_columns])

# Reduce dimensionality for visualization
reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
embedding = reducer.fit_transform(df[opinion_columns])

# Add UMAP dimensions to the DataFrame for plotting
df['UMAP1'] = embedding[:, 0]
df['UMAP2'] = embedding[:, 1]

# Define colors for each quantile group
quantile_colors = {
    'Quantile 1': 'red',
    'Quantile 2': 'green',
    'Quantile 3': 'blue',
    'Quantile 4': 'purple'
}

# Load cluster centers
cluster_centers = pd.read_csv('Analyses/User Data/Clustered/US_cluster_centers.csv')
embedding_centers = reducer.transform(cluster_centers)

# Plot the clusters
for topic in df['topic'].unique():
    plt.figure(figsize=(10, 7))
    for quantile, color in quantile_colors.items():
        quantile_data = df[(df['topic'] == topic) & (df['quantile_group'] == quantile)]
        plt.scatter(quantile_data['UMAP1'], quantile_data['UMAP2'], 
                    c=color, 
                    label=f'{quantile}', 
                    s=50, alpha=0.6)
    
    plt.scatter(embedding_centers[topic, 0], embedding_centers[topic, 1], 
                c='black', marker='X', s=200, label='Cluster Center')
    
    plt.legend()
    plt.title(f'Cluster {topic} with User Quantile Groups')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.show()
