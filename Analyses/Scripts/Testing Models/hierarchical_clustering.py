import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import umap
import matplotlib.pyplot as plt

df = pd.read_csv('Analyses/User Data/usersUK_preprocessed.csv')

# Define the opinion columns
opinion_columns = [
    'pro_brexit', 'anti_brexit',
    'pro_climateAction', 'anti_climateAction',
    'pro_NHS', 'anti_NHS',
    'pro_israel', 'pro_palestine',
    'pro_company_taxation', 'pro_worker_taxation',
    'Brexit_neutral', 'ClimateChangeUK_neutral',
    'HealthcareUK_neutral', 'IsraelPalestineUK_neutral',
    'TaxationUK_neutral'
]

# Normalize the opinion columns
scaler = StandardScaler()
df[opinion_columns] = scaler.fit_transform(df[opinion_columns])

# Perform Hierarchical Clustering
n_clusters = 3  # Define the maximum number of clusters
hierarchical_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
df['cluster'] = hierarchical_clustering.fit_predict(df[opinion_columns])

# Display the resulting clusters
print(df)

# Step 3: Analyze and Visualize Results

# Cluster distribution
cluster_distribution = df['cluster'].value_counts().sort_index()
print(cluster_distribution)

# Optionally, display the cluster distribution in a DataFrame for a better view
cluster_distribution_df = cluster_distribution.reset_index()
cluster_distribution_df.columns = ['Cluster', 'User Count']
print(cluster_distribution_df)

# Calculate mean values of opinion columns for each cluster
cluster_means = df.groupby('cluster')[opinion_columns].mean()
print(cluster_means)

# Reduce dimensionality for visualization
reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
embedding = reducer.fit_transform(df[opinion_columns])

# Plot the clusters
plt.figure(figsize=(10, 7))
plt.scatter(embedding[:, 0], embedding[:, 1], c=df['cluster'], cmap='Spectral', s=50)
plt.colorbar(boundaries=np.arange(len(cluster_distribution_df)+1)-0.5).set_ticks(np.arange(len(cluster_distribution_df)))
plt.title('User Clusters based on Political Opinions')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.show()