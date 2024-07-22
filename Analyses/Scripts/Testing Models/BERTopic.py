import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import umap
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('Analyses/User Data/Preprocessed/usersUK_general.csv')
region = 'UK'

if region == 'UK':
    opinion_columns = [
        'Brexit',
        'ClimateChangeUK',
        'HealthcareUK',
        'IsraelPalestineUK',
        'TaxationUK',
    ]
elif region == 'US':
    opinion_columns = [
        'ImmigrationUS',
        'ClimateChangeUS',
        'HealthcareUS',
        'IsraelPalestineUS',
        'TaxationUS',
    ]

# Normalize the opinion columns
scaler = StandardScaler()
df[opinion_columns] = scaler.fit_transform(df[opinion_columns])

# Convert normalized opinion columns to a string 
df['opinions'] = df[opinion_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)

# Create a CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['opinions'])

cluster_model = KMeans(n_clusters=8, random_state=42) # Modify number of clusters
topic_model = BERTopic(hdbscan_model=cluster_model)

# Fit and transform
topics, _ = topic_model.fit_transform(df['opinions'].tolist())  

df['topic'] = topics

# Calculate and save distance to cluster center for each user
distances = cluster_model.transform(df[opinion_columns])
df['distance_to_center'] = [distances[i][topics[i]] for i in range(len(topics))]

# Save the clustered data to a CSV file
df.to_csv('Analyses/User Data/Clustered/usersUK_nr8.csv', index=False) # Rename here

# Save the cluster centers
cluster_centers = cluster_model.cluster_centers_
cluster_centers_df = pd.DataFrame(cluster_centers, columns=opinion_columns)
cluster_centers_df.to_csv('Analyses/User Data/Clustered/UK_cluster_centers.csv', index=False)

'''
Analyses (can be uncommented)
# Plotting clusters (for verification)
reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
embedding = reducer.fit_transform(df[opinion_columns])

plt.figure(figsize=(10, 7))
plt.scatter(embedding[:, 0], embedding[:, 1], c=df['topic'], cmap='Spectral', s=50)
plt.colorbar(boundaries=np.arange(len(df['topic'].unique())+1)-0.5).set_ticks(np.arange(len(df['topic'].unique())))
plt.title('User Clusters based on Political Opinions')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.show()

print(df)

# Topic distribution
topic_distribution = df['topic'].value_counts().sort_index()
print(topic_distribution)

# Display the topic distribution in a DataFrame for a better view
topic_distribution_df = topic_distribution.reset_index()
topic_distribution_df.columns = ['Topic', 'User Count']
print(topic_distribution_df)

# Calculate mean values of opinion columns for each cluster
cluster_means = df.groupby('topic')[opinion_columns].mean()
print(cluster_means)
'''