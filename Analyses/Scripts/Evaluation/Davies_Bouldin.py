"""
The Davies-Bouldin Index (DBI) is used to evaluate the quality of the clustering.
It measures how well the clusters are separated and how compact the clusters are. 
A lower DBI indicates better clustering performance. 

Key points:
1. **Cluster Centroids**: Compute the centroid (mean vector) for each cluster.
2. **Cluster Dispersion**: Measure the average distance of all points in a cluster to the centroid of that cluster.
3. **Cluster Separation**: Measure the distance between the centroids of different clusters.
4. **DBI Calculation**: For each cluster, find the maximum ratio of the sum of dispersions to the separation between the cluster and any other cluster.
5. **DBI Formula**: The DBI is the average of these maximum ratios for all clusters.

"""

#This code is based on the structure of the BERTopic.py file 
from sklearn.metrics import davies_bouldin_score

def evaluate_davies_bouldin(df, opinion_columns):
    # Extract the features and cluster labels
    X = df[opinion_columns].values
    labels = df['topic'].values
    
    # Compute Davies-Bouldin
    db_index = davies_bouldin_score(X, labels)
    return db_index