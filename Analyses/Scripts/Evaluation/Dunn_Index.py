# References
# https://permetrics.readthedocs.io/en/stable/pages/clustering/DI.html
# https://www.datanovia.com/en/lessons/cluster-validation-statistics-must-know-methods/
# https://en.wikipedia.org/wiki/Dunn_index


# Dunn Dunn Index is a metric used to evaluate the quality of clustering. 
# It measures the compactness and separation of the clusters formed by a clustering algorithm. 
# A higher Dunn Index indicates better clustering performance.

#                     (a minimum inter-cluster distance)
# Dunn Index = ---------------------------------------------------
#                     (a maximum intra-cluster distance)

# Key Concepts:
# Intra-cluster Distance: For each cluster, calculate the maximum distance between all pairs of points within the cluster.
# Inter-cluster Distance: For each pair of clusters, calculate the minimum distance between points from the two different clusters.
# Dunn Index Formula: The ratio of the smallest inter-cluster distance to the largest intra-cluster distance.
# Range: The Dunn Index ranges from 0 to infinity, with higher values indicating better clustering quality.

import pandas as pd
import numpy as np
from permetrics import ClusteringMetric

# Load the data
df_clustered = pd.read_csv('usersUK_clustered_Kmeans10_trial.csv')

# Define the opinion columns
opinion_columns = [ # UK
    'pro_brexit', 'anti_brexit',
    'pro_climateAction', 'anti_climateAction',
    'pro_NHS', 'anti_NHS',
    'pro_israel', 'pro_palestine',
    'pro_company_taxation', 'pro_worker_taxation',
    'Brexit_neutral', 'ClimateChangeUK_neutral',
    'HealthcareUK_neutral', 'IsraelPalestineUK_neutral',
    'TaxationUK_neutral'
]

# opinion_columns = [   # US
#     'pro_immigration', 'anti_immigration',
#     'pro_climateAction', 'anti_climateAction',
#     'public_healthcare', 'private_healthcare',
#     'pro_israel', 'pro_palestine',
#     'pro_middle_low_tax', 'pro_wealthy_corpo_tax',
#     'ImmigrationUS_neutral', 'ClimateChangeUS_neutral',
#     'HealthcareUS_neutral', 'IsraelPalestineUS_neutral',
#     'TaxationUS_neutral',
# ]

def dunn(df_clustered):
    # Extract the feature values and cluster labels
    X = df_clustered[opinion_columns].values
    labels = df_clustered['topic'].values
    
    # Use permetrics to calculate the Dunn Index
    cm = ClusteringMetric(X=X, y_pred=labels)
    
    # Output Dunn Index
    dunn_index = cm.dunn_index()
    print(f"Dunn Index: {dunn_index}")
    
    # Alternatively, use the alias method
    dunn_index_alias = cm.DI()
    print(f"Dunn Index (alias method): {dunn_index_alias}")

dunn(df_clustered)
