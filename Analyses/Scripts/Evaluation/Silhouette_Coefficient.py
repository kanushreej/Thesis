'''
The silhouette coefficient is a metric that measures how well each data point fits into its assigned cluster. 
It combines information about both the cohesion (how close a data point is to other points in its own cluster) 
and the separation (how far a data point is from points in other clusters) of the data point.

Key Points:
1. The values range from -1 to 1
2. Here is what the values mean:- Close to 1: Well-clustered datapoint, 0: Overlapping Clusters, Close to -1: Missclassified Datapoint
3. Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.
4. The silhouette coefficient is calculated for each data point and then computes the average silhouette coefficient for the entire dataset.
5. The Silhouette Coefficient is calculated using the mean intra-cluster distance "a" and the mean nearest-cluster distance "b" for each sample. 
The Silhouette Coefficient for a sample is (b - a) / max(a, b). 
To clarify, b is the distance between a sample and the nearest cluster that the sample is not a part of. 
'''

import pandas as pd
from sklearn.metrics import silhouette_score

df_clustered = pd.read_csv('Analyses/User Data/usersUK_clustered_Kmeans.csv')

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


# Extract the opinions and the cluster labels
X = df_clustered[opinion_columns]
labels = df_clustered['topic']

# Calculate the Calinski-Harabasz score
ch_score = silhouette_score(X, labels)

print(f"Silhouette Score: {ch_score}")
