# References
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html
# https://www.tandfonline.com/doi/abs/10.1080/03610927408827101

# The Calinski-Harabasz score measures the ratio of the sum of between-cluster dispersion to the sum of within-cluster dispersion for all clusters. It is defined as:

#             (between-cluster dispersion matrix/(number of clusters-1))
#     -------------------------------------------------------------------------------       = CH
#     (within-cluster dispersion matrix/(number of data points - number of clusters))

# The between-cluster dispersion matrix measures how much the clusters are spread out relative to the overall mean of the data. 
# The within-cluster dispersion matrix measures how spread out the points are within each cluster.

# Numerator: This term represents the average between-cluster dispersion, normalized by the number of clusters minus one. 
# Higher values indicate that the clusters are well separated from each other.

# Denominator: This term represents the average within-cluster dispersion, normalized by the total number of data points minus the number of clusters. 
# Lower values indicate that the data points within each cluster are close to each other.

# A high Calinski-Harabasz score indicates that the clusters are:

# 1. Well-separated from each other (high between-cluster dispersion).
# 2. Compact within themselves (low within-cluster dispersion).

import pandas as pd
from sklearn.metrics import calinski_harabasz_score

df_clustered = pd.read_csv('Analyses/User Data/Clustered/usersUK_nr6.csv')
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


# Extract the opinions and the cluster labels
X = df_clustered[opinion_columns]
labels = df_clustered['topic']

# Calculate the Calinski-Harabasz score
ch_score = calinski_harabasz_score(X, labels)

print(f"Calinski-Harabasz Score: {ch_score}")
