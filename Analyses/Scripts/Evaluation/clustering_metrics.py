import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from permetrics import ClusteringMetric

df = pd.read_csv('Analyses/Research Questions/RQ1/Data/usersUK_5-10 years.csv')
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

X = df[opinion_columns]
labels = df['topic']

"""
The Calinski-Harabasz score measures the ratio of the sum of between-cluster dispersion to the sum of within-cluster dispersion for all clusters. It is defined as:

            (between-cluster dispersion matrix/(number of clusters-1))
    -------------------------------------------------------------------------------       = CH
    (within-cluster dispersion matrix/(number of data points - number of clusters))

The between-cluster dispersion matrix measures how much the clusters are spread out relative to the overall mean of the data. 
The within-cluster dispersion matrix measures how spread out the points are within each cluster.

Numerator: This term represents the average between-cluster dispersion, normalized by the number of clusters minus one. 
Higher values indicate that the clusters are well separated from each other.

Denominator: This term represents the average within-cluster dispersion, normalized by the total number of data points minus the number of clusters. 
Lower values indicate that the data points within each cluster are close to each other.

A high Calinski-Harabasz score indicates that the clusters are:

1. Well-separated from each other (high between-cluster dispersion).
2. Compact within themselves (low within-cluster dispersion).

References
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html
https://www.tandfonline.com/doi/abs/10.1080/03610927408827101
"""
# Calculate the Calinski-Harabasz score
ch_score = calinski_harabasz_score(X, labels)
print(f"Calinski-Harabasz Score: {ch_score}")

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
# Calculate the Davies-Bouldin Index
db_index = davies_bouldin_score(X, labels)
print(f"Davies-Bouldin Index: {db_index}")


"""
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
"""
# Calculate the Silhouette Coefficient
ch_score = silhouette_score(X, labels)
print(f"Silhouette Score: {ch_score}")

"""
Dunn Dunn Index is a metric used to evaluate the quality of clustering. 
It measures the compactness and separation of the clusters formed by a clustering algorithm. 
A higher Dunn Index indicates better clustering performance.

                    (a minimum inter-cluster distance)
Dunn Index = ---------------------------------------------------
                    (a maximum intra-cluster distance)

Key Concepts:
Intra-cluster Distance: For each cluster, calculate the maximum distance between all pairs of points within the cluster.
Inter-cluster Distance: For each pair of clusters, calculate the minimum distance between points from the two different clusters.
Dunn Index Formula: The ratio of the smallest inter-cluster distance to the largest intra-cluster distance.
Range: The Dunn Index ranges from 0 to infinity, with higher values indicating better clustering quality.

References
https://permetrics.readthedocs.io/en/stable/pages/clustering/DI.html
https://www.datanovia.com/en/lessons/cluster-validation-statistics-must-know-methods/
https://en.wikipedia.org/wiki/Dunn_index
"""
# Output Dunn Index

X = df[opinion_columns].values
labels = df['topic'].values

cm = ClusteringMetric(X=X, y_pred=labels)    
dunn_index = cm.dunn_index()
print(f"Dunn Index: {dunn_index}")