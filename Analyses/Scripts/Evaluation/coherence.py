from sklearn.metrics import silhouette_score

def evaluate_silhouette_score(df, opinion_columns):
    """
    The Silhouette Score is used to evaluate the quality of clustering.
    It measures how similar an object is to its own cluster compared to other clusters.
    A higher Silhouette Score indicates better clustering performance.

    Key points:
    1. **Silhouette Coefficient Calculation**: For each sample, calculate the mean distance 
       between the sample and all other points in the same cluster (a), and the mean distance 
       between the sample and all points in the nearest cluster (b).
    2. **Silhouette Score for a Sample**: The silhouette score for a sample is defined as 
       (b - a) / max(a, b). This score ranges from -1 to 1.
    3. **Average Silhouette Score**: The average silhouette score for all samples is computed 
       to evaluate the overall clustering quality.
    4. **Interpretation**: A silhouette score close to +1 indicates that samples are far from 
       the neighboring clusters, and thus well clustered. A score around 0 indicates overlapping 
       clusters, and a negative score indicates that samples might have been assigned to the 
       wrong clusters.
    """
    
    # Extract the features and cluster labels
    X = df[opinion_columns].values
    labels = df['topic'].values
    
    # Compute Silhouette Score
    silhouette_avg = silhouette_score(X, labels)
    return silhouette_avg
