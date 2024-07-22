import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import umap
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

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
if region == 'US':
    opinion_columns = [
        'ImmigrationUS',
        'ClimateChangeUS',
        'HealthcareUS',
        'IsraelPalestineUS',
        'TaxationUS',
    ]

# Convert UNIX timestamp to datetime
df['account_creation_date'] = pd.to_datetime(df['account_creation_date'], unit='s')

# Calculate current date
current_date = datetime.now()

# Calculate the number of days since account creation
df['days_since_creation'] = (current_date - df['account_creation_date']).dt.days

# Convert days to years
df['years_since_creation'] = df['days_since_creation'] / 365.25

# Define cohort boundaries correctly
cohort_boundaries = [0, 1, 5, 10, df['years_since_creation'].max()]

# Create cohorts with proper bin labels
cohort_labels = ['<1 year', '1-5 years', '5-10 years', '>10 years']
df['cohort'] = pd.cut(df['years_since_creation'], 
                      bins=cohort_boundaries, 
                      labels=cohort_labels,
                      right=False)

# Initialize BERTopic
topic_model = BERTopic()

# Function to cluster each cohort
def cluster_cohort(cohort_df, cohort_label):
    # Normalize the opinion columns within the cohort
    scaler = StandardScaler()
    cohort_df[opinion_columns] = scaler.fit_transform(cohort_df[opinion_columns])

    # Convert normalized opinion columns to a string 
    cohort_df['opinions'] = cohort_df[opinion_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)

    # Create a CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(cohort_df['opinions'])

    # Initialize KMeans
    cluster_model = KMeans(n_clusters=8, random_state=42) # Modify number of clusters
    topic_model = BERTopic(hdbscan_model=cluster_model)

    # Fit and transform
    topics, _ = topic_model.fit_transform(cohort_df['opinions'].tolist())  

    cohort_df['topic'] = topics

    # Calculate and save distance to cluster center for each user
    distances = cluster_model.transform(cohort_df[opinion_columns])
    cohort_df['distance_to_center'] = [distances[i][topics[i]] for i in range(len(topics))]

    # Save the clustered data to a CSV file
    cohort_df.to_csv(f'Analyses/User Data/Clustered/usersUK_{cohort_label}.csv', index=False) # Rename here

    print(cohort_df)

    # Topic distribution
    topic_distribution = cohort_df['topic'].value_counts().sort_index()
    print(topic_distribution)

    # Display the topic distribution in a DataFrame for a better view
    topic_distribution_df = topic_distribution.reset_index()
    topic_distribution_df.columns = ['Topic', 'User Count']
    print(topic_distribution_df)

    # Calculate mean values of opinion columns for each cluster
    cluster_means = cohort_df.groupby('topic')[opinion_columns].mean()
    print(cluster_means)

    # Reduce dimensionality for visualization
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
    embedding = reducer.fit_transform(cohort_df[opinion_columns])

    # Plot the clusters
    plt.figure(figsize=(10, 7))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=cohort_df['topic'], cmap='Spectral', s=50)
    plt.colorbar(boundaries=np.arange(len(topic_distribution_df)+1)-0.5).set_ticks(np.arange(len(topic_distribution_df)))
    plt.title(f'User Clusters based on Political Opinions ({cohort_label})')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.show()

    # Calculate and print the total average distance to center
    total_avg_distance_to_center = cohort_df['distance_to_center'].mean()
    print("\nTotal average distance to center:")
    print(total_avg_distance_to_center)

# Perform clustering for each cohort
for cohort_label in cohort_labels:
    cohort_df = df[df['cohort'] == cohort_label].copy()
    if not cohort_df.empty:
        cluster_cohort(cohort_df, cohort_label)
