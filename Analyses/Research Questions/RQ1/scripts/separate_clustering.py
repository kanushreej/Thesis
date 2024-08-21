import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import umap
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

df = pd.read_csv('Analyses/User Data/Preprocessed/usersUS_general.csv')
region = 'US'

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

df['account_creation_date'] = pd.to_datetime(df['account_creation_date'], unit='s')
current_date = datetime.now()
df['days_since_creation'] = (current_date - df['account_creation_date']).dt.days
df['years_since_creation'] = df['days_since_creation'] / 365.25


n_quantiles = 4
quantile_labels = [f'Quantile {i+1}' for i in range(n_quantiles)]
df['cohort'] = pd.qcut(df['years_since_creation'], q=n_quantiles, labels=quantile_labels)


quantile_ranges = df.groupby('cohort')['years_since_creation'].agg(['min', 'max', 'count'])
print("\nRange and distribution of years within each quantile group:")
print(quantile_ranges)


topic_model = BERTopic()

def cluster_cohort(cohort_df, cohort_label):

    scaler = StandardScaler()
    cohort_df[opinion_columns] = scaler.fit_transform(cohort_df[opinion_columns])

    cohort_df['opinions'] = cohort_df[opinion_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(cohort_df['opinions'])

    cluster_model = KMeans(n_clusters=3, random_state=42) 
    topic_model = BERTopic(hdbscan_model=cluster_model)

    topics, _ = topic_model.fit_transform(cohort_df['opinions'].tolist())  

    cohort_df['topic'] = topics

    distances = cluster_model.transform(cohort_df[opinion_columns])
    cohort_df['distance_to_center'] = [distances[i][topics[i]] for i in range(len(topics))]

    cohort_df.to_csv(f'Analyses/Research Questions/RQ1/Data/separate-clusters/usersUS_{cohort_label}.csv', index=False) 

    print(f"\nClustering results for {cohort_label}:")
    print(cohort_df)

    topic_distribution = cohort_df['topic'].value_counts().sort_index()
    print(topic_distribution)

    topic_distribution_df = topic_distribution.reset_index()
    topic_distribution_df.columns = ['Topic', 'User Count']
    print(topic_distribution_df)

    cluster_means = cohort_df.groupby('topic')[opinion_columns].mean()
    print(cluster_means)

    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
    embedding = reducer.fit_transform(cohort_df[opinion_columns])

    plt.figure(figsize=(10, 7))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=cohort_df['topic'], cmap='Spectral', s=50)
    plt.colorbar(boundaries=np.arange(len(topic_distribution_df)+1)-0.5).set_ticks(np.arange(len(topic_distribution_df)))
    plt.title(f'User Clusters based on Political Opinions ({cohort_label})')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.show()

    total_avg_distance_to_center = cohort_df['distance_to_center'].mean()
    print("\nTotal average distance to center:")
    print(total_avg_distance_to_center)

for cohort_label in quantile_labels:
    cohort_df = df[df['cohort'] == cohort_label].copy()
    if not cohort_df.empty:
        cluster_cohort(cohort_df, cohort_label)
