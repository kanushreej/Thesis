import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import umap
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.utils import resample



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

# Calculate current date
current_date = datetime.now()

df['account_creation_date'] = pd.to_datetime(df['account_creation_date'], unit='s')
# Calculate the number of days since account creation
df['days_since_creation'] = (current_date - df['account_creation_date']).dt.days

# Convert days to years
df['years_active'] = df['days_since_creation'] / 365.25

df['tenure'] = pd.cut(df['years_active'], 
                      bins=[0, 1, 5, 10, float('inf')], 
                      labels=['<1 year', '1-5 years', '5-10 years', '>10 years'])

# Now proceed with the balancing
target_size = min(df[df['tenure'] == '<1 year'].shape[0], 
                  df[df['tenure'] == '1-5 years'].shape[0], 
                  df[df['tenure'] == '5-10 years'].shape[0], 
                  df[df['tenure'] == '>10 years'].shape[0])

# Resample each group
df_under_1 = resample(df[df['tenure'] == '<1 year'], replace=False, n_samples=target_size, random_state=42)
df_1_5 = resample(df[df['tenure'] == '1-5 years'], replace=False, n_samples=target_size, random_state=42)
df_5_10 = resample(df[df['tenure'] == '5-10 years'], replace=False, n_samples=target_size, random_state=42)
df_over_10 = resample(df[df['tenure'] == '>10 years'], replace=False, n_samples=target_size, random_state=42)

# Combine the resampled data
df_balanced = pd.concat([df_under_1, df_1_5, df_5_10, df_over_10])

# Proceed with the same clustering and analysis steps on the balanced dataset
scaler = StandardScaler()
df_balanced[opinion_columns] = scaler.fit_transform(df_balanced[opinion_columns])

df_balanced['opinions'] = df_balanced[opinion_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df_balanced['opinions'])

cluster_model = KMeans(n_clusters=8)
topic_model = BERTopic(hdbscan_model=cluster_model)

topics, _ = topic_model.fit_transform(df_balanced['opinions'].tolist())

df_balanced['topic'] = topics

distances = cluster_model.transform(df_balanced[opinion_columns])
df_balanced['distance_to_center'] = [distances[i][topics[i]] for i in range(len(topics))]

df_balanced.to_csv('Analyses/User Data/Clustered/usersUK_clustered_Kmeans_balanced.csv', index=False)

print(df_balanced)

topic_distribution = df_balanced['topic'].value_counts().sort_index()
print(topic_distribution)

topic_distribution_df = topic_distribution.reset_index()
topic_distribution_df.columns = ['Topic', 'User Count']
print(topic_distribution_df)

cluster_means = df_balanced.groupby('topic')[opinion_columns].mean()
print(cluster_means)

reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
embedding = reducer.fit_transform(df_balanced[opinion_columns])

plt.figure(figsize=(10, 7))
plt.scatter(embedding[:, 0], embedding[:, 1], c=df_balanced['topic'], cmap='Spectral', s=50)
plt.colorbar(boundaries=np.arange(len(topic_distribution_df)+1)-0.5).set_ticks(np.arange(len(topic_distribution_df)))
plt.title('User Clusters based on Political Opinions')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.show()