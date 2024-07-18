import pandas as pd
from sklearn.preprocessing import StandardScaler
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import umap
import matplotlib.pyplot as plt

df = pd.read_csv('Analyses/User Data/usersUK_preprocessed.csv')

# Define the opinion columns
opinion_columns = [
    'pro_brexit', 'anti_brexit',
    'pro_climateAction', 'anti_climateAction',
    'pro_NHS', 'anti_NHS',
    'pro_israel', 'pro_palestine',
    'pro_company_taxation', 'pro_worker_taxation',
    'Brexit_neutral', 'ClimateChangeUK_neutral',
    'HealthcareUK_neutral', 'IsraelPalestineUK_neutral',
    'TaxationUK_neutral'
]

# Normalize the opinion columns
scaler = StandardScaler()
df[opinion_columns] = scaler.fit_transform(df[opinion_columns])

# Convert normalized opinion columns to a string format for BERTopic
df['opinions'] = df[opinion_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)

# Ensure the conversion to strings
print(df['opinions'])

# Create a CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['opinions'])

# Create and fit BERTopic model
topic_model = BERTopic()
topics, _ = topic_model.fit_transform(df['opinions'].tolist())  # Pass list of strings to BERTopic

# Add topics to DataFrame
df['topic'] = topics

# Display the resulting clusters
print(df)

# Step 3: Analyze and Visualize Results

# Topic distribution
topic_distribution = df['topic'].value_counts().sort_index()
print(topic_distribution)

# Optionally, display the topic distribution in a DataFrame for a better view
topic_distribution_df = topic_distribution.reset_index()
topic_distribution_df.columns = ['Topic', 'User Count']
print(topic_distribution_df)

# Calculate mean values of opinion columns for each cluster
cluster_means = df.groupby('topic')[opinion_columns].mean()
print(cluster_means)

# Reduce dimensionality for visualization
reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
embedding = reducer.fit_transform(df[opinion_columns])

# Plot the clusters
plt.figure(figsize=(10, 7))
plt.scatter(embedding[:, 0], embedding[:, 1], c=df['topic'], cmap='Spectral', s=50)
plt.colorbar(boundaries=np.arange(len(topic_distribution_df)+1)-0.5).set_ticks(np.arange(len(topic_distribution_df)))
plt.title('User Clusters based on Political Opinions')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.show()
