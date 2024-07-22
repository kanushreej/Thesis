import pandas as pd
from datetime import datetime
import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv('Analyses/User Data/Clustered/usersUK_nr8.csv')

# Convert UNIX timestamp to datetime
df['account_creation_date'] = pd.to_datetime(df['account_creation_date'])

# Calculate current date
current_date = datetime.now()

# Calculate the number of days since account creation
df['days_since_creation'] = (current_date - df['account_creation_date']).dt.days

# Convert days to years
df['years_since_creation'] = df['days_since_creation'] / 365.25

# Print summary statistics
print("Summary statistics for account creation dates:")
print(df['years_since_creation'].describe())

# Define cohort boundaries correctly
cohort_boundaries = [0, 1, 5, 10, df['years_since_creation'].max()]

# Create cohorts with proper bin labels
cohort_labels = ['<1 year', '1-5 years', '5-10 years', '>10 years']
df['cohort'] = pd.cut(df['years_since_creation'], 
                      bins=cohort_boundaries, 
                      labels=cohort_labels,
                      right=False)

# Load the pre-clustered data
df = pd.read_csv('Analyses/User Data/Clustered/usersUK_nr8.csv')

# Define the opinion columns
opinion_columns = [
    'Brexit',
    'ClimateChangeUK',
    'HealthcareUK',
    'IsraelPalestineUK',
    'TaxationUK',
]

# Normalize the opinion columns (if not already normalized)
scaler = StandardScaler()
df[opinion_columns] = scaler.fit_transform(df[opinion_columns])

# Reduce dimensionality for visualization
reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
embedding = reducer.fit_transform(df[opinion_columns])

# Add UMAP dimensions to the DataFrame for plotting
df['UMAP1'] = embedding[:, 0]
df['UMAP2'] = embedding[:, 1]

# Define colors for each cohort
cohort_colors = {
    '<1 year': 'red',
    '1-5 years': 'green',
    '5-10 years': 'blue',
    '>10 years': 'purple'
}

# Plot the clusters
plt.figure(figsize=(15, 10))

for topic in df['topic'].unique():
    plt.figure(figsize=(10, 7))
    for cohort, color in cohort_colors.items():
        cohort_data = df[(df['topic'] == topic) & (df['cohort'] == cohort)]
        plt.scatter(cohort_data['UMAP1'], cohort_data['UMAP2'], 
                    c=color, 
                    label=f'{cohort}', 
                    s=50, alpha=0.6)
    
    # Plot cluster centers
    cluster_centers = pd.read_csv('Analyses/User Data/Clustered/cluster_centers.csv')
    embedding_centers = reducer.transform(cluster_centers)
    plt.scatter(embedding_centers[topic, 0], embedding_centers[topic, 1], 
                c='black', marker='X', s=200, label='Cluster Center')
    
    plt.legend()
    plt.title(f'Cluster {topic} with User Cohorts')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.show()

