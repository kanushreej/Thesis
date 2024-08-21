from datetime import datetime
import pandas as pd


df = pd.read_csv('Analyses/User Data/Clustered/usersUK_nr8.csv')
df['account_creation_date'] = pd.to_datetime(df['account_creation_date'], unit='s')

current_date = datetime.now()
df['days_since_creation'] = (current_date - df['account_creation_date']).dt.days
df['years_since_creation'] = df['days_since_creation'] / 365.25

quantile_labels = ['Quantile 1', 'Quantile 2', 'Quantile 3', 'Quantile 4']
df['quantile_group'] = pd.qcut(df['years_since_creation'], q=4, labels=quantile_labels)

# Calculate quantile distribution per cluster
quantile_distribution = pd.crosstab(index=df['topic'], columns=df['quantile_group'], normalize='index') * 100
print("Quantile Distribution per Cluster (%):")
print(quantile_distribution.round(2).to_string(index=True))

# Calculate probability of a quantile being in a cluster
probability_distribution = pd.crosstab(index=df['quantile_group'], columns=df['topic'], normalize='index') * 100
print("\nProbability of Quantile being in a Cluster (%):")
print(probability_distribution.round(2).to_string(index=True))

quantile_distribution.to_csv('quantile_distribution_per_cluster.csv', index=True)
probability_distribution.to_csv('probability_distribution_quantile_in_cluster.csv', index=True)
