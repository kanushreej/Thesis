import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np

data = pd.read_csv('Analyses/User Data/Clustered/usersUK_nr8.csv')
data['account_creation_date'] = pd.to_datetime(data['account_creation_date'], unit='s')
current_date = datetime.now()


data['days_since_creation'] = (current_date - data['account_creation_date']).dt.days
data['years_since_creation'] = data['days_since_creation'] / 365.25


print("Summary statistics for account creation dates:")
print(data['years_since_creation'].describe())

print("\nSummary statistics for distance to center:")
print(data['distance_to_center'].describe())

# Scatter plot with line of best fit
plt.figure(figsize=(12, 8))
sns.scatterplot(x='years_since_creation', y='distance_to_center', data=data, alpha=0.5)
