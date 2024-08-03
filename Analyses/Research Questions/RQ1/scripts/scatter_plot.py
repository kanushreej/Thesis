import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the dataset
data = pd.read_csv('Analyses/User Data/Clustered/usersUK_nr8.csv')

# Convert account_creation_date to datetime
data['account_creation_date'] = pd.to_datetime(data['account_creation_date'], unit='s')

# Calculate current date
current_date = datetime.now()

# Calculate the number of days since account creation
data['days_since_creation'] = (current_date - data['account_creation_date']).dt.days

# Convert days to years
data['years_since_creation'] = data['days_since_creation'] / 365.25

# Print summary statistics for verification
print("Summary statistics for account creation dates:")
print(data['years_since_creation'].describe())

print("\nSummary statistics for distance to center:")
print(data['distance_to_center'].describe())

# Scatter plot with line of best fit
plt.figure(figsize=(12, 8))
sns.scatterplot(x='years_since_creation', y='distance_to_center', data=data, alpha=0.5)

# Fit a linear regression model
X = data['years_since_creation'].values.reshape(-1, 1)
y = data['distance_to_center'].values
model = LinearRegression()
model.fit(X, y)
line = model.predict(X)

# Plot the line of best fit
plt.plot(data['years_since_creation'], line, color='red')

plt.title('Scatter Plot of Account Creation Date vs Distance to Cluster Center with Best Fit Line')
plt.xlabel('Years Since Account Creation')
plt.ylabel('Distance to Cluster Center')
plt.show()
