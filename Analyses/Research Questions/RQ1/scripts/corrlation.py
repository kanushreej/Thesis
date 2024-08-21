import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load the dataset
data = pd.read_csv('Analyses/User Data/Clustered/usersUS_nr3.csv')

# Convert UNIX timestamp to datetime
data['account_creation_date'] = pd.to_datetime(data['account_creation_date'], unit='s')

# Calculate current date
current_date = datetime.now()

# Calculate the number of days since account creation
data['days_since_creation'] = (current_date - data['account_creation_date']).dt.days

# Plotting the correlation graph with a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data['days_since_creation'], data['distance_to_center'], alpha=0.6)
plt.title('Correlation between Days Since Creation and Distance to Center')
plt.xlabel('Days Since Account Creation')
plt.ylabel('Distance to Center')
plt.grid(True)
plt.show()

correlation_coefficient = data['days_since_creation'].corr(data['distance_to_center'])
print(f"Correlation Coefficient between Days Since Creation and Distance to Center: {correlation_coefficient:.4f}")
