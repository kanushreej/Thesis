from datetime import datetime
import pandas as pd
import statsmodels.api as sm


user_data = pd.read_csv(f'Analyses/User Data/Clustered/usersUK_nr8.csv')  # Load updated user data

# Drop rows with missing values in 'duration' or 'distance_to_center'
user_data.dropna(subset=['account_creation_date', 'distance_to_center'], inplace=True)

# Convert UNIX timestamp to datetime
user_data['account_creation_date'] = pd.to_datetime(user_data['account_creation_date'], unit='s')

# Calculate current date
current_date = datetime.now()

# Calculate the number of days since account creation
user_data['duration'] = (current_date - user_data['account_creation_date']).dt.days

# Prepare the data for regression
X = user_data['duration']
y = user_data['distance_to_center']

# Add a constant to the independent variable (this adds the intercept term)
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print the regression results
print(model.summary())

# Plot the results (optional)
import matplotlib.pyplot as plt

plt.scatter(user_data['duration'], user_data['distance_to_center'], label='Data points')
plt.plot(user_data['duration'], model.predict(X), color='red', label='Regression line')
plt.xlabel('Duration (seconds)')
plt.ylabel('Distance to Center')
plt.title('Regression Analysis of Duration and Distance to Center')
plt.legend()
plt.show()
