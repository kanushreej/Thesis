from datetime import datetime
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


user_data = pd.read_csv('Analyses/User Data/Clustered/usersUS_nr3.csv')

user_data['account_creation_date'] = pd.to_datetime(user_data['account_creation_date'], unit='s')
current_date = datetime.now()
user_data['duration'] = (current_date - user_data['account_creation_date']).dt.days


X = user_data['duration']
y = user_data['distance_to_center']

# Add a quadratic term
user_data['duration_squared'] = user_data['duration'] ** 2

# Add both the linear and quadratic terms to the model
X = user_data[['duration', 'duration_squared']]

# Add the intercept term)
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()
print(model.summary())

# Scatter plot
plt.scatter(user_data['duration'], user_data['distance_to_center'], label='Data points')

# Generate a range of duration values for plotting the regression line
duration_range = pd.DataFrame({'duration': range(int(user_data['duration'].min()), int(user_data['duration'].max()))})
duration_range['duration_squared'] = duration_range['duration'] ** 2
duration_range = sm.add_constant(duration_range)

# Plot the regression line
plt.plot(duration_range['duration'], model.predict(duration_range), color='red', label='Regression curve')
plt.xlabel('Duration (days)')
plt.ylabel('Distance to Center')
plt.title('Regression Analysis of Duration and Distance to Center with Quadratic Term')
plt.legend()
plt.show()
