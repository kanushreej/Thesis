import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

# Load the data
data = pd.read_csv('/Users/kanushreejaiswal/Desktop/RQ3/combined_csv_file/usersUK_nr8_preprocessed.csv')

# Extract the variables
X = data['total_karma']
y = data['distance_to_center']

# Add a constant to the predictor variable
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Get the predictions
predictions = model.predict(X)

# Plot the data and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(data['total_karma'], data['distance_to_center'], alpha=0.25, s=0.1, color='orange')
plt.plot(data['total_karma'], predictions, color='black')
plt.title('Relationship between Total Karma and Polarization - US')
plt.xlabel('Total Karma')
plt.ylabel('Distance to Cluster Center')
plt.grid(True)
plt.show()
