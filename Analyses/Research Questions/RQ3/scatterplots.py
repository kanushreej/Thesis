'''import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('/Users/kanushreejaiswal/Desktop/RQ3/combined_csv_file/usersUK_nr8_preprocessed.csv')

# Extract the variables
X = data['total_karma']
y = data['distance_to_center']

# Calculate the line of best fit
m, b = np.polyfit(X, y, 1)  # m is the slope, b is the intercept

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.25, s=0.1, color='orange')

# Plot the line of best fit with a specified line width
plt.plot(X, m*X + b, color='black', linewidth=0.3)  # Adjust the value of linewidth to change the width

plt.title('Relationship between Total Karma and Polarization - US')
plt.xlabel('Total Karma')
plt.ylabel('Distance to Cluster Center')
plt.grid(True)
plt.show()'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load the data
data = pd.read_csv('/Users/kanushreejaiswal/Desktop/RQ3/combined_csv_file/usersUS_nr3_preprocessed.csv')

# Extract the variables
X = data['total_karma']
y = data['distance_to_center']

# Add a constant to the predictor variable
X_with_constant = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X_with_constant).fit()

# Get the predictions
predictions = model.predict(X_with_constant)

# Calculate the line of best fit
m, b = np.polyfit(X, y, 1)  # m is the slope, b is the intercept

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.25, s=1, color='orange')

# Plot the line of best fit with a specified line width
#plt.plot(X, m*X + b, color='black', linewidth=0.5)  # Adjust the value of linewidth to change the width

plt.title('Relationship between Total Karma and Polarization - US')
plt.xlabel('Total Karma')
plt.ylabel('Distance to Cluster Centre')
plt.grid(True)
plt.show()

# Print the regression summary
print(model.summary())
