import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Read the data
file_path = '/content/drive/MyDrive/usersUK_nr8.csv' 
df = pd.read_csv(file_path)

# Extract features and target variable
X = df[['Brexit', 'ClimateChangeUK', 'HealthcareUK', 'IsraelPalestineUK', 'TaxationUK']]
y = df['distance_to_center']

# US
# Read the data
#file_path = '/content/drive/MyDrive/usersUS_nr3.csv' 
#df = pd.read_csv(file_path)

# Extract features and target variable
#X = df[['ImmigrationUS', 'ClimateChangeUS', 'HealthcareUS', 'IsraelPalestineUS', 'TaxationUS']]
#y = df['distance_to_center']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model with the best parameters
best_rf = RandomForestRegressor(max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=300, random_state=42)
best_rf.fit(X_train, y_train)

# Validate the model
y_pred = best_rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Get feature importances
feature_importances = best_rf.feature_importances_
feature_names = X.columns
importances_dict = dict(zip(feature_names, feature_importances))

# Convert importances to percentages
total_importance = sum(importances_dict.values())
importances_dict_percentage = {k: (v / total_importance) * 100 for k, v in importances_dict.items()}

# Sort the feature importances in descending order
sorted_importances = sorted(importances_dict_percentage.items(), key=lambda x: x[1], reverse=True)
sorted_feature_names = [item[0] for item in sorted_importances]
sorted_importance_values = [item[1] for item in sorted_importances]

# Plot feature importances as percentages (Bar Chart)
plt.figure(figsize=(10, 6))
bars = plt.bar(sorted_feature_names, sorted_importance_values, color='skyblue')
bars[0].set_color('orange')  # Highlight the most important feature
plt.title('Feature Importances (Percentage)')
plt.ylabel('Importance (%)')
plt.xlabel('Features')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot feature importances as percentages (Pie Chart)
plt.figure(figsize=(8, 8))
plt.pie(sorted_importance_values, labels=sorted_feature_names, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title('Feature Importances (Percentage)')
plt.tight_layout()
plt.show()

# Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.3)
plt.hlines(0, min(y_pred), max(y_pred), colors='red')
plt.title('Residuals')
plt.ylabel('Residuals')
plt.xlabel('Predicted Values')
plt.tight_layout()
plt.show()

# Plot density of actual vs predicted values
plt.figure(figsize=(10, 6))
sns.kdeplot(y_test, label='Actual Values', fill=True, color='skyblue')
sns.kdeplot(y_pred, label='Predicted Values', fill=True, color='orange')
plt.title('Density of Actual vs Predicted Values')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

# Plot predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.title('Predicted vs Actual Values')
plt.ylabel('Predicted Values')
plt.xlabel('Actual Values')
plt.tight_layout()
plt.show()
