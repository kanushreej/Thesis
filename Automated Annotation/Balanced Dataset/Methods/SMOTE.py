import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load the dataset
file_path = '/Users/kanushreejaiswal/Desktop/Thesis/Automated Annotation/Labelled Data/UK/all_labelled.csv'
df = pd.read_csv(file_path)

# Select the columns for balancing
columns_to_balance = [
    'pro_brexit', 'anti_brexit', 'pro_climateAction', 'anti_climateAction', 
    'pro_NHS', 'anti_NHS', 'pro_israel', 'pro_palestine', 
    'pro_company_taxation', 'pro_worker_taxation', 'neutral', 'irrelevant'
]

# Create the feature matrix (X) and the target variable (y)
X_numeric = df[columns_to_balance]
X_non_numeric = df.drop(columns=columns_to_balance)

# Convert non-numeric columns using Label Encoding
label_encoders = {}
for column in X_non_numeric.columns:
    if X_non_numeric[column].dtype == 'object':
        le = LabelEncoder()
        X_non_numeric[column] = le.fit_transform(X_non_numeric[column].astype(str))
        label_encoders[column] = le

# Convert y to a single column representing the class labels
y_single_column = X_numeric.idxmax(axis=1)

# Apply SMOTE to numeric columns only
smote = SMOTE(random_state=42)
X_resampled_numeric, y_resampled_single_column = smote.fit_resample(X_numeric, y_single_column)

# Find nearest neighbors to impute non-numeric columns
nn = NearestNeighbors(n_neighbors=1)
nn.fit(X_numeric)

_, indices = nn.kneighbors(X_resampled_numeric)
X_resampled_non_numeric = X_non_numeric.iloc[indices.flatten()].reset_index(drop=True)

# Decode non-numeric columns back to their original text form
for column in X_non_numeric.columns:
    if column in label_encoders:
        le = label_encoders[column]
        X_resampled_non_numeric[column] = le.inverse_transform(X_resampled_non_numeric[column])

# Combine resampled numeric and non-numeric columns
X_resampled = pd.concat([pd.DataFrame(X_resampled_numeric, columns=columns_to_balance), X_resampled_non_numeric], axis=1)

# Convert the resampled y back to the original multi-column format
y_resampled = pd.get_dummies(y_resampled_single_column)

# Ensure the combined DataFrame maintains the original structure and order
df_resampled = pd.concat([X_resampled_non_numeric, pd.DataFrame(X_resampled_numeric, columns=columns_to_balance)], axis=1)
for col in columns_to_balance:
    df_resampled[col] = y_resampled[col]

# Ensure the columns are in the same order as the original dataframe
df_resampled = df_resampled[X_non_numeric.columns.tolist() + columns_to_balance]

# Display the class distribution after resampling
class_distribution_after = y_resampled.sum().to_frame(name='count').reset_index()
print(class_distribution_after)

# Save the resampled dataframe
output_path = '/Users/kanushreejaiswal/Desktop/Thesis/Automated Annotation/Balanced Dataset/UK/SMOTE/allUK_SMOTEbalanced.csv'
df_resampled.to_csv(output_path, index=False)
print(f"Resampled data saved to {output_path}")

