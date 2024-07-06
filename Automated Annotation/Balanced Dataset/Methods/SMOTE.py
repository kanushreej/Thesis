import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = '/Users/kanushreejaiswal/Desktop/Thesis/Automated Annotation/Labelled Data/UK/Brexit_preprocessed.csv'
df = pd.read_csv(file_path)

# Function to convert string representation of list to numpy array
def str_to_array(s):
    return np.fromstring(s.strip("[]"), sep=' ')

# Combine text_vector and context_vector to form the feature set
features = df['text_vector'].apply(str_to_array).tolist()
context = df['context_vector'].apply(str_to_array).tolist()
X = np.array([np.concatenate((f, c)) for f, c in zip(features, context)])

# Target columns
targets = ['pro_brexit', 'anti_brexit', 'neutral', 'irrelevant']

# Initialize an array to hold the combined resampled targets
y_combined = np.array(df[targets])

# Apply SMOTE to the combined target array
smote = SMOTE()
X_resampled, y_resampled_combined = smote.fit_resample(X, y_combined)

# Split the resampled features back into text_vector and context_vector
text_vector_length = len(str_to_array(df['text_vector'].iloc[0]))
context_vector_length = len(str_to_array(df['context_vector'].iloc[0]))

text_vectors_resampled = X_resampled[:, :text_vector_length]
context_vectors_resampled = X_resampled[:, text_vector_length:]

# Create a new DataFrame with the resampled data
resampled_data = pd.DataFrame()
resampled_data['text_vector'] = list(text_vectors_resampled)
resampled_data['context_vector'] = list(context_vectors_resampled)

# Add the target columns
for i, target in enumerate(targets):
    resampled_data[target] = y_resampled_combined[:, i]

# Save the resampled dataset to a CSV file
resampled_data.to_csv('/Users/kanushreejaiswal/Desktop/Brexit_resampled.csv', index=False)

# Display the first few rows of the resampled dataset
print(resampled_data.head())

