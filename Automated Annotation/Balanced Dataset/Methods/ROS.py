import pandas as pd
from imblearn.over_sampling import RandomOverSampler

# Load the dataset
file_path = '/Users/kanushreejaiswal/Desktop/Thesis/Automated Annotation/Labelled Data/UK/all_labelled_with_context.csv'
df = pd.read_csv(file_path)

# Select the columns for balancing
columns_to_balance = [
    'pro_brexit', 'anti_brexit', 'pro_climateAction', 'anti_climateAction', 
    'pro_NHS', 'anti_NHS', 'pro_israel', 'pro_palestine', 
    'pro_company_taxation', 'pro_worker_taxation', 'neutral', 'irrelevant'
]

# Create the feature matrix (X) and the target variable (y)
X = df.drop(columns=columns_to_balance)
y = df[columns_to_balance]

# Convert y to a single column representing the class labels
y_single_column = y.idxmax(axis=1)

# Apply Random Oversampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled_single_column = ros.fit_resample(X, y_single_column)

# Convert the resampled y back to the original multi-column format
y_resampled = pd.get_dummies(y_resampled_single_column)

# Combine the resampled features and target back into a single dataframe
df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

# Display the class distribution after resampling
class_distribution_after = y_resampled.sum().to_frame(name='count').reset_index()
print(class_distribution_after)

# Save the resampled dataframe
df_resampled.to_csv('/Users/kanushreejaiswal/Desktop/Thesis/Automated Annotation/Balanced Dataset/UK/ROS/allUK_withcontext_ROSbalanced.csv', index=False)