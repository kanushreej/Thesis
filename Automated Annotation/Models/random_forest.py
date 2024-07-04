import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# Load the datasets
labelled_data_path = r"C:\Users\vshap\OneDrive\Desktop\allUK_withcontext_ROSbalanced.csv"
unlabelled_data_path = 'Annotation/UK/Viktor/Progress/Brexit_data.csv'

if not os.path.exists(labelled_data_path):
    raise FileNotFoundError(f"File not found: {labelled_data_path}")

if not os.path.exists(unlabelled_data_path):
    raise FileNotFoundError(f"File not found: {unlabelled_data_path}")

labelled_data = pd.read_csv(labelled_data_path)
unlabelled_data = pd.read_csv(unlabelled_data_path)

# Function to combine title and body
def combine_title_body(row):
    title = row['title'] if pd.notna(row['title']) else ""
    body = row['body'] if pd.notna(row['body']) else ""
    return f"{title} {body}".strip()

labelled_data['text'] = labelled_data.apply(combine_title_body, axis=1)
unlabelled_data['text'] = unlabelled_data.apply(combine_title_body, axis=1)

# Specify the target columns
target_columns = ['pro_brexit', 'anti_brexit', 'neutral', 'irrelevant']

# Columns to include in the final output before the predicted target columns
additional_columns = ['subreddit', 'type', 'keyword', 'id', 'author', 'title', 'body', 'created_utc']

# Separate features and target variables
X_labelled = labelled_data.drop(columns=target_columns)
y_labelled = labelled_data[target_columns]

# Ensure both DataFrames have the same columns
common_columns = list(set(X_labelled.columns).intersection(set(unlabelled_data.columns)))
X_labelled = X_labelled[common_columns]
unlabelled_features = unlabelled_data[common_columns]

# Encode categorical features in labelled data
encoders = {col: LabelEncoder() for col in X_labelled.columns if X_labelled[col].dtype == 'object'}
for col, encoder in encoders.items():
    X_labelled[col] = encoder.fit_transform(X_labelled[col].astype(str))

# Encode target variables in labelled data
target_encoders = {col: LabelEncoder() for col in y_labelled.columns}
for col, encoder in target_encoders.items():
    y_labelled[col] = encoder.fit_transform(y_labelled[col].astype(str))

# Split the labelled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_labelled, y_labelled, test_size=0.2, random_state=42)

# Train Random Forest models for each target with hyperparameter tuning
models = {
    col: RandomForestClassifier(
        n_estimators=100,
        max_depth=4,  # Control the depth of the tree
        min_samples_split=25,  # Minimum number of samples required to split an internal node
        min_samples_leaf=25,  # Minimum number of samples required to be at a leaf node
        max_features='sqrt',  # Number of features to consider when looking for the best split
        random_state=42
    ) for col in target_columns
}

for col in target_columns:
    y_train_temp = y_train[col]
    y_test_temp = y_test[col]
    
    # Use cross-validation for better evaluation
    cv_scores = cross_val_score(models[col], X_train, y_train_temp, cv=10)
    print(f'Cross-validation scores for {col}: {cv_scores}')
    print(f'Average cross-validation score for {col}: {cv_scores.mean()}')
    
    models[col].fit(X_train, y_train_temp)
    y_pred = models[col].predict(X_test)
    
    # Calculate and print metrics
    accuracy = accuracy_score(y_test_temp, y_pred)
    precision = precision_score(y_test_temp, y_pred, average='macro')
    recall = recall_score(y_test_temp, y_pred, average='macro')
    f1 = f1_score(y_test_temp, y_pred, average='macro')
    
    print(f'Metrics for {col}:')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print('--------------------------------')

# Preprocess the unlabelled dataset using the same encoders
for col, encoder in encoders.items():
    if col in unlabelled_features.columns:
        unlabelled_features[col] = unlabelled_data[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

# Predict the target values for the unlabelled dataset
predictions = {}
for col in target_columns:
    predictions[col] = models[col].predict(unlabelled_features)

# Convert predictions to DataFrame
predictions_df = pd.DataFrame(predictions)

# Include the specified columns in the final output
output_df_unlabelled = unlabelled_data[additional_columns].reset_index(drop=True)
output_df_unlabelled = pd.concat([output_df_unlabelled, predictions_df], axis=1)

# Save the predictions to a new CSV file
predictions_output_path = 'Annotation/UK/Viktor/Progress/Predicted_Brexit_Data_randomforest.csv'
output_df_unlabelled.to_csv(predictions_output_path, index=False)
print(f'Predictions saved to {predictions_output_path}')

# Save the labelled data with predictions for reference
labelled_predictions = y_labelled.copy()
for col in target_columns:
    labelled_predictions[col] = models[col].predict(X_labelled)

output_df_labelled = pd.concat([labelled_data[additional_columns].reset_index(drop=True), labelled_predictions], axis=1)
labelled_predictions_output_path = r"C:\Users\vshap\OneDrive\Desktop\allUK_withcontext_ROSbalanced_predictions_random_forest.csv"
output_df_labelled.to_csv(labelled_predictions_output_path, index=False)
print(f'Labelled data with predictions saved to {labelled_predictions_output_path}')
