import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVC
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier

# Base directory and file paths
base_directory = "C:/Users/rapha/Documents/CS_VU/Thesis"
moderator_name = "Raphael"
issue = "Brexit"
team = "UK"

# Load full data containing the text of all posts/comments
full_data = pd.read_csv(f"{base_directory}/Thesis/Subreddit Data/{team}/{issue}_data.csv")

# Convert full data to a dictionary for quick lookup
id_to_text = pd.Series(full_data.body.values, index=full_data.id).to_dict()

# Function to fetch parent text based on parent_id
def fetch_parent_text(parent_id):
    return id_to_text.get(parent_id, "")

# Load labeled data
labeled_data = pd.read_csv(f"{base_directory}/Thesis/Automated Annotation/Balanced Dataset/{team}/ROS/all{team}_ROSbalanced.csv")

# Fetch and merge parent post text with comment text
def merge_parent_post(data):
    data['parent_text'] = data['parent_id'].apply(fetch_parent_text)
    data['merged_text'] = data['body'].fillna('') + ' ' + data['parent_text'].fillna('')
    return data

print('Merging data for training..')
labeled_data = merge_parent_post(labeled_data)

# Define feature columns and target columns
feature_columns = ['pro_brexit', 'anti_brexit', 'pro_climateAction', 'anti_climateAction', 'pro_NHS', 'anti_NHS', 'pro_israel', 'pro_palestine', 'pro_company_taxation', 'pro_worker_taxation']
target_columns = ['pro_brexit', 'anti_brexit', 'neutral', 'irrelevant']

# Extract features and targets from labeled data
X_features = labeled_data[feature_columns]
y = labeled_data[target_columns]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# Vectorize the merged text
vectorizer = TfidfVectorizer(max_features=1000)
X_text = vectorizer.fit_transform(labeled_data['merged_text']).toarray()

# Combine text features with other features
X_combined = np.hstack((X_scaled, X_text))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Define the nu-SVM model
svm = NuSVC()

# Define the parameter grid
param_grid = {
    'estimator__nu': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    'estimator__kernel': ['linear', 'rbf', 'poly'],
    'estimator__gamma': ['scale', 'auto']
}

# Set up GridSearchCV with 10-fold cross-validation
grid_search = GridSearchCV(estimator=MultiOutputClassifier(svm), param_grid=param_grid, cv=10, n_jobs=-1, verbose=2)

# Fit the model
print('Fitting the model...')
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best parameters found: ", grid_search.best_params_)

# Evaluate the model on the test set
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred, target_names=target_columns))

# Calculate precision, recall, and F1 score for each target column
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Optionally, predict labels for the unlabeled data and save the results
# (uncomment and adjust as needed)

# unlabeled_data = pd.read_csv(f"{base_directory}/Thesis/Subreddit Data/{team}/{issue}_data.csv")
# unlabeled_data = merge_parent_post(unlabeled_data)
# unlabeled_features_scaled = scaler.transform(unlabeled_data[feature_columns])
# unlabeled_text = vectorizer.transform(unlabeled_data['merged_text']).toarray()
# unlabeled_combined = np.hstack((unlabeled_features_scaled, unlabeled_text))
# unlabeled_data_predictions = grid_search.best_estimator_.predict(unlabeled_combined)
# for i, column in enumerate(target_columns):
#     unlabeled_data[column] = unlabeled_data_predictions[:, i]
# unlabeled_data.to_csv(f'{base_directory}/Thesis/Subreddit Data/{team}/{issue}_labeled_unlabeled_comments.csv', index=False)
