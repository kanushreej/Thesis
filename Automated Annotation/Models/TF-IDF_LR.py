import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Load preprocessed data
training_data = load_data('/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Automated Annotation/Balanced Dataset/UK/ROS/allUK_withcontext_ROSbalanced.csv')

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=3000)
tfidf_matrix_training = tfidf_vectorizer.fit_transform(training_data['context'])

# Scale the data
scaler = StandardScaler(with_mean=False)
tfidf_matrix_training = scaler.fit_transform(tfidf_matrix_training)

# Updated stances and stance groups
stances = ['pro_brexit', 'anti_brexit', 'pro_climateAction', 'anti_climateAction',
           'pro_NHS', 'anti_NHS', 'pro_israel', 'pro_palestine',
           'pro_company_taxation', 'pro_worker_taxation', 'neutral', 'irrelevant']

# Perform cross-validation for each stance
def cross_validate_stance(stance, tfidf_matrix_training, labels):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    all_test_labels = np.array([], dtype=int)
    all_predictions = np.array([], dtype=int)

    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    clf = GridSearchCV(LogisticRegression(class_weight='balanced', max_iter=500), param_grid, cv=5, scoring='f1')
    
    for train_index, test_index in skf.split(tfidf_matrix_training, labels):
        train_vectors, test_vectors = tfidf_matrix_training[train_index], tfidf_matrix_training[test_index]
        train_labels, test_labels = labels[train_index].astype(int), labels[test_index].astype(int)  # Ensure labels are integers

        clf.fit(train_vectors, train_labels)
        best_clf = clf.best_estimator_
        predictions = best_clf.predict(test_vectors)

        acc = accuracy_score(test_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average='binary', zero_division=0)

        accuracy_scores.append(acc)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        
        all_test_labels = np.concatenate((all_test_labels, test_labels))
        all_predictions = np.concatenate((all_predictions, predictions))

    return {
        'accuracy': np.mean(accuracy_scores),
        'precision': np.mean(precision_scores),
        'recall': np.mean(recall_scores),
        'f1_score': np.mean(f1_scores),
        'test_labels': all_test_labels,
        'predictions': all_predictions
    }

# Run cross-validation in parallel and collect results
results_list = [cross_validate_stance(stance, tfidf_matrix_training, training_data[stance].values) for stance in stances]

# Ensure results are correctly mapped to stances
results = {stance: metrics for stance, metrics in zip(stances, results_list)}

# Print the results and counts
for stance, metrics in results.items():
    print(f"Stance: {stance}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1 Score: {metrics['f1_score']}")

    # Print counts of predicted vs actual values
    predicted_counts = np.bincount(metrics['predictions'])
    actual_counts = np.bincount(metrics['test_labels'])
    print(f"Predicted counts: {predicted_counts}")
    print(f"Actual counts: {actual_counts}")
    print("\n")

# Terminate the script
sys.exit(0)
