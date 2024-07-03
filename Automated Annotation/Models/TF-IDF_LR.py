import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
import numpy as np
from joblib import Parallel, delayed
import sys

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Load preprocessed data
training_data = load_data('/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Automated Annotation/Labelled Data/UK/all_labelled_with_context.csv')

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix_training = tfidf_vectorizer.fit_transform(training_data['context'])

# Updated stances and stance groups
stances = ['pro_brexit', 'anti_brexit', 'pro_climateAction', 'anti_climateAction',
           'pro_NHS', 'anti_NHS', 'pro_israel', 'pro_palestine',
           'pro_company_taxation', 'pro_worker_taxation', 'neutral', 'irrelevant']

stance_groups = [
    ['pro_brexit', 'anti_brexit'],
    ['pro_climateAction', 'anti_climateAction'],
    ['pro_NHS', 'anti_NHS'],
    ['pro_israel', 'pro_palestine'],
    ['pro_company_taxation', 'pro_worker_taxation']
]

# Perform cross-validation for each stance
def cross_validate_stance(stance, tfidf_matrix_training, labels):
    # Determine the number of splits for cross-validation
    min_class_count = np.min(np.bincount(labels.astype(int)))
    n_splits = min(10, min_class_count)  # Adjust n_splits based on the smallest class size
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    all_test_labels = np.array([], dtype=int)
    all_predictions = np.array([], dtype=int)

    for train_index, test_index in skf.split(tfidf_matrix_training, labels):
        train_vectors, test_vectors = tfidf_matrix_training[train_index], tfidf_matrix_training[test_index]
        train_labels, test_labels = labels[train_index].astype(int), labels[test_index].astype(int)  # Ensure labels are integers

        # Adjust k_neighbors for SMOTE based on the number of samples in the minority class
        minority_class_count = min(np.bincount(train_labels))
        k_neighbors = min(5, minority_class_count - 1) if minority_class_count > 1 else 1
        if minority_class_count > 1:
            smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
            train_vectors, train_labels = smote.fit_resample(train_vectors, train_labels)

        clf = LogisticRegression(class_weight='balanced')
        clf.fit(train_vectors, train_labels)
        predictions = clf.predict(test_vectors)

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
results_list = Parallel(n_jobs=-1)(delayed(cross_validate_stance)(stance, tfidf_matrix_training, training_data[stance].values) for stance in stances)

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
