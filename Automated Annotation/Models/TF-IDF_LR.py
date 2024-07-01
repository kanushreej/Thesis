import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
import numpy as np
from joblib import Parallel, delayed

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Load preprocessed data
training_data = load_data('/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Automated Annotation/Labelled Data/UK/all_labelled_with_context.csv')
test_data = load_data('/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Collected Data/UK/Subreddit Data/Brexit_data_with_context.csv')

# Remove training data IDs from the test data
test_data = test_data[~test_data['id'].isin(training_data['id'])]

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix_training = tfidf_vectorizer.fit_transform(training_data['context'])
tfidf_matrix_test = tfidf_vectorizer.transform(test_data['context'])

# Prepare data for each stance
stances = ['pro_brexit', 'anti_brexit', 'pro_climateAction', 'anti_climateAction',
           'pro_NHS', 'anti_NHS', 'pro_israel', 'pro_palestine',
           'pro_company_taxation', 'pro_worker_taxation', 'neutral', 'irrelevant']

# Perform 10-fold cross-validation for each stance
results = {}

def cross_validate_stance(stance, tfidf_matrix_training, labels):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    smote = SMOTE(random_state=42)

    for train_index, test_index in skf.split(tfidf_matrix_training, labels):
        train_vectors, test_vectors = tfidf_matrix_training[train_index], tfidf_matrix_training[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        train_vectors_resampled, train_labels_resampled = smote.fit_resample(train_vectors, train_labels)

        clf = LogisticRegression()
        clf.fit(train_vectors_resampled, train_labels_resampled)
        predictions = clf.predict(test_vectors)

        acc = accuracy_score(test_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average='binary', zero_division=0)

        accuracy_scores.append(acc)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    return {
        'accuracy': np.mean(accuracy_scores),
        'precision': np.mean(precision_scores),
        'recall': np.mean(recall_scores),
        'f1_score': np.mean(f1_scores)
    }

results = Parallel(n_jobs=-1)(delayed(cross_validate_stance)(stance, tfidf_matrix_training, training_data[stance].values) for stance in stances)

# Post-processing to handle logical contradictions
def resolve_contradictions(probabilities):
    resolved_stances = np.zeros_like(probabilities)
    stance_groups = [
        ['pro_brexit', 'anti_brexit'],
        ['pro_climateAction', 'anti_climateAction'],
        ['pro_NHS', 'anti_NHS'],
        ['pro_israel', 'pro_palestine'],
        ['pro_company_taxation', 'pro_worker_taxation']
    ]

    for i, prob in enumerate(probabilities):
        prob_dict = {stance: prob[j] for j, stance in enumerate(stances)}

        max_stance = max(prob_dict, key=prob_dict.get)
        if max_stance in ['irrelevant', 'neutral']:
            resolved_stances[i][stances.index(max_stance)] = 1
        else:
            any_above_threshold = any(p > 0.5 for p in prob_dict.values())
            if any_above_threshold:
                for group in stance_groups:
                    max_stance = max(group, key=lambda x: prob_dict[x])
                    if prob_dict[max_stance] > 0.5:
                        resolved_stances[i][stances.index(max_stance)] = 1
            else:
                max_stance = max(prob_dict, key=prob_dict.get)
                resolved_stances[i][stances.index(max_stance)] = 1

    return resolved_stances

# Predicting on test data
def predict_on_test_data():
    predictions = []

    for stance in stances:
        clf = SVC(probability=True)
        clf.fit(tfidf_matrix_training, training_data[stance])
        stance_predictions = clf.predict_proba(tfidf_matrix_test)[:, 1]
        predictions.append(stance_predictions)

    predictions = np.array(predictions).transpose(1, 0)
    resolved_predictions = resolve_contradictions(predictions)

    return resolved_predictions

# Print the results
for stance, metrics in zip(stances, results):
    print(f"Stance: {stance}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1 Score: {metrics['f1_score']}")
    print("\n")

# Predict and resolve contradictions for test data
resolved_predictions = predict_on_test_data()
