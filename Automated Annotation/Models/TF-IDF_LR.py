import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from joblib import Parallel, delayed

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path, dtype={'parent_id': str, 'body': str, 'title': str, 'id': str})

# Combine title and body
def combine_title_body(row):
    title = row['title'] if not pd.isna(row['title']) else ''
    body = row['body'] if not pd.isna(row['body']) else ''
    return f"{title} {body}".strip()

# Function to aggregate labelled data
def aggregate_labelled_data(directory_path):
    combined_data = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith('_labelled.csv'):
            file_path = os.path.join(directory_path, file_name)
            data = load_data(file_path)
            data['text'] = data.apply(combine_title_body, axis=1)
            combined_data.append(data)
    return pd.concat(combined_data, ignore_index=True)

# Load and combine labelled data
labelled_data_directory = '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Automated Annotation/Labelled Data/UK'
training_data = aggregate_labelled_data(labelled_data_directory)

# Load test and context data
test_data = load_data('/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Subreddit Data/UK/Test Data/Brexit_test.csv')
context_data = load_data('/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Subreddit Data/UK/Context Handling/Brexit_context.csv')

# Process test data
test_data['text'] = test_data.apply(combine_title_body, axis=1)
test_data = test_data[test_data['text'].str.strip().astype(bool)]

# Context Handling
def build_thread(data, comment_id):
    thread = []
    try:
        parent_id = data.loc[data['id'] == comment_id, 'parent_id'].values[0]
    except IndexError:
        return " ".join(thread)

    while parent_id:
        if parent_id.startswith('t1_'):
            parent_comment = data[data['id'] == parent_id[3:]]
            if not parent_comment.empty:
                parent_comment = parent_comment.iloc[0]
                thread.insert(0, parent_comment['body'])
                parent_id = parent_comment['parent_id']
            else:
                break
        elif parent_id.startswith('t3_'):
            parent_post = data[data['id'] == parent_id[3:]]
            if not parent_post.empty:
                parent_post = parent_post.iloc[0]
                thread.insert(0, f"{parent_post['title']} {parent_post['body']}")
            break
        else:
            break
    return " ".join(thread)

training_data['context'] = training_data.apply(lambda x: build_thread(context_data, x['id']) + " " + x['text'] if x['type'] == 'comment' else x['text'], axis=1)
test_data['context'] = test_data.apply(lambda x: build_thread(context_data, x['id']) + " " + x['text'] if x['type'] == 'comment' else x['text'], axis=1)

# Drop unnecessary columns
training_data = training_data.drop(columns=['title', 'body', 'id', 'author'])
test_data = test_data.drop(columns=['title', 'body', 'id', 'author'])

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix_training = tfidf_vectorizer.fit_transform(training_data['context'])
tfidf_matrix_test = tfidf_vectorizer.transform(test_data['context'])

# Prepare data for each stance
stances = ['pro_brexit', 'anti_brexit', 'pro_climateAction', 'anti_climateAction',
           'public_healthcare', 'private_healthcare', 'pro_israel', 'pro_palestine',
           'increase_tax', 'decrease_tax', 'neutral', 'irrelevant']

# Perform 10-fold cross-validation for each stance
results = {}

def cross_validate_stance(stance, tfidf_matrix_training, labels):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for train_index, test_index in skf.split(tfidf_matrix_training, labels):
        train_vectors, test_vectors = tfidf_matrix_training[train_index], tfidf_matrix_training[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        clf = LogisticRegression()
        clf.fit(train_vectors, train_labels)
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
        ['public_healthcare', 'private_healthcare'],
        ['pro_israel', 'pro_palestine'],
        ['increase_tax', 'decrease_tax']
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
