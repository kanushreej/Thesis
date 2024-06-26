import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Load Data
labelled_data = pd.read_csv('/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Automated Annotation/Training Data/Brexit_labelled.csv', dtype={'parent_id': str, 'body': str, 'title': str, 'id': str})
unlabelled_data = pd.read_csv('/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Subreddit Data/UK/Brexit_data.csv', dtype={'parent_id': str, 'body': str, 'title': str, 'id': str})

# Combine title and body
def combine_title_body(row):
    title = row['title'] if not pd.isna(row['title']) else ''
    body = row['body'] if not pd.isna(row['body']) else ''
    return f"{title} {body}".strip()

labelled_data['text'] = labelled_data.apply(combine_title_body, axis=1)
unlabelled_data['text'] = unlabelled_data.apply(combine_title_body, axis=1)

# Remove rows where both title and body are empty
labelled_data = labelled_data[labelled_data['text'].str.strip().astype(bool)]
unlabelled_data = unlabelled_data[unlabelled_data['text'].str.strip().astype(bool)]

# Context Handling
def build_thread(data, comment_id):
    thread = ""
    try:
        parent_id = data.loc[data['id'] == comment_id, 'parent_id'].values[0]
    except IndexError:
        return thread

    while parent_id:
        if parent_id.startswith('t1_'):
            parent_comment = data[data['id'] == parent_id[3:]]
            if not parent_comment.empty:
                parent_comment = parent_comment.iloc[0]
                thread = f"{parent_comment['body']}\n\n" + thread
                parent_id = parent_comment['parent_id']
            else:
                break
        elif parent_id.startswith('t3_'):
            parent_post = data[data['id'] == parent_id[3:]]
            if not parent_post.empty:
                parent_post = parent_post.iloc[0]
                thread = f"{parent_post['title']}\n\n{parent_post['body']}\n\n" + thread
            break
        else:
            break
    return thread

# Apply context handling
labelled_data['context'] = labelled_data.apply(lambda x: build_thread(unlabelled_data, x['id']) + x['text'] if x['type'] == 'comment' else x['text'], axis=1)
unlabelled_data['context'] = unlabelled_data.apply(lambda x: build_thread(unlabelled_data, x['id']) + x['text'] if x['type'] == 'comment' else x['text'], axis=1)

# Drop unnecessary columns
labelled_data = labelled_data.drop(columns=['title', 'body', 'id', 'author'])
unlabelled_data = unlabelled_data.drop(columns=['title', 'body', 'id', 'author'])

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix_labelled = tfidf_vectorizer.fit_transform(labelled_data['context'])
tfidf_matrix_unlabelled = tfidf_vectorizer.transform(unlabelled_data['context'])

# Prepare data for each stance
stances = ['pro_brexit', 'anti_brexit', 'pro_climateAction', 'anti_climateAction',
           'public_healthcare', 'private_healthcare', 'pro_israel', 'pro_palestine',
           'increase_tax', 'decrease_tax', 'neutral', 'irrelevant']

# Perform 10-fold cross-validation for all stances simultaneously
results = {stance: {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []} for stance in stances}

kf = KFold(n_splits=10, shuffle=True, random_state=42)
for train_index, test_index in kf.split(tfidf_matrix_labelled):
    train_vectors, test_vectors = tfidf_matrix_labelled[train_index], tfidf_matrix_labelled[test_index]
    
    for stance in stances:
        train_labels = labelled_data[stance].values[train_index]
        test_labels = labelled_data[stance].values[test_index]
        
        clf = LogisticRegression()
        # clf = SVC(probability=True)
        
        clf.fit(train_vectors, train_labels)
        predictions = clf.predict(test_vectors)
        
        acc = accuracy_score(test_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average='binary')
        
        results[stance]['accuracy'].append(acc)
        results[stance]['precision'].append(precision)
        results[stance]['recall'].append(recall)
        results[stance]['f1_score'].append(f1)

# Average the results
for stance in stances:
    results[stance] = {
        'accuracy': np.mean(results[stance]['accuracy']),
        'precision': np.mean(results[stance]['precision']),
        'recall': np.mean(results[stance]['recall']),
        'f1_score': np.mean(results[stance]['f1_score'])
    }

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
        
        # Check for irrelevant or neutral
        max_stance = max(prob_dict, key=prob_dict.get)
        if max_stance in ['irrelevant', 'neutral']:
            resolved_stances[i][stances.index(max_stance)] = 1
        else:
            any_above_threshold = any(p > 0.5 for p in prob_dict.values())
            if any_above_threshold:
                for group in stance_groups:
                    max_stance = max(group, key=lambda x: prob_dict[x])
                    if prob_dict[max_stance] > 0.5:  # Threshold can still be tuned
                        resolved_stances[i][stances.index(max_stance)] = 1
            else:
                max_stance = max(prob_dict, key=prob_dict.get)
                resolved_stances[i][stances.index(max_stance)] = 1

    return resolved_stances

# Predicting on unlabelled data
def predict_on_unlabelled_data():
    predictions = []

    for stance in stances:
        clf = LogisticRegression()
        # clf = SVC(probability=True)

        clf.fit(tfidf_matrix_labelled, labelled_data[stance])  # Use the labelled data for fitting
        stance_predictions = clf.predict_proba(tfidf_matrix_unlabelled)[:, 1]
        predictions.append(stance_predictions)

    predictions = np.array(predictions).transpose(1, 0)
    resolved_predictions = resolve_contradictions(predictions)
    
    return resolved_predictions

# Print the results
for stance, metrics in results.items():
    print(f"Stance: {stance}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1 Score: {metrics['f1_score']}")
    print("\n")

# Predict and resolve contradictions for unlabelled data
resolved_predictions = predict_on_unlabelled_data()
