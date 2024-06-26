import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from empath import Empath
import numpy as np

# Initialize Empath
lexicon = Empath()

# Load Data
labelled_data = pd.read_csv('/Users/kanushreejaiswal/Desktop/Thesis/Manual Annotation/UK/Final Labelled Data/ClimateChangeUK_labelled.csv')
unlabelled_data = pd.read_csv('/Users/kanushreejaiswal/Desktop/Thesis/Subreddit Data/UK/User Data/ClimateChangeUK_data.csv')

# Combine title and body
def combine_title_body(row):
    if pd.isna(row['title']):
        return row['body']
    elif pd.isna(row['body']):
        return row['title']
    else:
        return row['title'] + ' ' + row['body']

labelled_data['text'] = labelled_data.apply(combine_title_body, axis=1)
unlabelled_data['text'] = unlabelled_data.apply(combine_title_body, axis=1)

# Remove rows where both title and body are empty
labelled_data = labelled_data[labelled_data['text'].str.strip().astype(bool)]
unlabelled_data = unlabelled_data[unlabelled_data['text'].str.strip().astype(bool)]

# Context Handling
def build_thread(data, comment_id):
    thread = ""
    parent_id = data.loc[data['id'] == comment_id, 'parent_id'].values[0]
    while pd.notna(parent_id):
        if isinstance(parent_id, str):
            if parent_id.startswith('t1_'):
                parent_comment = data[data['id'] == parent_id[3:]].iloc[0]
                thread = f"{parent_comment['body']}\n\n" + thread
                parent_id = parent_comment['parent_id']
            elif parent_id.startswith('t3_'):
                parent_post = data[data['id'] == parent_id[3:]].iloc[0]
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

# Extract Empath features
def extract_empath_features(text):
    empath_features = lexicon.analyze(text, categories=None, normalize=True)
    return list(empath_features.values())

labelled_data['features'] = labelled_data['context'].apply(extract_empath_features)
unlabelled_data['features'] = unlabelled_data['context'].apply(extract_empath_features)

# Prepare data for each stance
stances = ['pro_brexit', 'anti_brexit', 'pro_climateAction', 'anti_climateAction',
           'pro_NHS', 'anti_NHS', 'pro_israel', 'pro_palestine',
           'pro_company_taxation', 'pro_worker_taxation', 'neutral', 'irrelevant']

# Perform 10-fold cross-validation for each stance
results = {}

for stance in stances:
    labels = labelled_data[stance].values
    features = np.stack(labelled_data['features'].values)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for train_index, test_index in kf.split(features):
        train_features, test_features = features[train_index], features[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        # Ensure both classes are present in the training data
        if len(np.unique(train_labels)) < 2:
            continue

        # You can use Logistic Regression or SVM
        clf = LogisticRegression()
        # clf = SVC(probability=True)

        clf.fit(train_features, train_labels)
        predictions = clf.predict(test_features)
        probabilities = clf.predict_proba(test_features)[:, 1]

        acc = accuracy_score(test_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average='binary')

        accuracy_scores.append(acc)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    if accuracy_scores:  # Only store results if there are valid splits
        results[stance] = {
            'accuracy': np.mean(accuracy_scores),
            'precision': np.mean(precision_scores),
            'recall': np.mean(recall_scores),
            'f1_score': np.mean(f1_scores)
        }

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
def predict_on_unlabelled_data(stances, labelled_data, unlabelled_data):
    # Train classifiers on labelled data
    trained_classifiers = {}
    for stance in stances:
        labels = labelled_data[stance].values
        features = np.stack(labelled_data['features'].values)
        
        clf = LogisticRegression()
        clf.fit(features, labels)
        trained_classifiers[stance] = clf

    # Predict on unlabelled data
    features = np.stack(unlabelled_data['features'].values)
    predictions = []

    for stance in stances:
        clf = trained_classifiers[stance]
        stance_predictions = clf.predict_proba(features)[:, 1]
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
resolved_predictions = predict_on_unlabelled_data(stances, labelled_data, unlabelled_data)
