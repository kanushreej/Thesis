import pandas as pd
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import fasttext
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from tempfile import NamedTemporaryFile

# Define stances early
stances = ['pro_brexit', 'anti_brexit', 'pro_climateAction', 'anti_climateAction',
           'pro_NHS', 'anti_NHS', 'pro_israel', 'pro_palestine',
           'pro_company_taxation', 'pro_worker_taxation', 'neutral', 'irrelevant']

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
    while parent_id:
        if pd.isna(parent_id):
            break
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

# Text Preprocessing
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    return ' '.join(tokens)

labelled_data['clean_text'] = labelled_data['context'].apply(preprocess_text)
unlabelled_data['clean_text'] = unlabelled_data['context'].apply(preprocess_text)

# Save labelled data to a temporary file for FastText
with NamedTemporaryFile(mode='w', delete=False) as f:
    temp_file = f.name
    for text, labels in zip(labelled_data['clean_text'], labelled_data[stances].values):
        label_str = ' '.join(f"__label__{stance}" for stance, label in zip(stances, labels) if label == 1)
        f.write(f"{label_str} {text}\n")

# Train FastText model
fasttext_model = fasttext.train_unsupervised(temp_file, model='skipgram', dim=100)

# Get vector representation for texts
def get_fasttext_vector(text, model):
    return model.get_sentence_vector(text)

labelled_data['vector'] = labelled_data['clean_text'].apply(lambda x: get_fasttext_vector(x, fasttext_model))
unlabelled_data['vector'] = unlabelled_data['clean_text'].apply(lambda x: get_fasttext_vector(x, fasttext_model))

# Convert stances to a multi-label target array
labels = labelled_data[stances].values
vectors = np.stack(labelled_data['vector'].values)

kf = KFold(n_splits=10, shuffle=True, random_state=42)

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for train_index, test_index in kf.split(vectors):
    train_vectors, test_vectors = vectors[train_index], vectors[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]

    clf = MultiOutputClassifier(LogisticRegression())
    clf.fit(train_vectors, train_labels)
    predictions = clf.predict(test_vectors)

    acc = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average='micro')

    accuracy_scores.append(acc)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

results = {
    'accuracy': np.mean(accuracy_scores),
    'precision': np.mean(precision_scores),
    'recall': np.mean(recall_scores),
    'f1_score': np.mean(f1_scores)
}

# Print the results
print(f"Accuracy: {results['accuracy']}")
print(f"Precision: {results['precision']}")
print(f"Recall: {results['recall']}")
print(f"F1 Score: {results['f1_score']}")
print("\n")

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
def predict_on_unlabelled_data():
    vectors = np.stack(unlabelled_data['vector'].values)
    predictions = []

    clf = MultiOutputClassifier(LogisticRegression())
    clf.fit(vectors, np.zeros((len(vectors), len(stances))))  # Dummy fitting
    stance_predictions = clf.predict_proba(vectors)
    predictions.append(stance_predictions)

    predictions = np.array(predictions).transpose(1, 0)
    resolved_predictions = resolve_contradictions(predictions)
    
    return resolved_predictions

# Predict and resolve contradictions for unlabelled data
resolved_predictions = predict_on_unlabelled_data()
