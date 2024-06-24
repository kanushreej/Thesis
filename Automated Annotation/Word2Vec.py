import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import gensim
from gensim.models import Word2Vec
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import re
import time


start_time = time.time()

# Check if 'punkt' is already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load Data
print("Loading data...")
labelled_data = pd.read_csv('ClimateChangeUS_labelled.csv')
unlabelled_data = pd.read_csv('ClimateChangeUS_data.csv')

# Combine title and body
def combine_title_body(row):
    title = row['title'] if pd.notna(row['title']) else ''
    body = row['body'] if pd.notna(row['body']) else ''
    if title and body:
        return title + ' ' + body
    else:
        return title or body

print("Combining title and body...")
labelled_data['text'] = labelled_data.apply(combine_title_body, axis=1)
unlabelled_data['text'] = unlabelled_data.apply(combine_title_body, axis=1)

# Remove rows where both title and body are empty
print("Removing empty rows...")
labelled_data = labelled_data[labelled_data['text'].str.strip().astype(bool)]
unlabelled_data = unlabelled_data[unlabelled_data['text'].str.strip().astype(bool)]

# Context Handling
def build_thread(data, comment_id):
    thread = ""
    try:
        parent_id = data.loc[data['id'] == comment_id, 'parent_id'].values[0]
    except IndexError:
        print(f"No parent_id found for comment_id: {comment_id}")
        return thread

    while parent_id:
        if parent_id.startswith('t1_'):
            if parent_id[3:] not in data['id'].values:
                print(f"No parent_comment found for parent_id: {parent_id}")
                break
            parent_comment = data[data['id'] == parent_id[3:]].iloc[0]
            parent_comment_body = parent_comment['body'] if pd.notna(parent_comment['body']) else ''
            thread = f"{parent_comment_body}\n\n" + thread
            parent_id = parent_comment['parent_id']
        elif parent_id.startswith('t3_'):
            if parent_id[3:] not in data['id'].values:
                print(f"No parent_post found for parent_id: {parent_id}")
                break
            parent_post = data[data['id'] == parent_id[3:]].iloc[0]
            parent_post_title = parent_post['title'] if pd.notna(parent_post['title']) else ''
            parent_post_body = parent_post['body'] if pd.notna(parent_post['body']) else ''
            thread = f"{parent_post_title}\n\n{parent_post_body}\n\n" + thread
            break
    return thread

print("Building context...")
labelled_data['context'] = labelled_data.apply(
    lambda x: build_thread(unlabelled_data, x['id']) + (x['text'] if pd.notna(x['text']) else '') if x['type'] == 'comment' else (x['text'] if pd.notna(x['text']) else ''), axis=1
)
unlabelled_data['context'] = unlabelled_data.apply(
    lambda x: build_thread(unlabelled_data, x['id']) + (x['text'] if pd.notna(x['text']) else '') if x['type'] == 'comment' else (x['text'] if pd.notna(x['text']) else ''), axis=1
)

# Drop unnecessary columns
print("Dropping unnecessary columns...")
labelled_data = labelled_data.drop(columns=['title', 'body', 'id', 'author'])
unlabelled_data = unlabelled_data.drop(columns=['title', 'body', 'id', 'author'])

# Text Preprocessing
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    return tokens

print("Preprocessing text...")
labelled_data['tokens'] = labelled_data['context'].apply(preprocess_text)
unlabelled_data['tokens'] = unlabelled_data['context'].apply(preprocess_text)

# Train Word2Vec model
print("Training Word2Vec model...")
all_tokens = labelled_data['tokens'].tolist() + unlabelled_data['tokens'].tolist()
w2v_model = Word2Vec(sentences=all_tokens, vector_size=100, window=5, min_count=2, workers=4)

def get_mean_vector(tokens, model, vector_size):
    mean_vector = np.zeros(vector_size)
    count = 0
    for token in tokens:
        if token in model.wv:
            mean_vector += model.wv[token]
            count += 1
    if count > 0:
        mean_vector /= count
    return mean_vector

print("Computing mean vectors...")
labelled_data['vector'] = labelled_data['tokens'].apply(lambda x: get_mean_vector(x, w2v_model, 100))
unlabelled_data['vector'] = unlabelled_data['tokens'].apply(lambda x: get_mean_vector(x, w2v_model, 100))

# Prepare data for each stance
all_stances = ['pro_immigration', 'anti_immigration', 'pro_climateAction', 'anti_climateAction',
               'public_healthcare', 'private_healthcare', 'pro_israel', 'pro_palestine',
               'pro_middle_low_tax', 'pro_wealthy_corpo_tax', 'neutral', 'irrelevant']

# Determine which stances have sufficient samples
stances = [stance for stance in all_stances if labelled_data[stance].sum() > 10]  # Threshold can be adjusted

# Perform 10-fold cross-validation for each stance
print("Starting 10-fold cross-validation...")
results = {}

for stance in stances:
    print(f"Processing stance: {stance}")
    labels = labelled_data[stance].values
    vectors = np.stack(labelled_data['vector'].values)
    
    # Check if there are at least two classes in the data
    if len(np.unique(labels)) < 2:
        print(f"Skipping stance {stance} due to insufficient class diversity")
        continue
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for train_index, test_index in kf.split(vectors):
        train_vectors, test_vectors = vectors[train_index], vectors[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        # Check if there are at least two classes in the train and test sets
        if len(np.unique(train_labels)) < 2 or len(np.unique(test_labels)) < 2:
            print(f"Skipping fold due to insufficient class diversity in train or test set for stance {stance}")
            continue

        # You can use Logistic Regression or SVM
        clf = LogisticRegression()
        # clf = SVC(probability=True)

        clf.fit(train_vectors, train_labels)
        predictions = clf.predict(test_vectors)
        probabilities = clf.predict_proba(test_vectors)[:, 1]

        acc = accuracy_score(test_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average='binary', zero_division=0)

        accuracy_scores.append(acc)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    results[stance] = {
        'accuracy': np.mean(accuracy_scores),
        'precision': np.mean(precision_scores),
        'recall': np.mean(recall_scores),
        'f1_score': np.mean(f1_scores)
    }

# Post-processing to handle logical contradictions
print("Resolving contradictions...")
def resolve_contradictions(probabilities):
    resolved_stances = np.zeros_like(probabilities)
    stance_groups = [
        ['pro_immigration', 'anti_immigration'],
        ['pro_climateAction', 'anti_climateAction'],
        ['public_healthcare', 'private_healthcare'],
        ['pro_israel', 'pro_palestine'],
        ['pro_middle_low_tax', 'pro_wealthy_corpo_tax']
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
print("Predicting on unlabelled data...")
def predict_on_unlabelled_data():
    vectors = np.stack(unlabelled_data['vector'].values)
    predictions = []

    for stance in stances:
        clf = LogisticRegression()
        # clf = SVC(probability=True)

        clf.fit(vectors, np.zeros(len(vectors)))  # Dummy fitting
        stance_predictions = clf.predict_proba(vectors)[:, 1]
        predictions.append(stance_predictions)

    predictions = np.array(predictions).transpose(1, 0)
    resolved_predictions = resolve_contradictions(predictions)
    
    return resolved_predictions

# Print the results
print("Printing results...")
for stance, metrics in results.items():
    print(f"Stance: {stance}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1 Score: {metrics['f1_score']}")
    print("\n")

# Predict and resolve contradictions for unlabelled data
resolved_predictions = predict_on_unlabelled_data()
print("Done!")

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Total execution time: {int(elapsed_time // 3600)} hours, {int((elapsed_time % 3600) // 60)} minutes, {elapsed_time % 60:.2f} seconds")
