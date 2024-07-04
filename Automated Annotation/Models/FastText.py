import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from joblib import Parallel, delayed
import fasttext
import fasttext.util
import os

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to save data for FastText
def save_fasttext_format(data, file_path, stance):
    with open(file_path, 'w') as f:
        for text, label in zip(data['context'], data[stance]):
            text = text.replace('\n', ' ')  # Remove newline characters
            label = float(label)  # Ensure label is float
            f.write(f"__label__{label} {text}\n")

# Load preprocessed data
file_path = '/Users/kanushreejaiswal/Desktop/Thesis/Automated Annotation/Balanced Dataset/UK/ROS/allUK_withcontext_ROSbalanced.csv'
all_data = load_data(file_path)

# Updated stances and stance groups
stances = ['pro_brexit', 'anti_brexit', 'pro_climateAction', 'anti_climateAction',
           'pro_NHS', 'anti_NHS', 'pro_israel', 'pro_palestine',
           'pro_company_taxation', 'pro_worker_taxation', 'neutral', 'irrelevant']

# Split data into training and testing sets (80:20 ratio)
train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)

# Define stance groups for logical contradictions resolution
stance_groups = [
    ['pro_brexit', 'anti_brexit'],
    ['pro_climateAction', 'anti_climateAction'],
    ['pro_NHS', 'anti_NHS'],
    ['pro_israel', 'pro_palestine'],
    ['pro_company_taxation', 'pro_worker_taxation']
]

# Perform cross-validation for each stance
def cross_validate_stance(stance, train_data, n_splits=7):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_index, test_index in skf.split(train_data, train_data[stance]):
        train_fold = train_data.iloc[train_index]
        test_fold = train_data.iloc[test_index]

        train_file = f'train_{stance}.txt'
        test_file = f'test_{stance}.txt'
        
        save_fasttext_format(train_fold, train_file, stance)
        save_fasttext_format(test_fold, test_file, stance)
        
        model = fasttext.train_supervised(train_file)
        
        test_labels = [f"__label__{float(label)}" for label in test_fold[stance].astype(str).tolist()]
        predictions = [model.predict(text.replace('\n', ' '))[0][0] for text in test_fold['context']]
        
        # Convert predictions and labels back to floats
        test_labels = np.array([float(label.replace("__label__", "")) for label in test_labels])
        predictions = np.array([float(pred.replace("__label__", "")) for pred in predictions])

        acc = accuracy_score(test_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average='binary', zero_division=0)

        accuracy_scores.append(acc)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        
        # Remove temporary files
        os.remove(train_file)
        os.remove(test_file)

    return {
        'accuracy': np.mean(accuracy_scores),
        'precision': np.mean(precision_scores),
        'recall': np.mean(recall_scores),
        'f1_score': np.mean(f1_scores)
    }

# Run cross-validation in parallel and collect results
results_list = Parallel(n_jobs=2)(  # Using 2 cores for parallel processing to manage memory usage
    delayed(cross_validate_stance)(stance, train_data) 
    for stance in stances
)

# Ensure results are correctly mapped to stances
results = {stance: metrics for stance, metrics in zip(stances, results_list)}

# Print the results
for stance, metrics in results.items():
    print(f"Stance: {stance}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1 Score: {metrics['f1_score']}")
    print("\n")

# Post-processing to handle logical contradictions
def resolve_contradictions(probabilities):
    resolved_stances = np.zeros_like(probabilities)

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
    actuals = []

    for stance in stances:
        train_file = f'train_{stance}.txt'
        save_fasttext_format(train_data, train_file, stance)
        
        model = fasttext.train_supervised(train_file)
        
        test_labels = test_data[stance].values.astype(float)
        stance_predictions = [model.predict(text.replace('\n', ' '))[0][0] for text in test_data['context']]
        
        # Convert predictions back to floats
        stance_predictions = np.array([float(pred.replace("__label__", "")) for pred in stance_predictions])
        
        predictions.append(stance_predictions)
        actuals.append(test_labels)
        
        # Remove temporary file
        os.remove(train_file)

    predictions = np.array(predictions).transpose(1, 0)
    actuals = np.array(actuals).transpose(1, 0)
    resolved_predictions = resolve_contradictions(predictions)

    return resolved_predictions, actuals

# Predict and resolve contradictions for test data
resolved_predictions, actuals = predict_on_test_data()

# Print counts of predicted vs actual values for each stance
for i, stance in enumerate(stances):
    predicted_counts = np.bincount(resolved_predictions[:, i].astype(int))
    actual_counts = np.bincount(actuals[:, i].astype(int))
    print(f"Stance: {stance}")
    print(f"Predicted counts: {predicted_counts}")
    print(f"Actual counts: {actual_counts}")
    print("\n")
