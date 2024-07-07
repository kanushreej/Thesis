import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import os

def classify_issue(issue):
    stance_groups = {
        'brexit': ['pro_brexit', 'anti_brexit'],
        'climateAction': ['pro_climateAction', 'anti_climateAction'],
        'NHS': ['pro_NHS', 'anti_NHS'],
        'israel_palestine': ['pro_israel', 'pro_palestine'],
        'taxation': ['pro_company_taxation', 'pro_worker_taxation']
    }
    
    if issue not in stance_groups:
        raise ValueError(f"Unknown issue: {issue}")
    
    targets = stance_groups[issue] + ['neutral', 'irrelevant']

    file_path = r'C:\Users\vshap\OneDrive\Desktop\work\code\Thesis\Thesis\Automated Annotation\Training Data\UK\{}_training.csv'.format(issue)
    df = pd.read_csv(file_path)

    ## SMOTE ##
    def str_to_array(s):
        return np.fromstring(s.strip("[]"), sep=' ')

    features = df['text_vector'].apply(str_to_array).tolist()
    context = df['context_vector'].apply(str_to_array).tolist()
    X = np.array([np.concatenate((f, c)) for f, c in zip(features, context)])
    y_combined = np.array(df[targets])

    smote = SMOTE()
    X_resampled, y_resampled_combined = smote.fit_resample(X, y_combined)

    text_vector_length = len(str_to_array(df['text_vector'].iloc[0]))
    context_vector_length = len(str_to_array(df['context_vector'].iloc[0]))

    text_vectors_resampled = X_resampled[:, :text_vector_length]
    context_vectors_resampled = X_resampled[:, text_vector_length:]

    def array_to_str(arr):
        return ' '.join(map(str, arr))

    resampled_data = pd.DataFrame()
    resampled_data['text_vector'] = list(map(array_to_str, text_vectors_resampled))
    resampled_data['context_vector'] = list(map(array_to_str, context_vectors_resampled))

    for i, target in enumerate(targets):
        resampled_data[target] = y_resampled_combined[:, i]

    data = resampled_data

    ## FEATURE VECTOR PROCESSING ##
    TEXT_VECTOR_SIZE = 100
    CONTEXT_VECTOR_SIZE = 100

    def extract_vectors(row):
        text_vector = np.array(row['text_vector'].split(), dtype=float)
        context_vector = np.array(row['context_vector'].split(), dtype=float)

        if len(text_vector) > TEXT_VECTOR_SIZE:
            text_vector = text_vector[:TEXT_VECTOR_SIZE]
        else:
            text_vector = np.pad(text_vector, (0, TEXT_VECTOR_SIZE - len(text_vector)), 'constant')

        if len(context_vector) > CONTEXT_VECTOR_SIZE:
            context_vector = context_vector[:CONTEXT_VECTOR_SIZE]
        else:
            context_vector = np.pad(context_vector, (0, CONTEXT_VECTOR_SIZE - len(context_vector)), 'constant')

        return np.concatenate([text_vector, context_vector])
    
    ## MODEL ##
    X = data.apply(extract_vectors, axis=1)
    X = np.stack(X.values)
    y = data[targets]

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    def evaluate_model(X, y, stances, n_splits=10):
        results = {stance: {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'predictions': [], 'test_labels': []} for stance in stances}
        overall_predictions = []
        overall_true_labels = []

        skf = StratifiedKFold(n_splits=n_splits)

        adjusted_params = {
            'pro_brexit': {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 50},
            'anti_brexit': {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50},
            'neutral': {'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200},
            'irrelevant': {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
        }

        for train_index, test_index in skf.split(X, y['anti_brexit']):  # Using 'neutral' just for stratification
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            for stance in stances:
                clf = RandomForestClassifier(
                    max_depth=adjusted_params[stance]['max_depth'],
                    max_features=adjusted_params[stance]['max_features'],
                    min_samples_leaf=adjusted_params[stance]['min_samples_leaf'],
                    min_samples_split=adjusted_params[stance]['min_samples_split'],
                    n_estimators=adjusted_params[stance]['n_estimators'],
                    random_state=42
                )
                clf.fit(X_train, y_train[stance])

                y_pred = clf.predict(X_test)

                results[stance]['accuracy'].append(accuracy_score(y_test[stance], y_pred))
                results[stance]['precision'].append(precision_score(y_test[stance], y_pred, zero_division=0))
                results[stance]['recall'].append(recall_score(y_test[stance], y_pred, zero_division=0))
                results[stance]['f1_score'].append(f1_score(y_test[stance], y_pred, zero_division=0))
                results[stance]['predictions'].extend(y_pred)
                results[stance]['test_labels'].extend(y_test[stance])

                overall_predictions.extend(y_pred)
                overall_true_labels.extend(y_test[stance])

        overall_metrics = {'Stance': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}
        
        for stance, metrics in results.items():
            accuracy = np.mean(metrics['accuracy'])
            precision = np.mean(metrics['precision'])
            recall = np.mean(metrics['recall'])
            f1 = np.mean(metrics['f1_score'])

            overall_metrics['Stance'].append(stance)
            overall_metrics['Accuracy'].append(accuracy)
            overall_metrics['Precision'].append(precision)
            overall_metrics['Recall'].append(recall)
            overall_metrics['F1 Score'].append(f1)
            
            print(f"Stance: {stance}")
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")

            predicted_counts = np.bincount(metrics['predictions'])
            actual_counts = np.bincount(metrics['test_labels'])
            print(f"Predicted counts: {predicted_counts}")
            print(f"Actual counts: {actual_counts}")
            print("\n")

        overall_accuracy = accuracy_score(overall_true_labels, overall_predictions)
        overall_precision = precision_score(overall_true_labels, overall_predictions, zero_division=0)
        overall_recall = recall_score(overall_true_labels, overall_predictions, zero_division=0)
        overall_f1 = f1_score(overall_true_labels, overall_predictions, zero_division=0)

        print(f"Overall Accuracy: {overall_accuracy}")
        print(f"Overall Precision: {overall_precision}")
        print(f"Overall Recall: {overall_recall}")
        print(f"Overall F1 Score: {overall_f1}")

        overall_metrics_df = pd.DataFrame(overall_metrics)
        print("Overall Performance Metrics by Stance:\n", overall_metrics_df)

    evaluate_model(X, y, targets)

classify_issue('brexit')
