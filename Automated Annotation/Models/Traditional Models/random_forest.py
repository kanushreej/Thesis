import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, learning_curve
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

    def evaluate_model(X, y, stances, n_splits=5):
        overall_train_scores = []
        overall_val_scores = []
        overall_train_sizes = None
        results = {stance: {'train_sizes': [], 'train_scores': [], 'val_scores': [], 'metrics': {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}} for stance in stances}

        adjusted_params = {
            'pro_brexit': {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 250},
            'anti_brexit': {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 200},
            'neutral': {'max_depth': 15, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 250},
            'irrelevant': {'max_depth': 15, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 250}
        }


        overall_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}

        for stance in stances:
            clf = RandomForestClassifier(
                max_depth=adjusted_params[stance]['max_depth'],
                max_features=adjusted_params[stance]['max_features'],
                min_samples_leaf=adjusted_params[stance]['min_samples_leaf'],
                min_samples_split=adjusted_params[stance]['min_samples_split'],
                n_estimators=adjusted_params[stance]['n_estimators'],
                random_state=42
            )

            train_sizes, train_scores, val_scores = learning_curve(
                clf, X, y[stance], cv=StratifiedKFold(n_splits=n_splits), 
                scoring='recall', train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
            )

            if overall_train_sizes is None:
                overall_train_sizes = train_sizes

            overall_train_scores.append(np.mean(train_scores, axis=1))
            overall_val_scores.append(np.mean(val_scores, axis=1))

            results[stance]['train_sizes'] = train_sizes
            results[stance]['train_scores'] = np.mean(train_scores, axis=1)
            results[stance]['val_scores'] = np.mean(val_scores, axis=1)

            for train_index, test_index in StratifiedKFold(n_splits=n_splits).split(X, y[stance]):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                clf.fit(X_train, y_train[stance])
                y_pred = clf.predict(X_test)

                accuracy = accuracy_score(y_test[stance], y_pred)
                precision = precision_score(y_test[stance], y_pred, zero_division=0)
                recall = recall_score(y_test[stance], y_pred, zero_division=0)
                f1 = f1_score(y_test[stance], y_pred, zero_division=0)

                results[stance]['metrics']['accuracy'].append(accuracy)
                results[stance]['metrics']['precision'].append(precision)
                results[stance]['metrics']['recall'].append(recall)
                results[stance]['metrics']['f1_score'].append(f1)

                overall_metrics['accuracy'].append(accuracy)
                overall_metrics['precision'].append(precision)
                overall_metrics['recall'].append(recall)
                overall_metrics['f1_score'].append(f1)

        # Average scores across all stances
        avg_train_scores = np.mean(overall_train_scores, axis=0)
        avg_val_scores = np.mean(overall_val_scores, axis=0)

        plt.figure()
        plt.plot(overall_train_sizes, avg_train_scores, label='Training score')
        plt.plot(overall_train_sizes, avg_val_scores, label='Cross-validation score')
        plt.xlabel('Training Examples')
        plt.ylabel('Recall')
        plt.ylim(0.0, 1.0)
        plt.title('Average Learning Curve for All Stances')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

        print("\nOverall Metrics:")
        print(f"Accuracy: {np.mean(overall_metrics['accuracy']):.4f}")
        print(f"Precision: {np.mean(overall_metrics['precision']):.4f}")
        print(f"Recall: {np.mean(overall_metrics['recall']):.4f}")
        print(f"F1 Score: {np.mean(overall_metrics['f1_score']):.4f}")

    evaluate_model(X, y, targets)

classify_issue('brexit')
