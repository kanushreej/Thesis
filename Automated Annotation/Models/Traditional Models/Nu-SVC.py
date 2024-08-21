import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.svm import NuSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def classify_issue(issue):
    stance_groups = {
        'ImmigrationUS': ['pro_immigration','anti_immigration'],
        'ClimateChangeUS': ['pro_climateAction', 'anti_climateAction'],
        'HealthcareUS': ['public_healthcare','private_healthcare'],
        'IsraelPalestineUS': ['pro_israel', 'pro_palestine'],
        'TaxationUS': ['pro_company_taxation', 'pro_worker_taxation']
    }
    
    if issue not in stance_groups:
        raise ValueError(f"Unknown issue: {issue}")
    
    targets = stance_groups[issue] + ['neutral', 'irrelevant']

    file_path = 'C:/Users/rapha/Documents/CS_VU/Thesis/Thesis/Automated Annotation/Training Data/US/{}_training.csv'.format(issue)
    df = pd.read_csv(file_path)

    ## SMOTE ##

    def str_to_array(s):
        return np.fromstring(s.strip("[]"), sep=' ')

    features = df['text_vector'].apply(str_to_array).tolist()
    context = df['context_vector'].apply(str_to_array).tolist()
    X = np.array([np.concatenate((f, c)) for f, c in zip(features, context)])
    y_combined = np.array(df[targets])

    def combine_targets(row):
        return np.argmax(row[targets])
    
    y_combined = df.apply(combine_targets, axis=1)


    smote = SMOTE(random_state=42)
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
        resampled_data[target] = (y_resampled_combined ==i).astype(int)

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

    def resolve_contradictions(probabilities, stances):
        resolved_stances = np.zeros_like(probabilities)
        stance_groups = {
            'ImmigrationUS': ['pro_brexit', 'anti_brexit'],
        'ClimateChangeUS': ['pro_climateAction', 'anti_climateAction'],
        'HealthcareUS': ['pro_NHS', 'anti_NHS'],
        'IsraelPalestineUS': ['pro_israel', 'pro_palestine'],
        'TaxationUS': ['pro_company_taxation', 'pro_worker_taxation']
    }

        for i, prob in enumerate(probabilities):
            prob_dict = {stance: prob[j] for j, stance in enumerate(stances)}

            max_stance = max(prob_dict, key=prob_dict.get)
            if max_stance in ['irrelevant', 'neutral']:
                resolved_stances[i][stances.index(max_stance)] = 1
            else:
                any_above_threshold = any(p > 0.5 for p in prob_dict.values())
                if any_above_threshold:
                    for group in stance_groups[issue]:
                        relevant_group = [s for s in group if s in stances]
                        if relevant_group:
                            max_stance = max(relevant_group, key=lambda x: prob_dict[x])
                            if prob_dict[max_stance] > 0.5:  # Threshold can still be tuned
                                resolved_stances[i][stances.index(max_stance)] = 1
                else:
                    max_stance = max(prob_dict, key=prob_dict.get)
                    resolved_stances[i][stances.index(max_stance)] = 1

        return resolved_stances

    def plot_learning_curve(estimator, title, X, y, cv, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5)):
        plt.figure()
        plt.title(title)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='f1')
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        plt.grid()
        
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        
        plt.legend(loc="best")
        return plt

    def evaluate_model(X, y, stances, n_splits=10):
        results = {stance: {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'predictions': [], 'test_labels': []} for stance in stances}
        overall_predictions = []
        overall_true_labels = []

        skf = StratifiedKFold(n_splits=n_splits)

        for train_index, test_index in skf.split(X, y['neutral']):  # Using 'neutral' just for stratification
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            clf = MultiOutputClassifier(NuSVC())
            param_grid = {
                'estimator__nu': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                'estimator__kernel': ['linear', 'rbf', 'poly'],
                'estimator__gamma': ['scale', 'auto']
            }
            grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, n_jobs=-1, verbose=2)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            y_pred_prob = best_model.predict(X_test)

            for i, stance in enumerate(stances):
                y_pred = (y_pred_prob[:, i] > 0.5).astype(int)
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

        for stance in stances:
            plot_learning_curve(NuSVC(), f"Learning Curves ({stance})", X, y[stance], cv=StratifiedKFold(n_splits=10))
            plt.show()

    def predict_and_resolve(X, stances):
        clf = MultiOutputClassifier(NuSVC())
        param_grid = {
            'estimator__nu': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
            'estimator__kernel': ['linear', 'rbf', 'poly'],
            'estimator__gamma': ['scale', 'auto']
        }
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, n_jobs=-1, verbose=2)
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_

        predictions = best_model.predict(X)
        resolved_predictions = resolve_contradictions(predictions, stances)
        
        return resolved_predictions

    evaluate_model(X, y, targets)

    # resolved_predictions = predict_and_resolve(X, targets)
    # print("Resolved Predictions:", resolved_predictions)

classify_issue('ImmigrationUS')
