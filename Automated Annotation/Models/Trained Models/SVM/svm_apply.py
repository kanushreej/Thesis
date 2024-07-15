import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

TEXT_VECTOR_SIZE = 150
CONTEXT_VECTOR_SIZE = 250

issue = 'IsraelPalestineUS'
data_path = '/path/to/new/data.csv'  # Update with the actual path to data
output_path = '/path/to/output/directory'  # Update with the actual path to Analyses/Labelled Data

def load_model(issue):
    model_filename = f'{issue}_svm_model.pkl'
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    return model

def predict_data(issue, data_path, output_path):
    model = load_model(issue)
    
    # Load the new data
    data = pd.read_csv(data_path)

    # Preprocess new data (similar to training data preprocessing)
    def str_to_array(s):
        return np.fromstring(s.strip("[]"), sep=' ')

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

    data['features'] = data.apply(extract_vectors, axis=1)
    X_new = np.stack(data['features'].values)

    # Scale the features
    scaler = StandardScaler()
    X_new = scaler.fit_transform(X_new)

    # Apply PCA
    pca = PCA(n_components=5)
    X_new_pca = pca.fit_transform(X_new)

    # Predict stances
    y_pred_prob = model.predict_proba(X_new_pca)
    y_pred = np.zeros_like(y_pred_prob)
    max_prob_indices = np.argmax(y_pred_prob, axis=1)
    for i, idx in enumerate(max_prob_indices):
        y_pred[i, idx] = 1

    # Define the targets based on the issue
    stance_groups = {
        'brexit': ['pro_brexit', 'anti_brexit'],
        'ClimateChangeUK': ['pro_climateAction', 'anti_climateAction'],
        'HealthcareUK': ['pro_NHS', 'anti_NHS'],
        'IsraelPalestineUK': ['pro_israel', 'pro_palestine'],
        'TaxationUK': ['pro_company_taxation', 'pro_worker_taxation'],

        'ImmigrationUS': ['pro_immigration', 'anti_immigration'],
        'ClimateChangeUS': ['pro_climateAction', 'anti_climateAction'],
        'HealthcareUS': ['public_healthcare', 'private_healthcare'],
        'IsraelPalestineUS': ['pro_israel', 'pro_palestine'],
        'TaxationUS': ['pro_middle_low_tax', 'pro_wealthy_corpo_tax']
    }

    if issue not in stance_groups:
        raise ValueError(f"Unknown issue: {issue}")

    targets = stance_groups[issue] + ['neutral', 'irrelevant']

    # Add predictions to new data
    for i, target in enumerate(targets):
        data[target] = y_pred[:, i]

    # Save the labelled new data
    output_filename = f'{issue}_labelled.csv'
    data.to_csv(f'{output_path}/{output_filename}', index=False)

    return data

labelled_data = predict_data(issue, data_path, output_path)
print("Labelled data saved to:", output_path)
