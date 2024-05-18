import pandas as pd
import os
from sklearn.metrics import adjusted_rand_score
import numpy as np

def calculate_ari_matrix(issue, dir):

    file_paths = [os.path.join(dir, f) for f in os.listdir(dir) if issue in f and f.endswith('.csv')]
    
    labels_data = []
    for i, file in enumerate(file_paths):
        df = pd.read_csv(file).dropna(subset=['Relevant'])
        df.rename(columns={'Relevant': f'Relevant_{i}'}, inplace=True)  
        labels_data.append(df[['Keyword', f'Relevant_{i}']])

    merged_data = labels_data[0]
    for data in labels_data[1:]:
        merged_data = pd.merge(merged_data, data, on='Keyword', how='inner')
    
    relevant_columns = [col for col in merged_data.columns if 'Relevant_' in col]
    aligned_labels = [merged_data[col].values for col in relevant_columns]

    moderators = [os.path.splitext(os.path.basename(file))[0] for file in file_paths]
    ari_matrix = np.zeros((len(aligned_labels), len(aligned_labels)))
    
    for i in range(len(aligned_labels)):
        for j in range(i + 1, len(aligned_labels)):
            if len(aligned_labels[i]) > 0 and len(aligned_labels[j]) > 0:
                ari_matrix[i, j] = adjusted_rand_score(aligned_labels[i], aligned_labels[j])
                ari_matrix[j, i] = ari_matrix[i, j] 

    ari_df = pd.DataFrame(ari_matrix, index=moderators, columns=moderators)
    print(ari_df)

# Update issue and local directory up to /Labelled
calculate_ari_matrix('HealthcareUK', '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Keyword Collection/ARI/Labelled')
