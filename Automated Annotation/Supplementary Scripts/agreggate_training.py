import pandas as pd
import os

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path, dtype={'parent_id': str, 'body': str, 'title': str, 'id': str})

# Function to aggregate labelled data
def aggregate_labelled_data(directory_path):
    combined_data = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith('_labelled.csv'):
            file_path = os.path.join(directory_path, file_name)
            data = load_data(file_path)
            combined_data.append(data)
    return pd.concat(combined_data, ignore_index=True)

# Load and combine labelled data
labelled_data_directory = '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Automated Annotation/Labelled Data/UK'

training_data = aggregate_labelled_data(labelled_data_directory)

output_path = '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Automated Annotation/Labelled Data/UK/all_labelled.csv'
training_data.to_csv(output_path, index=False)