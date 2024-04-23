import pandas as pd
from sklearn.metrics import cohen_kappa_score

file1 = 'UKhealthcare.csv'
file2 = 'UKhealthcare_copy.csv'

def calculate_kappa_score(file1, file2):
    # Read the CSV files into DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Ensure both DataFrames have the same structure and order of rows
    
    # Extract true labels from the 'public' column
    true_labels1 = df1['public'].tolist()
    true_labels2 = df2['public'].tolist()
    
    # Extract predicted labels from the 'private' column
    predicted_labels1 = df1['private'].tolist()
    predicted_labels2 = df2['private'].tolist()
    
    # Calculate Cohen's Kappa score
    kappa_score = cohen_kappa_score(true_labels1 + true_labels2, predicted_labels1 + predicted_labels2)
    
    return kappa_score

if __name__ == "__main__":

    # Calculate Cohen's Kappa score
    kappa_score = calculate_kappa_score(file1, file2)
    
    print(f"Cohen's Kappa Score: {kappa_score}")
