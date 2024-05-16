import pandas as pd
from sklearn.metrics import cohen_kappa_score

file1 = 'UKhealthcare_extracopy.csv' #change file names!!!
file2 = 'UKhealthcare_copy.csv' #change file names!!!

def calculate_kappa_score(file1, file2):
    # Read the CSV files into DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    df1.fillna(-1, inplace=True)
    df2.fillna(-1, inplace=True)
    
    # Extract true labels from the columns
    true_labels1 = df1[['public', 'private', 'neutral', 'irrelevant']].values #change column names!!!
    true_labels2 = df2[['public', 'private', 'neutral', 'irrelevant']].values #change column names!!!
    
    # Flatten the arrays for cohen_kappa_score
    true_labels1 = true_labels1.flatten()
    true_labels2 = true_labels2.flatten()
    
    # Calculate Cohen's Kappa score
    kappa_score = cohen_kappa_score(true_labels1, true_labels2)
    
    return kappa_score

if __name__ == "__main__":
    
    # Calculate Cohen's Kappa score
    kappa_score = calculate_kappa_score(file1,file2)
    
    print(f"Cohen's Kappa Score: {kappa_score}")
