import pandas as pd
from sklearn.metrics import cohen_kappa_score
import numpy as np

# Load the provided CSV files
file1_path = '/Users/kanushreejaiswal/Desktop/ClimateChangeUS_labelledfile1data.csv'
file2_path = '/Users/kanushreejaiswal/Desktop/ClimateChangeUS_labelledfile2data.csv'

# Read the CSV files
file1_df = pd.read_csv(file1_path)
file2_df = pd.read_csv(file2_path)

# Specify the columns to calculate the kappa score for
columns_to_evaluate = [
    'pro_brexit', 'anti_brexit', 'pro_climateAction', 'anti_climateAction', 
    'public_healthcare', 'private_healthcare', 'pro_israel', 'pro_palestine', 
    'increase_tax', 'decrease_tax', 'neutral', 'irrelevant'
]

# Filter out rows with NaN values in any of the specified columns in both files
file1_filtered = file1_df[columns_to_evaluate].dropna()
file2_filtered = file2_df[columns_to_evaluate].dropna()

# Ensure the number of rows is the same in both files after dropping NaNs
merged_df = pd.merge(file1_filtered, file2_filtered, left_index=True, right_index=True, suffixes=('_file1', '_file2'))

# Initialize dictionaries to hold the lists
file1_lists = {}
file2_lists = {}

# Loop through each column to evaluate
for column in columns_to_evaluate:
    file1_lists[column] = merged_df[column + '_file1'].tolist()
    file2_lists[column] = merged_df[column + '_file2'].tolist()

count = 0
kappa_score_total = 0
for column in columns_to_evaluate:
   kappa_score_total = kappa_score_total + cohen_kappa_score(file1_lists[column], file2_lists[column])
   count = count + 1
   print("Kappa Score for", column, "is:", cohen_kappa_score(file1_lists[column], file2_lists[column]))
   print()

print("Overall Kappa Score is:", kappa_score_total/count)



