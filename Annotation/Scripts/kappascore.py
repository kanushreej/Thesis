import pandas as pd
from sklearn.metrics import cohen_kappa_score
import numpy as np

base_directory = "/Users/kanushreejaiswal/Desktop" #Change this
issue = 'Brexit' #Change this
moderator_name1 = "Kanushree" #Change this
moderator_name2 = "Adam" #Change this

columns_for_aggregate = ['pro_brexit', 'anti_brexit', 'neutral', 'irrelevant'] #Change this

start_row = 0 # Change this
end_row = 30 #Change this

columns_to_evaluate = [
    'pro_brexit', 'anti_brexit', 'pro_climateAction', 'anti_climateAction', 
    'public_healthcare', 'private_healthcare', 'pro_israel', 'pro_palestine', 
    'increase_tax', 'decrease_tax', 'neutral', 'irrelevant'
]

file1_path = f"{base_directory}/Thesis/Annotation/UK/{moderator_name1}/{issue}_labelled.csv"
file2_path = f"{base_directory}/Thesis/Annotation/UK/{moderator_name2}/{issue}_labelled.csv"

file1_df = pd.read_csv(file1_path)
file2_df = pd.read_csv(file2_path)

file1_df = file1_df.iloc[start_row:end_row]
file2_df = file2_df.iloc[start_row:end_row]

file1_filtered = file1_df[columns_to_evaluate].dropna()
file2_filtered = file2_df[columns_to_evaluate].dropna()

# Ensure the number of rows is the same in both files after dropping NaNs
merged_df = pd.merge(file1_filtered, file2_filtered, left_index=True, right_index=True, suffixes=('_file1', '_file2'))

file1_lists = {}
file2_lists = {}

# Loop through each column to evaluate
for column in columns_to_evaluate:
    file1_lists[column] = merged_df[column + '_file1'].tolist()
    file2_lists[column] = merged_df[column + '_file2'].tolist()

count = 0
kappa_score_total = 0
kappa_scores = {}

for column in columns_to_evaluate:
    kappa_score = cohen_kappa_score(file1_lists[column], file2_lists[column])
    if not np.isnan(kappa_score):
        kappa_scores[column] = kappa_score
        if column in columns_for_aggregate:
            kappa_score_total += kappa_score
            count += 1

# Print individual kappa scores
print()
for column, kappa_score in kappa_scores.items():
    print(f"Kappa Score for {column} is: {kappa_score}")
    print()

# Calculate and print the overall kappa score
if count > 0:
    overall_kappa_score = kappa_score_total / count
    print(f"Overall Kappa Score for relevant columns is: {overall_kappa_score}")
else:
    print("No valid kappa scores calculated.")
print()
