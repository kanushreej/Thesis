import pandas as pd
import os
from sklearn.metrics import cohen_kappa_score

# Paths to the labeled data files
moderator1_file = 'Annotation/UK/Adam/IsraelPalestine_labelled.csv' # Change as needed
moderator2_file = 'Annotation/UK/AnotherModerator/IsraelPalestine_labelled.csv' # Change as needed

# Columns containing the labels
label_columns = [
    'pro_brexit', 'anti_brexit',
    'pro_climateAction', 'anti_climateAction',
    'public_healthcare', 'private_healthcare',
    'pro_israel', 'pro_palestine',    
    'increase_tax', 'decrease_tax',  
    'neutral', 'irrelevant'  
]

df_moderator1 = pd.read_csv(moderator1_file)
df_moderator2 = pd.read_csv(moderator2_file)

df_moderator1.sort_values(by=['id'], inplace=True)
df_moderator2.sort_values(by=['id'], inplace=True)

df_moderator1.reset_index(drop=True, inplace=True)
df_moderator2.reset_index(drop=True, inplace=True)

kappa_scores = {}
for column in label_columns:
    if column in df_moderator1.columns and column in df_moderator2.columns:
        kappa = cohen_kappa_score(df_moderator1[column], df_moderator2[column])
        kappa_scores[column] = kappa
    else:
        kappa_scores[column] = 'N/A'

for column, kappa in kappa_scores.items():
    print(f"Cohen's kappa for {column}: {kappa}")

valid_kappa_scores = [k for k in kappa_scores.values() if k != 'N/A']
average_kappa = sum(valid_kappa_scores) / len(valid_kappa_scores) if valid_kappa_scores else 'N/A'
print(f"Average Cohen's kappa score: {average_kappa}")
