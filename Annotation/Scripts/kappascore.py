import pandas as pd
import os
from sklearn.metrics import cohen_kappa_score

moderator1_file = '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Annotation/UK/Kanushree/BrexitLabelled.csv' # Change as needed
moderator2_file = '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Annotation/UK/Adam/Brexit_labelled_Adam.csv' # Change as needed

# Columns containing the labels
label_columns = [
    'pro_brexit','anti_brexit',
    'pro_climateAction','anti_climateAction',
    'public_healthcare', 'private_healthcare',
    'pro_israel', 'pro_palestine',    
    'increase_tax', 'decrease_tax',  
    'neutral', 'irrelevant'  
]


df_moderator1 = pd.read_csv(moderator1_file, dtype={'id': str})
df_moderator2 = pd.read_csv(moderator2_file, dtype={'id': str})

df_moderator1.sort_values(by=['id'], inplace=True)
df_moderator2.sort_values(by=['id'], inplace=True)

df_moderator1.reset_index(drop=True, inplace=True)
df_moderator2.reset_index(drop=True, inplace=True)

merged_df = pd.merge(df_moderator1, df_moderator2, on='id', suffixes=('_mod1', '_mod2'))

for column in label_columns:
    merged_df = merged_df.dropna(subset=[f"{column}_mod1", f"{column}_mod2"])


kappa_scores = {}
for column in label_columns:
    if f"{column}_mod1" in merged_df.columns and f"{column}_mod2" in merged_df.columns:
        kappa = cohen_kappa_score(merged_df[f"{column}_mod1"], merged_df[f"{column}_mod2"])
        kappa_scores[column] = kappa
    else:
        kappa_scores[column] = 'N/A'

for column, kappa in kappa_scores.items():
    print(f"Cohen's kappa for {column}: {kappa}")

valid_kappa_scores = [k for k in kappa_scores.values() if k != 'N/A']
average_kappa = sum(valid_kappa_scores) / len(valid_kappa_scores) if valid_kappa_scores else 'N/A'
print(f"Average Cohen's kappa score: {average_kappa}")
