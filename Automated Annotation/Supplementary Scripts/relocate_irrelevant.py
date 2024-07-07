import os
import pandas as pd

# Define the path and issues with their respective stances
#base_path = "/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Automated Annotation/Labelled Data/UK"
# issues = {
#     'Brexit': ['pro_brexit', 'anti_brexit'],
#     'ClimateChangeUK': ['pro_climateAction', 'anti_climateAction'],
#     'HealthcareUK': ['pro_NHS', 'anti_NHS'],
#     'IsraelPalestineUK': ['pro_israel', 'pro_palestine'],
#     'TaxationUK': ['pro_company_taxation', 'pro_worker_taxation']
# }
base_path = "/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Automated Annotation/Training Data/US"
issues = {
    'ImmigrationUS': ['pro_immigration', 'anti_immigration'],
    'ClimateChangeUS': ['pro_climateAction', 'anti_climateAction'],
    'HealthcareUS': ['public_healthcare', 'private_healthcare'],
    'IsraelPalestineUS': ['pro_israel', 'pro_palestine'],
    'TaxationUS': ['pro_middle_low_tax', 'pro_wealthy_corpo_tax']
}

extra_labels = ['neutral', 'irrelevant']

def copy_relevant_rows(issue_name, stances):
    df = pd.read_csv(os.path.join(base_path, f"{issue_name}_labelled.csv"))
    all_columns = df.columns.tolist()
    
    for index, row in df.iterrows():
        for other_issue, other_stances in issues.items():
            if other_issue != issue_name:
                if any(row[col] == 1 for col in other_stances):
                    other_issue_path = os.path.join(base_path, f"{other_issue}_labelled.csv")
                    if os.path.exists(other_issue_path):
                        other_df = pd.read_csv(other_issue_path)
                    else:
                        other_df = pd.DataFrame(columns=all_columns)
                    
                    other_df = pd.concat([other_df, pd.DataFrame([row], columns=all_columns)], ignore_index=True)
                    other_df.to_csv(other_issue_path, index=False)
        
        if all(row[col] == 0 for col in stances) and row['neutral'] == 0:
            df.at[index, 'irrelevant'] = 1
    
    df.to_csv(os.path.join(base_path, f"{issue_name}_labelled.csv"), index=False)

def finalize_issue_files():
    for issue_name, stances in issues.items():
        df = pd.read_csv(os.path.join(base_path, f"{issue_name}_labelled.csv"))
        relevant_columns = stances + extra_labels
        all_columns = df.columns.tolist()
        irrelevant_columns = [col for col in all_columns if col not in relevant_columns and (col.startswith('pro') or col.startswith('anti') or col.startswith('private') or col.startswith('public'))]
        
        df.drop(columns=irrelevant_columns, inplace=True)
        df.drop_duplicates(subset=['id'], inplace=True)
        df.to_csv(os.path.join(base_path, f"{issue_name}_labelled.csv"), index=False)

for issue, stances in issues.items():
    copy_relevant_rows(issue, stances)

finalize_issue_files()
