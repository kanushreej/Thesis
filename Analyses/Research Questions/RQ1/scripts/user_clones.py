import pandas as pd
import os
import numpy as np

# Define the function to preprocess issue data
def preprocess_issue_data(issue_data, stances):
    issue_data['combined_text'] = issue_data['text_raw'].fillna('') + issue_data['context_raw'].fillna('')
    issue_data = issue_data[~issue_data['combined_text'].str.contains('I am a bot', case=False)]
    issue_data = issue_data[~issue_data['combined_text'].str.contains('this action was performed automatically', case=False)]
    issue_data = issue_data[['author', 'created_utc'] + stances + ['neutral', 'irrelevant']]
    issue_data = issue_data[issue_data['irrelevant'] != 1].drop(columns=['irrelevant'])
    issue_data.rename(columns={'neutral': f'{issue}_neutral'}, inplace=True)
    for stance in stances + [f'{issue}_neutral']:
        if stance not in issue_data.columns:
            issue_data[stance] = 0
    return issue_data

# Step 1: Load the clustered user data and extract the top 20 most polarized users
clustered_data = pd.read_csv('Analyses/User Data/Clustered/usersUK_nr8.csv')
top_20_users = clustered_data.nsmallest(20, 'distance_to_center')['username'].tolist()

# Initialize a DataFrame to store versions of the top users
top_users_versions = pd.DataFrame()

# Define stance groups and opinion columns for the UK region (can be adjusted for other regions)
region = 'UK'  # Change as necessary
stance_groups = {           
    'Brexit': ['pro_brexit', 'anti_brexit'],
    'ClimateChangeUK': ['pro_climateAction', 'anti_climateAction'],
    'HealthcareUK': ['pro_NHS', 'anti_NHS'],
    'IsraelPalestineUK': ['pro_israel', 'pro_palestine'],
    'TaxationUK': ['pro_company_taxation', 'pro_worker_taxation'],
}
opinion_columns = [         
    'pro_brexit', 'anti_brexit',
    'pro_climateAction', 'anti_climateAction',
    'pro_NHS', 'anti_NHS',
    'pro_israel', 'pro_palestine',
    'pro_company_taxation', 'pro_worker_taxation',
    'Brexit_neutral', 'ClimateChangeUK_neutral',
    'HealthcareUK_neutral', 'IsraelPalestineUK_neutral',
    'TaxationUK_neutral'
]

# Step 2: For each top user, collect their data points over time from the issue files
directory = f'Analyses/Labelled Data/{region}'

for issue, stances in stance_groups.items():
    issue_file = os.path.join(directory, f'{issue}_labelled.csv')
    if not os.path.exists(issue_file):
        continue
    
    issue_data = pd.read_csv(issue_file)
    issue_data = preprocess_issue_data(issue_data, stances)
    
    # Filter data for the top 20 users
    top_user_data = issue_data[issue_data['author'].isin(top_20_users)]
    
    # Append to the overall top users versions DataFrame
    top_users_versions = pd.concat([top_users_versions, top_user_data], ignore_index=True)

# Step 3: Generate versions of each user based on their data points in chronological order
top_users_versions['created_utc'] = pd.to_datetime(top_users_versions['created_utc'])
top_users_versions.sort_values(by='created_utc', inplace=True)

# Create a version for each user at each time a data point appears
user_versions = []

for username in top_20_users:
    user_data = top_users_versions[top_users_versions['author'] == username]
    for idx, row in user_data.iterrows():
        version = user_data[user_data['created_utc'] <= row['created_utc']]
        version_summary = version.sum(numeric_only=True)
        version_summary['username'] = username
        version_summary['version_date'] = row['created_utc']
        version_summary['version_id'] = f"{username}_{row['created_utc'].strftime('%Y%m%d%H%M%S')}"
        user_versions.append(version_summary)

user_versions_df = pd.DataFrame(user_versions)

# Save the user versions data to a CSV file
output_file = 'Analyses/Research Questions/RQ1/Data/top_users_preprocessed.csv'
user_versions_df.to_csv(output_file, index=False)

print(f"User versions data has been saved to {output_file}")
