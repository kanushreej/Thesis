import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

region = 'US' # Set region to UK or US

if region == 'UK':
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
if region == 'US':
    stance_groups = {        
        'ImmigrationUS': ['pro_immigration', 'anti_immigration'],
        'ClimateChangeUS': ['pro_climateAction', 'anti_climateAction'],
        'HealthcareUS': ['public_healthcare', 'private_healthcare'],
        'IsraelPalestineUS': ['pro_israel', 'pro_palestine'],
        'TaxationUS': ['pro_middle_low_tax', 'pro_wealthy_corpo_tax']
    }
    opinion_columns = [          
    'pro_immigration', 'anti_immigration',
    'pro_climateAction', 'anti_climateAction',
    'public_healthcare', 'private_healthcare',
    'pro_israel', 'pro_palestine',
    'pro_middle_low_tax', 'pro_wealthy_corpo_tax',
    'ImmigrationUS_neutral', 'ClimateChangeUS_neutral',
    'HealthcareUS_neutral', 'IsraelPalestineUS_neutral',
    'TaxationUS_neutral',
    ]


issues = list(stance_groups.keys())

user_data = pd.read_csv(f'Analyses/User Data/All users/{region}_all_users.csv') # Load user data
print(f"Total number of users: {len(user_data)}")

user_data = user_data[['user_id', 'username', 'account_creation_date', 'comment_karma', 'post_karma']]
directory = f'Analyses/Labelled Data/{region}' # Directory containing the issue files 

# Process each issue file
for issue, stances in stance_groups.items():
    issue_file = os.path.join(directory, f'{issue}_labelled.csv')
    if not os.path.exists(issue_file):
        print(f"File not found: {issue_file}")
        continue

    issue_data = pd.read_csv(issue_file)

    # Remove data points containing bot string
    issue_data['combined_text'] = issue_data['text_raw'].fillna('') + issue_data['context_raw'].fillna('') # Combine all text data
    issue_data = issue_data[~issue_data['combined_text'].str.contains('I am a bot', case=False)]
    issue_data = issue_data[~issue_data['combined_text'].str.contains('this action was performed automatically', case=False)]
    #issue_data.drop(columns=['combined_text'], inplace=True)
    
    issue_data = issue_data[['author', stances[0], stances[1], 'neutral', 'irrelevant']] # Keep only needed columns in the issue data
    issue_data = issue_data[issue_data['irrelevant'] != 1].drop(columns=['irrelevant']) # Remove data points with irrelevant == 1 and then drop the column
    issue_data.rename(columns={'neutral': f'{issue}_neutral'}, inplace=True)  # Convert the 'neutral' column to '{issue}_neutral'

    for stance in stances + [f'{issue}_neutral']:
        if stance not in issue_data.columns:
            issue_data[stance] = 0

    for stance in stances + [f'{issue}_neutral']:
        stance_counts = issue_data.groupby('author')[stance].sum().reset_index()
        stance_counts.rename(columns={stance: f'{stance}_count'}, inplace=True)
        user_data = user_data.merge(stance_counts, how='left', left_on='username', right_on='author')
        user_data[f'{stance}_count'] = user_data[f'{stance}_count'].fillna(0).astype(int)
        user_data.drop(columns=['author'], inplace=True)

# Print counts of user involvement in issues
user_data['issues_count'] = user_data[[f'{stance}_count' for stances in stance_groups.values() for stance in stances] + 
                                      [f'{issue}_neutral_count' for issue in issues]].gt(0).sum(axis=1)
for i in range(len(issues) + 1):
    print(f"Users present in {i} issues: {sum(user_data['issues_count'] == i)}")

# Remove users with only irrelevant stances
user_data = user_data[user_data['issues_count'] > 0].drop(columns=['issues_count'])

print(f"Total number of users left after removal of irrelevant: {len(user_data)}")

for issue, stances in stance_groups.items():
    for stance in stances + [f'{issue}_neutral']:
        user_data.drop(columns=[stance], inplace=True, errors='ignore')
        user_data.rename(columns={f'{stance}_count': stance}, inplace=True)



# Filter by Normal Distribution
user_data['total_opinions'] = user_data[opinion_columns].sum(axis=1)
mu, std = norm.fit(user_data['total_opinions'])


plt.figure(figsize=(10, 6))
plt.hist(user_data['total_opinions'], bins=30, density=True, alpha=0.6, color='b')
xmin, xmax = -200, 200  # Adjusted x-axis range
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = f"Fit results: mu = {mu:.2f},  std = {std:.2f}"
plt.title(title)
plt.xlabel('Number of Opinions')
plt.ylabel('Density')
plt.xlim(xmin, xmax) 
plt.grid(True)
plt.show()

# Print key values of the distribution
print(f"Mean (mu): {mu}")
print(f"Standard Deviation (std): {std}")
print(f"1 standard deviation range: {mu - std} to {mu + std}")
print(f"2 standard deviations range: {mu - 2 * std} to {mu + 2 * std}")

# Filter users within 2 standard deviations
lower_bound = mu - 2 * std
upper_bound = mu + 2 * std
user_data = user_data[(user_data['total_opinions'] >= lower_bound) & (user_data['total_opinions'] <= upper_bound)]
user_data.drop(columns=['total_opinions'], inplace=True, errors='ignore')

# Print the number of users to be deleted by normalization
print(f"Number of users left after removal of outliers: {len(user_data)}")

# Create one issue column
for issue, stances in stance_groups.items():
    pro_stance, anti_stance = stances
    neutral_column = f'{issue}_neutral'
    
    user_data[issue] = (user_data[f'{pro_stance}'] - user_data[f'{anti_stance}']) / (user_data[neutral_column] + 1)

# Remove old stance columns
for issue, stances in stance_groups.items():
    for stance in stances + [f'{issue}_neutral']:
        user_data.drop(columns=[stance], inplace=True, errors='ignore')


user_data.to_csv(f'Analyses/User Data/Collected Stances/users{region}_general.csv', index=False)