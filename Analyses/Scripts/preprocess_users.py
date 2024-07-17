import pandas as pd
import os

# Step 1: Create the list of issues through the stance groups variable
# stance_groups = {
#     'Brexit': ['pro_brexit', 'anti_brexit'],
#     'ClimateChangeUK': ['pro_climateAction', 'anti_climateAction'],
#     'HealthcareUK': ['pro_NHS', 'anti_NHS'],
#     'IsraelPalestineUK': ['pro_israel', 'pro_palestine'],
#     'TaxationUK': ['pro_company_taxation', 'pro_worker_taxation'],
# }
stance_groups = {
        'ImmigrationUS': ['pro_immigration', 'anti_immigration'],
        'ClimateChangeUS': ['pro_climateAction', 'anti_climateAction'],
        'HealthcareUS': ['public_healthcare', 'private_healthcare'],
        'IsraelPalestineUS': ['pro_israel', 'pro_palestine'],
        'TaxationUS': ['pro_middle_low_tax', 'pro_wealthy_corpo_tax']
    }
issues = list(stance_groups.keys())

# Step 2: Load the user data file and print a count of the total number of users
user_data = pd.read_csv('Analyses/User Data/US_all_users.csv')
print(f"Total number of users: {len(user_data)}")

user_data = user_data[['user_id', 'username', 'account_creation_date', 'comment_karma', 'post_karma']]

# Directory containing the issue-based files
directory = 'Analyses/Labelled Data'

# Step 8: Initialize columns for each issue in the user data
for issue, stances in stance_groups.items():
    for stance in stances:
        user_data[stance] = 0
    user_data[f'{issue}_neutral'] = 0

# Step 4-10: Process each issue-based file
for issue, stances in stance_groups.items():
    issue_file = os.path.join(directory, f'{issue}_labelled.csv')
    if not os.path.exists(issue_file):
        print(f"File not found: {issue_file}")
        continue

    issue_data = pd.read_csv(issue_file)

    # Step 5: Keep only specific columns in the issue data
    issue_data = issue_data[['author', stances[0], stances[1], 'neutral', 'irrelevant']]

    # Step 6: Remove data points with irrelevant == 1 and then drop the column
    issue_data = issue_data[issue_data['irrelevant'] != 1].drop(columns=['irrelevant'])

    # Step 7: Convert the 'neutral' column to '{issue}_neutral'
    issue_data.rename(columns={'neutral': f'{issue}_neutral'}, inplace=True)

    # Ensure all stance columns exist in the issue data
    for stance in stances + [f'{issue}_neutral']:
        if stance not in issue_data.columns:
            issue_data[stance] = 0

    # Step 8: Append the columns stance 1, stance 2, and {issue}_neutral to the user data
    for stance in stances + [f'{issue}_neutral']:
        stance_counts = issue_data.groupby('author')[stance].sum().reset_index()
        stance_counts.rename(columns={stance: f'{stance}_count'}, inplace=True)
        user_data = user_data.merge(stance_counts, how='left', left_on='username', right_on='author')
        user_data[f'{stance}_count'] = user_data[f'{stance}_count'].fillna(0).astype(int)
        user_data.drop(columns=['author'], inplace=True)

# Step 11: Print counts of user involvement in issues
user_data['issues_count'] = user_data[[f'{stance}_count' for stances in stance_groups.values() for stance in stances] + 
                                      [f'{issue}_neutral_count' for issue in issues]].gt(0).sum(axis=1)

for i in range(len(issues) + 1):
    print(f"Users present in {i} issues: {sum(user_data['issues_count'] == i)}")

# Remove users who were present in none of the issues
user_data = user_data[user_data['issues_count'] > 0].drop(columns=['issues_count'])

# Step 12: Print a count of the total number of users left
print(f"Total number of users left: {len(user_data)}")


# Step 13: Save the new dataframe to a new CSV file
user_data.to_csv('Analyses/User Data/usersUS_preprocessed.csv', index=False)
