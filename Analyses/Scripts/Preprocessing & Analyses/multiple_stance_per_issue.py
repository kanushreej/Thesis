import pandas as pd

# Load the preprocessed user data
user_data = pd.read_csv('Analyses/User Data/Preprocessed/usersUS_preprocessed.csv')

# Define the stance groups
stance_groups = {
    # 'Brexit': ['pro_brexit', 'anti_brexit'],
    # 'ClimateChangeUK': ['pro_climateAction', 'anti_climateAction'],
    # 'HealthcareUK': ['pro_NHS', 'anti_NHS'],
    # 'IsraelPalestineUK': ['pro_israel', 'pro_palestine'],
    # 'TaxationUK': ['pro_company_taxation', 'pro_worker_taxation'],
    'ImmigrationUS': ['pro_immigration', 'anti_immigration'],
    'ClimateChangeUS': ['pro_climateAction', 'anti_climateAction'],
    'HealthcareUS': ['public_healthcare', 'private_healthcare'],
    'IsraelPalestineUS': ['pro_israel', 'pro_palestine'],
    'TaxationUS': ['pro_middle_low_tax', 'pro_wealthy_corpo_tax']
}

# Initialize a dictionary to store the count of users with multiple opinions per issue
multiple_opinions_count = {}
total_multiple_opinions_users = set()

# Check for multiple opinions per issue
for issue, stances in stance_groups.items():
    # Define the columns to check for the current issue
    columns_to_check = stances + [f'{issue}_neutral']
    
    # Count users with more than one opinion (value greater than 0) in the columns related to the issue
    user_data[f'{issue}_multiple_opinions'] = user_data[columns_to_check].gt(0).sum(axis=1)
    multiple_opinions_count[issue] = (user_data[f'{issue}_multiple_opinions'] > 1).sum()
    
    # Add users with multiple opinions in the current issue to the total set of unique users
    users_with_multiple_opinions = user_data[user_data[f'{issue}_multiple_opinions'] > 1]['username']
    total_multiple_opinions_users.update(users_with_multiple_opinions)

# Print the results
for issue, count in multiple_opinions_count.items():
    print(f"Number of users with multiple opinions for {issue}: {count}")

print(f"Total number of unique users with multiple opinions across all issues: {len(total_multiple_opinions_users)}")
