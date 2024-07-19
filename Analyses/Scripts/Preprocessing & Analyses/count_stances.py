import pandas as pd

# Load the preprocessed user data file
user_data = pd.read_csv('Analyses/User Data/usersUK_filtered.csv')

# Initialize dictionaries to store the counts
stance_min_count = {}
stance_total_count = {}

# List of all stance columns to consider
stance_columns = [
    'pro_brexit', 'anti_brexit',
    'pro_climateAction', 'anti_climateAction',
    'pro_NHS', 'anti_NHS',
    'pro_israel', 'pro_palestine',
    'pro_company_taxation', 'pro_worker_taxation',
    'Brexit_neutral', 'ClimateChangeUK_neutral',
    'HealthcareUK_neutral', 'IsraelPalestineUK_neutral',
    'TaxationUK_neutral'
]

# stance_columns = [
#     'pro_immigration', 'anti_immigration',
#     'pro_climateAction', 'anti_climateAction',
#     'public_healthcare', 'private_healthcare',
#     'pro_israel', 'pro_palestine',
#     'pro_middle_low_tax', 'pro_wealthy_corpo_tax',
#     'ImmigrationUS_neutral', 'ClimateChangeUS_neutral',
#     'HealthcareUS_neutral', 'IsraelPalestineUS_neutral',
#     'TaxationUS_neutral',
# ]

stance_groups = {
    'Brexit': ['pro_brexit', 'anti_brexit'],
    'ClimateChangeUK': ['pro_climateAction', 'anti_climateAction'],
    'HealthcareUK': ['pro_NHS', 'anti_NHS'],
    'IsraelPalestineUK': ['pro_israel', 'pro_palestine'],
    'TaxationUK': ['pro_company_taxation', 'pro_worker_taxation'],
}
issues = list(stance_groups.keys())

# Calculate the counts for each stance column
for stance in stance_columns:
    stance_min_count[stance] = (user_data[stance] > 0).sum()
    stance_total_count[stance] = user_data[stance].sum()

# Print the results
print("Count of users with a minimum of 1 value for each stance column:")
for stance, count in stance_min_count.items():
    print(f"{stance}: {count}")

print("\nTotal values of each stance column:")
for stance, total in stance_total_count.items():
    print(f"{stance}: {total}")

user_data['issues_count'] = user_data[[f'{stance}' for stances in stance_groups.values() for stance in stances] + 
                                      [f'{issue}_neutral' for issue in issues]].gt(0).sum(axis=1)

for i in range(len(issues) + 1):
    print(f"Users present in {i} issues: {sum(user_data['issues_count'] == i)}")
