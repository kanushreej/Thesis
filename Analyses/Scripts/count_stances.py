import pandas as pd

# Load the preprocessed user data file
user_data = pd.read_csv('Analyses/User Data/usersUK_preprocessed.csv')

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
