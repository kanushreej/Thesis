import pandas as pd

import pandas as pd
import numpy as np

# Load the preprocessed top users' versions data
top_users_versions_df = pd.read_csv('Analyses/Research Questions/RQ1/Data/topUK_users_preprocessed.csv')

# Define stance groups and opinion columns
region = 'UK'  # Set region to UK or US
if region == 'UK':
    stance_groups = {
        'Brexit': ['pro_brexit', 'anti_brexit'],
        'ClimateChangeUK': ['pro_climateAction', 'anti_climateAction'],
        'HealthcareUK': ['pro_NHS', 'anti_NHS'],
        'IsraelPalestineUK': ['pro_israel', 'pro_palestine'],
        'TaxationUK': ['pro_company_taxation', 'pro_worker_taxation'],
    }
elif region == 'US':
    stance_groups = {
        'ImmigrationUS': ['pro_immigration', 'anti_immigration'],
        'ClimateChangeUS': ['pro_climateAction', 'anti_climateAction'],
        'HealthcareUS': ['public_healthcare', 'private_healthcare'],
        'IsraelPalestineUS': ['pro_israel', 'pro_palestine'],
        'TaxationUS': ['pro_middle_low_tax', 'pro_wealthy_corpo_tax']
    }

# Normalize and process the top 20 users data
for issue, stances in stance_groups.items():
    pro_stance, anti_stance = stances
    neutral_column = f'{issue}_neutral'
    
    top_users_versions_df[issue] = (top_users_versions_df[f'{pro_stance}'] - top_users_versions_df[f'{anti_stance}']) / (top_users_versions_df[neutral_column] + 1)

# Remove old stance columns
for issue, stances in stance_groups.items():
    for stance in stances + [f'{issue}_neutral']:
        top_users_versions_df.drop(columns=[stance], inplace=True, errors='ignore')

# Add the 'top_20', 'version_date', and 'version_id' columns to the top users data
top_users_versions_df['top_20'] = True

# Load the general user data
general_user_data = pd.read_csv(f'Analyses/Research Questions/RQ1/Data/users{region}_general_with_top_20.csv')

# Remove the 'comment_karma' and 'post_karma' columns from the general user data
general_user_data.drop(columns=['comment_karma', 'post_karma'], inplace=True)

# Combine the general user data with the top users data
combined_data = pd.concat([general_user_data, top_users_versions_df], ignore_index=True)

# Save the combined data to a CSV file
output_file = f'Analyses/Research Questions/RQ1/Data/users{region}_general_with_top_20.csv'
combined_data.to_csv(output_file, index=False)

print(f"Combined data with top 20 users has been saved to {output_file}")

# Count the number of users with top_20 as True
top_20_count = combined_data['top_20'].sum()

# Print the result
print(f"Number of users with top_20 as True: {top_20_count}")
