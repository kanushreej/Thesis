import os
import pandas as pd

region = 'UK'  # Set region to UK or US

user_data = pd.read_csv(f'Analyses/User Data/Clustered/users{region}_nr8.csv')  # Load user data

directory = f'Analyses/Labelled Data/{region}'  # Directory containing the issue files 

# Initialize first and last post date columns
user_data['first_post'] = pd.NaT
user_data['last_post'] = pd.NaT

# Process each issue file
for file in os.listdir(directory):
    if file.endswith("_labelled.csv"):
        issue_file = os.path.join(directory, file)
        
        issue_data = pd.read_csv(issue_file)

        # Remove data points containing bot string
        issue_data['combined_text'] = issue_data['text_raw'].fillna('') + issue_data['context_raw'].fillna('')  # Combine all text data
        issue_data = issue_data[~issue_data['combined_text'].str.contains('I am a bot', case=False)]
        issue_data = issue_data[~issue_data['combined_text'].str.contains('this action was performed automatically', case=False)]

        issue_data = issue_data[['author', 'created_utc']]  # Keep only needed columns

        # Convert 'created_utc' to datetime
        issue_data['created_utc'] = pd.to_datetime(issue_data['created_utc'])

        # Update first and last post dates
        first_post_dates = issue_data.groupby('author')['created_utc'].min().reset_index()
        first_post_dates.rename(columns={'created_utc': 'first_post'}, inplace=True)
        last_post_dates = issue_data.groupby('author')['created_utc'].max().reset_index()
        last_post_dates.rename(columns={'created_utc': 'last_post'}, inplace=True)

        user_data = user_data.merge(first_post_dates, how='left', left_on='username', right_on='author', suffixes=('', '_first'))
        user_data = user_data.merge(last_post_dates, how='left', left_on='username', right_on='author', suffixes=('', '_last'))

        user_data['first_post'] = user_data.apply(
            lambda row: min(row['first_post'], row['first_post_first']) if pd.notna(row['first_post']) and pd.notna(row['first_post_first'])
            else row['first_post'] if pd.notna(row['first_post'])
            else row['first_post_first'], axis=1)
        
        user_data['last_post'] = user_data.apply(
            lambda row: max(row['last_post'], row['last_post_last']) if pd.notna(row['last_post']) and pd.notna(row['last_post_last'])
            else row['last_post'] if pd.notna(row['last_post'])
            else row['last_post_last'], axis=1)

        user_data.drop(columns=['first_post_first', 'last_post_last', 'author'], inplace=True, errors='ignore')

# Calculate the duration between first and last post in seconds
user_data['duration'] = (user_data['last_post'] - user_data['first_post']).dt.total_seconds()

# Save updated user data
user_data.to_csv(f'Analyses/Research Questions/RQ1/Data/{region}_all_users_with_duration.csv', index=False)

print(user_data.head())
