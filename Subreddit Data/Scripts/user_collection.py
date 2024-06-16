import praw
import os
import pandas as pd
import time

# Initialize PRAW with your credentials
reddit = praw.Reddit(
    client_id='vInV29b0TXkkpagkYMoPLQ',
    client_secret='VS-PBH-LXW_sXBbZWJvIKta5XeB6Yw',
    user_agent='adam'
)

def fetch_user_data(username):
    """Fetch detailed user data using Reddit's API, handling missing attributes gracefully."""
    try:
        user = reddit.redditor(username)
        print(f"Collecting data for {username}")
        user_data = {
            'username': username,
            'account_creation_date': str(reddit.redditor(username).created_utc),
            'comment_karma': reddit.redditor(username).comment_karma,
            'post_karma': reddit.redditor(username).link_karma,
            'is_employee': reddit.redditor(username).is_employee,
            'is_verified': reddit.redditor(username).has_verified_email,
            'is_mod': reddit.redditor(username).is_mod,
            'is_gold': reddit.redditor(username).is_gold
        }
        return user_data
    except praw.exceptions.APIException as e:
        print(f"API Exception while fetching data for {username}: {e}")
        time.sleep(120)  # Sleep to respect Reddit's rate limit
        return fetch_user_data(username)  # Retry fetching data
    except Exception as e:
        print(f"Other error fetching data for {username}: {e}")
        return None

def collect_unique_users(data_dir, output_dir, combined=False):
    # Check if the data directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return

    files = [f for f in os.listdir(data_dir) if f.endswith('_data.csv')]
    all_users_data = []

    if not files:
        print("No data files found to process.")
        return

    for file in files:
        file_path = os.path.join(data_dir, file)
        # Check if the file exists to handle cases where it might have been moved or deleted
        if not os.path.exists(file_path):
            print(f"File {file} not found in the directory. Skipping...")
            continue

        issue_name = file.replace('_data.csv', '')
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Failed to read {file}: {e}")
            continue

        # Fetch unique users
        usernames = df['author'].drop_duplicates().tolist()
        user_data_list = []

        for username in usernames:
            user_data = fetch_user_data(username)
            if user_data:
                user_data['issue'] = issue_name
                user_data_list.append(user_data)

        if combined:
            all_users_data.extend(user_data_list)
        else:
            if user_data_list:
                user_output_file = os.path.join(output_dir, f'{issue_name}_unique_users.csv')
                pd.DataFrame(user_data_list).to_csv(user_output_file, index=False)
                print(f"Data for {issue_name} saved to {user_output_file}")

    if combined and all_users_data:
        combined_file_path = os.path.join(output_dir, 'all_issues_unique_users.csv')
        pd.DataFrame(all_users_data).to_csv(combined_file_path, index=False)
        print(f"Combined user data saved to {combined_file_path}")

def main():
    data_dir = "/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Subreddit Data/UK"  # Path to the directory containing issue data CSVs
    output_dir = "/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Subreddit Data/UK/User Data"  # Path to save unique user data
    combined = True  # Set to False if you want separate files for each issue

    if not os.path.exists(output_dir):
        print(f"Creating output directory at {output_dir}")
        os.makedirs(output_dir)
    
    collect_unique_users(data_dir, output_dir, combined)

if __name__ == '__main__':
    main()
