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
    max_retries = 3  # Set a maximum number of retries
    retry_count = 0

    while retry_count < max_retries:
        try:
            user = reddit.redditor(username)
            #print(f"Collecting data for {username}")
            return {
                'user_id': user.id,
                'username': username,
                'account_creation_date': str(user.created_utc),
                'comment_karma': user.comment_karma,
                'post_karma': user.link_karma,
                'is_employee': user.is_employee,
                'is_verified': user.has_verified_email,
                'is_mod': user.is_mod,
                'is_gold': user.is_gold
            }
        except praw.exceptions.APIException as e:
            print(f"API Exception while fetching data for {username}: {e}")
            retry_count += 1
            sleep_duration = 120
            print(f"Retrying in {sleep_duration} seconds...")
            time.sleep(sleep_duration)
        except Exception as e:
            print(f"Failed to fetch data for {username}: {e}")
            return None

    print(f"Failed to fetch data for {username} after {max_retries} attempts.")
    return None

def collect_unique_users(data_dir, output_dir):
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return

    files = [f for f in os.listdir(data_dir) if f.endswith('_data.csv')]
    if not files:
        print("No data files found to process.")
        return

    for file in files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f"File {file} not found in the directory. Skipping...")
            continue

        issue_name = file.replace('_data.csv', '')
        user_data_file = os.path.join(output_dir, f'{issue_name}_unique_users.csv')
        existing_usernames = set()

        if os.path.exists(user_data_file):
            existing_data = pd.read_csv(user_data_file)
            existing_usernames.update(existing_data['username'].dropna().unique())

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Failed to read {file}: {e}")
            continue

        total_users = len(df['author'].drop_duplicates().tolist())
        user_data_list = []
        for index, username in enumerate(df['author'].drop_duplicates().tolist()):
            if username in existing_usernames:
                continue
            user_data = fetch_user_data(username)
            if user_data:
                print(f"Collecting for {username} from {issue_name} {index + 1}/{total_users}")
                user_data['issue'] = issue_name
                user_data_list.append(user_data)

        # Save the updated data
        mode = 'a' if os.path.exists(user_data_file) else 'w'
        pd.DataFrame(user_data_list).to_csv(user_data_file, mode=mode, index=False, header=not os.path.exists(user_data_file))
        print(f"Data for {issue_name} updated in {user_data_file}")

def main():
    data_dir = "/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Collected Data/UK/Subreddit Data"
    output_dir = "/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Collected Data/UK/User Data"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    collect_unique_users(data_dir, output_dir)

if __name__ == '__main__':
    main()
