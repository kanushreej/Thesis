import praw
import os
import pandas as pd
from datetime import datetime, timezone, timedelta
import time

# Initialize PRAW with your Reddit credentials
reddit = praw.Reddit(
    client_id='vInV29b0TXkkpagkYMoPLQ',
    client_secret='VS-PBH-LXW_sXBbZWJvIKta5XeB6Yw',
    user_agent='adam'
)

def collect_data(subreddit, keyword, start_date):
    """Collect posts and comments from a specific subreddit based on a keyword from a specific start date."""
    data = []
    for submission in reddit.subreddit(subreddit).search(keyword, sort='new', limit=None):
        created_date = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
        if created_date < start_date or not submission.author or not subreddit or not keyword:
            continue
        data.append({
            'subreddit': subreddit,
            'type': 'post',
            'keyword': keyword,
            'id': str(submission.id),  
            'author': str(submission.author),
            'title': submission.title,
            'body': submission.selftext,
            'created_utc': created_date.isoformat(),
            'parent_id': ''  # No parent for posts
        })

        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            data.append({
                'subreddit': subreddit,
                'type': 'comment',
                'keyword': keyword,
                'id': str(comment.id),
                'author': str(comment.author),
                'title': '',
                'body': comment.body,
                'created_utc': datetime.fromtimestamp(comment.created_utc, tz=timezone.utc).isoformat(),
                'parent_id': str(comment.parent_id)  # Parent ID for comments
            })

    return data

def fetch_parent_data(parent_id, data_ids_set, inherited_keyword):
    """Fetches missing parent data if not present in the dataset and inherits the keyword."""
    parent_type = parent_id.split('_')[0]
    parent = None  # Initialize parent as None to ensure it's defined

    try:
        if parent_type == 't1':  # Comment
            parent = reddit.comment(id=parent_id[3:])
        elif parent_type == 't3':  # Submission
            parent = reddit.submission(id=parent_id[3:])

        if parent and str(parent.id) not in data_ids_set:
            return {
                'subreddit': parent.subreddit.display_name,
                'type': 'comment' if parent_type == 't1' else 'post',
                'keyword': inherited_keyword,  # Inherit the keyword from the child
                'id': parent.id,
                'author': str(parent.author) if parent.author else None,
                'title': parent.title if parent_type == 't3' else '',
                'body': parent.body if parent_type == 't1' else parent.selftext,
                'created_utc': datetime.fromtimestamp(parent.created_utc, tz=timezone.utc).isoformat(),
                'parent_id': ''
            }
    except Exception as e:
        print(f"Error fetching parent data: {e}")

    return None


def validate_and_append_data(df):
    """Validates existing data and appends missing parent data."""
    print("Validating data")
    df['parent_id'] = df['parent_id'].astype(str)  # Ensure parent_id is string
    data_ids_set = set(df['id'])
    new_rows = []

    for index, row in df.iterrows():
        # Handle potential NaN or missing parent_id values
        if row['type'] == 'comment' and row['parent_id'] and not pd.isna(row['parent_id']):
            print(f"Fetching parent data for {row}")
            parent_data = fetch_parent_data(row['parent_id'], data_ids_set, row['keyword'])
            if parent_data:
                new_rows.append(parent_data)
                data_ids_set.add(parent_data['id'])

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    
    # Ensure all critical fields are filled
    df = df.dropna(subset=['subreddit', 'type', 'keyword', 'id', 'author', 'created_utc'])

    # Remove entries with missing dependent data
    while True:
        initial_len = len(df)
        valid_ids = set(df['id'])
        df = df[df['parent_id'].apply(lambda x: x == '' or x in valid_ids)]
        if len(df) == initial_len:
            break

    return df

def main():
    start_year = 2010
    start_date = datetime(start_year, 1, 1, tzinfo=timezone.utc)
    ## UK Subreddits ##
    subreddits = ['unitedkingdom', 'ukpolitics', 'AskUK', 'Scotland', 'Wales', 'northernireland', 'england', 'europe', 'uknews', 'LabourUK', 'Labour', 'tories', 'nhs',  'brexit', 'europeanunion']
    
    ## US Subreddits ##
    #subreddits = [ 'Politics', 'Conservative', 'Geopolitics', 'Libertarian', 'Democrats', 'PoliticalDiscussion', 'GreenParty', 'Republican', 'Neoliberal', 'Liberal', 'Conservatives', 'Climatechange', 'Healthcare', 'Progressive', 'USpolitics']

    ## UK Issues ##
    issues = ['Brexit']#,'ClimateChangeUK','HealthcareUK','IsraelPalestine','TaxationUK']
    ## US Issues ##
    #issues = ['HealthcareUS','IsraelPalestine','TaxationUS','ImmigrationUS']

    base_dir = "/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Keyword Selection/Final"
    data_dir = "/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Subreddit Data/UK" # update path to US or UK 

    for issue in issues:
        csv_path = os.path.join(data_dir, f"{issue}_data.csv")
        keyword_file = os.path.join(base_dir, f"{issue}_final_keywords.csv")
        keywords = pd.read_csv(keyword_file)['Keyword'].tolist()

        # Ensure the file exists and has the expected structure
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if df.empty or 'id' not in df.columns:
                df = pd.DataFrame(columns=['subreddit', 'type', 'keyword', 'id', 'author', 'title', 'body', 'created_utc', 'parent_id'])
        else:
            df = pd.DataFrame(columns=['subreddit', 'type', 'keyword', 'id', 'author', 'title', 'body', 'created_utc', 'parent_id'])
            df.to_csv(csv_path, index=False)

        if not df.empty:
            existing_ids = set(df['id'])
            for subreddit in subreddits:
                for keyword in keywords:
                    try:
                        subreddit_data = collect_data(subreddit, keyword, start_date)
                        df_new = pd.DataFrame([item for item in subreddit_data if item['id'] not in existing_ids])
                        if not df_new.empty:
                            df_new = validate_and_append_data(df_new)
                            df_new.to_csv(csv_path, mode='a', header=False, index=False)
                            existing_ids.update(df_new['id'])
                        print(f"Info for keyword '{keyword}' from subreddit '{subreddit}' added to {csv_path}")
                    except Exception as e:
                        print(f"Error collecting data for keyword '{keyword}' in subreddit '{subreddit}': {e}")
                        time.sleep(60)  # Adjust sleep time based on API usage

if __name__ == '__main__':
    main()