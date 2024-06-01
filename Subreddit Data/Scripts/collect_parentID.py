import praw
import os
import pandas as pd
from datetime import datetime, timezone, timedelta
import time

reddit = praw.Reddit(
    client_id='wEP3RUiICi5YvjDKhEKlkg',
    client_secret='FP8O4SUc6ocAGNgCx5HB1-nczz6uQw',
    user_agent='script:keyword_extractor:v1.0 (by u/Queasy-Parsnip-8103)'
)

def fetch_parent_id(comment_id):
    """Fetch the parent ID of a comment."""
    try:
        comment = reddit.comment(id=comment_id)
        return str(comment.parent_id)
    except Exception as e:
        print(f"Error fetching parent ID for comment {comment_id}: {e}")
        return ''

def fetch_parent_data(parent_id, keyword):
    """Fetch the parent data based on the parent ID and assign the keyword."""
    try:
        parent = reddit.comment(id=parent_id) if parent_id.startswith('t1_') else reddit.submission(id=parent_id[3:])
        return {
            'subreddit': parent.subreddit.display_name,
            'type': 'comment' if parent_id.startswith('t1_') else 'post',
            'keyword': keyword,  # Assign the same keyword as the child
            'id': str(parent.id),
            'author': str(parent.author),
            'title': '' if parent_id.startswith('t1_') else parent.title,
            'body': parent.body if parent_id.startswith('t1_') else parent.selftext,
            'created_utc': datetime.fromtimestamp(parent.created_utc, tz=timezone.utc).isoformat(),
            'parent_id': '' if not parent_id.startswith('t1_') else str(parent.parent_id)
        }
    except Exception as e:
        print(f"Error fetching parent data for {parent_id}: {e}")
        return {}

def collect_data(subreddit, keyword, start_date):
    """Collect posts and comments from a specific subreddit based on a keyword from a specific start date."""
    data = []
    try:
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
                if not comment.author or not subreddit or not keyword:
                    continue
                parent_id = str(comment.parent_id)
                data.append({
                    'subreddit': subreddit,
                    'type': 'comment',
                    'keyword': keyword,
                    'id': str(comment.id),
                    'author': str(comment.author),
                    'title': '',
                    'body': comment.body,
                    'created_utc': datetime.fromtimestamp(comment.created_utc, tz=timezone.utc).isoformat(),
                    'parent_id': parent_id  # Parent ID for comments
                })
        return data
    except praw.exceptions.APIException as e:
        print(f"API Exception for {subreddit} with keyword {keyword}: {e}")
        time.sleep(120)  # Sleep longer to respect Reddit's rate limit
        return []
    except Exception as e:
        print(f"General exception for {subreddit} with keyword {keyword}: {e}")
        return []

def update_parent_ids(df):
    """Ensure all comments have a parent ID and fetch parent data if missing."""
    if 'parent_id' not in df.columns:
        df['parent_id'] = ''

    parent_data = []
    missing_parent_ids = df[(df['type'] == 'comment') & (df['parent_id'] == '')]['id'].tolist()
    
    # Fetch missing parent IDs
    for comment_id in missing_parent_ids:
        parent_id = fetch_parent_id(comment_id)
        df.loc[df['id'] == comment_id, 'parent_id'] = parent_id

    # Ensure all parent_id values are strings
    df['parent_id'] = df['parent_id'].astype(str)
    
    # Fetch and add parent data if needed
    for index, row in df[df['type'] == 'comment'].iterrows():
        parent_id = row['parent_id']
        if parent_id and parent_id != 'nan' and df[df['id'] == parent_id[3:]].empty:
            parent_data_entry = fetch_parent_data(parent_id, row['keyword'])
            if parent_data_entry:
                parent_data.append(parent_data_entry)

    return df, parent_data

def verify_and_collect_data(subreddits, issues, base_dir, data_dir, start_date):
    for issue in issues:
        csv_path = os.path.join(data_dir, f"{issue}_data.csv")
        keyword_file = os.path.join(base_dir, f"{issue}_final_keywords.csv")
        keywords = pd.read_csv(keyword_file)['Keyword'].tolist()

        existing_data = pd.read_csv(csv_path, dtype={'id': str}) if os.path.exists(csv_path) else pd.DataFrame()
        existing_combinations = set(zip(existing_data['subreddit'], existing_data['keyword']))
        existing_ids = set(existing_data['id'])  # Set of existing IDs

        needed_combinations = {(subreddit, keyword) for subreddit in subreddits for keyword in keywords}
        missing_combinations = needed_combinations - existing_combinations

        for subreddit, keyword in missing_combinations:
            print(f"Collecting missing data for subreddit: {subreddit}, keyword: {keyword}")
            data = collect_data(subreddit, keyword, start_date)
            if data:
                df_new = pd.DataFrame(data)
                df_new = df_new[~df_new['id'].isin(existing_ids)]
                if not df_new.empty:
                    df_new.to_csv(csv_path, mode='a', header=False, index=False)
                    print(f"Added missing data for {subreddit} - {keyword} to {csv_path}")
                    existing_ids.update(df_new['id'].tolist())

        # Always update parent IDs for existing data and fetch parent data if needed
        if not existing_data.empty:
            existing_data, parent_data = update_parent_ids(existing_data)
            if parent_data:
                df_parents = pd.DataFrame(parent_data)
                existing_data = pd.concat([existing_data, df_parents], ignore_index=True)
            # Deduplicate based on ID
            existing_data = existing_data.drop_duplicates(subset=['id'])
            existing_data.to_csv(csv_path, index=False)
            print(f"Updated parent IDs and added parent data in {csv_path}")

def main():
    start_year = 2010
    start_date = datetime(start_year, 1, 1, tzinfo=timezone.utc)

    subreddits = ['Politics', 'Conservative', 'Geopolitics', 'Libertarian', 'Democrats',
                  'PoliticalDiscussion', 'GreenParty', 'Republican', 'Neoliberal', 'Liberal',
                  'Conservatives', 'Climatechange', 'Healthcare', 'Progressive', 'USpolitics']

    issues = ['ClimateChangeUS', 'HealthcareUS', 'IsraelPalestine', 'TaxationUS', 'ImmigrationUS']
    base_dir = "/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Keyword Selection/Final"
    data_dir = "/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Subreddit Data/US"

    verify_and_collect_data(subreddits, issues, base_dir, data_dir, start_date)

if __name__ == '__main__':
    main()
