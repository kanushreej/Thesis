import praw
import os
import pandas as pd
from datetime import datetime, timezone
import time

# Create a Reddit instance
reddit = praw.Reddit(
    client_id='vInV29b0TXkkpagkYMoPLQ',
    client_secret='VS-PBH-LXW_sXBbZWJvIKta5XeB6Yw',
    user_agent='adam'
)

def collect_data(subreddit, start_date):
    """Collect posts and comments from a specific subreddit after a start date."""
    data = []
    count = 0
    # Convert the start date string to a datetime object
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)

    for submission in reddit.subreddit(subreddit).new(limit=None):
        submission_created = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
        if submission_created >= start_datetime:
            #if count < item_limit:
            data.append({
                'subreddit': subreddit,
                'type': 'post',
                'id': submission.id,
                'author': str(submission.author),
                'title': submission.title,
                'body': submission.selftext,
                'created_utc': submission_created.isoformat()
            })
            #    count += 1

            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list():
                comment_created = datetime.fromtimestamp(comment.created_utc, tz=timezone.utc)
                if comment_created >= start_datetime: #and count < item_limit:
                    data.append({
                        'subreddit': subreddit,
                        'type': 'comment',
                        'id': comment.id,
                        'author': str(comment.author),
                        'title': '',
                        'body': comment.body,
                        'created_utc': comment_created.isoformat()
                    })
                    count += 1
                #if count >= item_limit:
                #    break
            #if count >= item_limit:
            #    break

    return data

def main():
    subreddits = ['ukpolitics']
    start_date = '2010-01-01'  # Start date for collecting data

    csv_path = 'ukpolitics.csv'
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=['subreddit', 'type', 'id', 'author', 'title', 'body', 'created_utc']).to_csv(csv_path, index=False)

    existing_ids = pd.read_csv(csv_path)['id'].tolist()
    existing_ids_set = set(existing_ids)

    for subreddit in subreddits:
        try:
            subreddit_data = collect_data(subreddit, start_date)
            df_new = pd.DataFrame([item for item in subreddit_data if item['id'] not in existing_ids_set])
            if not df_new.empty:
                df_new.to_csv(csv_path, mode='a', header=False, index=False)
                existing_ids_set.update(df_new['id'])
                print(f"Data from subreddit '{subreddit}' added to {csv_path}")
        except Exception as e:
            print(f"Error collecting data from subreddit '{subreddit}': {e}")
            time.sleep(60)  # Simple delay, adjust as needed 

if __name__ == '__main__':
    main()
