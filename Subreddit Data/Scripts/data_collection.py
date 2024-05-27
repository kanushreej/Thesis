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

def collect_data(subreddit, keyword, start_date):
    """Collect posts and comments from a specific subreddit based on a keyword from a specific start date."""
    data = []
    for submission in reddit.subreddit(subreddit).search(keyword, sort='new', limit=None):
        created_date = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
        if created_date < start_date:
            break
        data.append({
            'subreddit': subreddit,
            'type': 'post',
            'keyword': keyword,
            'id': str(submission.id),  
            'author': str(submission.author),
            'title': submission.title,
            'body': submission.selftext,
            'created_utc': created_date.isoformat()
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
                'created_utc': datetime.fromtimestamp(comment.created_utc, tz=timezone.utc).isoformat()
            })

    return data

def main():
    start_year = 2010
    start_date = datetime(start_year, 1, 1, tzinfo=timezone.utc)
    subreddits = ['unitedkingdom', 'ukpolitics', 'AskUK', 'Scotland', 'Wales', 'northernireland',
                  'england', 'europe', 'uknews', 'LabourUK', 'Labour', 'tories', 'nhs', 
                  'brexit', 'europeanunion']

    issues = ['Brexit']
    base_dir = "/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Keyword Selection/Final"
    data_dir = "/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Subreddit Data/UK"

    for issue in issues:
        csv_path = os.path.join(data_dir, f"{issue}_data.csv")
        keyword_file = os.path.join(base_dir, f"{issue}_final_keywords.csv")

        if not os.path.exists(csv_path):
            pd.DataFrame(columns=['subreddit', 'type', 'keyword', 'id', 'author', 'title', 'body', 'created_utc']).to_csv(csv_path, index=False)
        else:
            df = pd.read_csv(csv_path, dtype={'id': str})
            df['id'] = df['id'].astype(str)  # Ensure 'id' is string
            df.to_csv(csv_path, index=False)

        keywords = pd.read_csv(keyword_file)['Keyword'].tolist()
        existing_ids = pd.read_csv(csv_path, dtype={'id': str})['id'].tolist()
        existing_ids_set = set(existing_ids)

        for subreddit in subreddits:
            for keyword in keywords:
                try:
                    subreddit_data = collect_data(subreddit, keyword, start_date)
                    df_new = pd.DataFrame([item for item in subreddit_data if item['id'] not in existing_ids_set])
                    if not df_new.empty:
                        df_new.to_csv(csv_path, mode='a', header=False, index=False)
                        existing_ids_set.update(df_new['id'])
                    print(f"Info for keyword '{keyword}' from subreddit '{subreddit}' added to {csv_path}")
                except Exception as e:
                    print(f"Error collecting data for keyword '{keyword}' in subreddit '{subreddit}': {e}")
                    time.sleep(60)  # Adjust sleep time based on API usage

if __name__ == '__main__':
    main()
    
