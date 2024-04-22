import praw
import os
import pandas as pd
from datetime import datetime, timezone

reddit = praw.Reddit(
    client_id='wEP3RUiICi5YvjDKhEKlkg',
    client_secret='FP8O4SUc6ocAGNgCx5HB1-nczz6uQw',
    user_agent='script:keyword_extractor:v1.0 (by u/Queasy-Parsnip-8103)'
)

def collect_data(subreddit, keyword):
    """ Collect posts and comments from a specific subreddit based on a keyword. """
    data = []
    
    for submission in reddit.subreddit(subreddit).search(keyword, limit=None):
        data.append({
            'type': 'post',
            'keyword': keyword,
            'id': submission.id,
            'author': str(submission.author),
            'title': submission.title,
            'body': submission.selftext,
            'created_utc': datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).isoformat()
        })
        
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            data.append({
                'type': 'comment',
                'keyword': keyword,
                'id': comment.id,
                'author': str(comment.author),
                'title': '',
                'body': comment.body,
                'created_utc': datetime.fromtimestamp(comment.created_utc, tz=timezone.utc).isoformat()
            })
            
    return data

def main():
    subreddits = ['immigrationlaw', 'law']  # List of subreddits
    keywords = ['immigration', 'legal']  # Corresponding keywords

    for subreddit, keyword in zip(subreddits, keywords):
        subreddit_data = collect_data(subreddit, keyword)
        
        if os.path.exists('reddit_data.csv'):
            df_existing = pd.read_csv('reddit_data.csv')
            df_new = pd.DataFrame(subreddit_data)
            df = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df = pd.DataFrame(subreddit_data)
        
        df.to_csv('reddit_data.csv', index=False)
        print(f"info from subreddit '{subreddit}' added to reddit_data.csv")

if __name__ == '__main__':
    main()
