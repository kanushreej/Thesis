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
                'author': str(submission.author) if submission.author else None,
                'title': submission.title,
                'body': submission.selftext,
                'created_utc': created_date.isoformat()
            })

            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list():
                if not comment.author or not subreddit or not keyword:
                    continue
                data.append({
                    'subreddit': subreddit,
                    'type': 'comment',
                    'keyword': keyword,
                    'id': str(comment.id),
                    'author': str(comment.author) if comment.author else None,
                    'title': '',
                    'body': comment.body,
                    'created_utc': datetime.fromtimestamp(comment.created_utc, tz=timezone.utc).isoformat()
                })
        return data
    except praw.exceptions.APIException as e:
        print(f"API Exception for {subreddit} with keyword {keyword}: {e}")
        time.sleep(120)  # Sleep longer to respect Reddit's rate limit
        return []
    except Exception as e:
        print(f"General exception for {subreddit} with keyword {keyword}: {e}")
        return []

def verify_and_collect_data(subreddits, issues, base_dir, data_dir, start_date):
    for issue in issues:
        csv_path = os.path.join(data_dir, f"{issue}_data.csv")
        keyword_file = os.path.join(base_dir, f"{issue}_final_keywords.csv")
        keywords = pd.read_csv(keyword_file)['Keyword'].tolist()

        existing_data = pd.read_csv(csv_path, dtype={'id': str}) if os.path.exists(csv_path) else pd.DataFrame()
        existing_combinations = set(zip(existing_data['subreddit'], existing_data['keyword']))

        needed_combinations = {(subreddit, keyword) for subreddit in subreddits for keyword in keywords}
        missing_combinations = needed_combinations - existing_combinations

        for subreddit, keyword in missing_combinations:
            print(f"Collecting missing data for subreddit: {subreddit}, keyword: {keyword}")
            data = collect_data(subreddit, keyword, start_date)
            if data:
                df_new = pd.DataFrame(data)
                df_new.to_csv(csv_path, mode='a', header=False, index=False)
                print(f"Added missing data for {subreddit} - {keyword} to {csv_path}")

def main():
    start_year = 2010
    start_date = datetime(start_year, 1, 1, tzinfo=timezone.utc)

    subreddits = [ 'Politics', 'Conservative', 'Geopolitics', 'Libertarian', 'Democrats',
    'PoliticalDiscussion', 'GreenParty', 'Republican', 'Neoliberal', 'Liberal',
    'Conservatives', 'Climatechange', 'Healthcare', 'Progressive', 'USpolitics'
    ]

    issues = ['ImmigrationUS']
    base_dir = "/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Keyword Selection/Final"
    data_dir = "/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Subreddit Data/US"

    verify_and_collect_data(subreddits, issues, base_dir, data_dir, start_date)

if __name__ == '__main__':
    main()
