import praw
import os
import pandas as pd
from datetime import datetime, timezone
import time

reddit = praw.Reddit(
    client_id='vInV29b0TXkkpagkYMoPLQ',
    client_secret='VS-PBH-LXW_sXBbZWJvIKta5XeB6Yw',
    user_agent='adam'
)

def collect_data(subreddit, keyword, item_limit=30):
    """Collect posts and comments from a specific subreddit based on a keyword."""
    data = []
    count = 0

    for submission in reddit.subreddit(subreddit).search(keyword, limit=None):
        if count < item_limit:
            data.append({
                'subreddit': subreddit,
                'type': 'post',
                'keyword': keyword,
                'id': submission.id,
                'author': str(submission.author),
                'title': submission.title,
                'body': submission.selftext,
                'created_utc': datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).isoformat()
            })
            count += 1

        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            if count < item_limit:
                data.append({
                    'subreddit': subreddit,
                    'type': 'comment',
                    'keyword': keyword,
                    'id': comment.id,
                    'author': str(comment.author),
                    'title': '',
                    'body': comment.body,
                    'created_utc': datetime.fromtimestamp(comment.created_utc, tz=timezone.utc).isoformat()
                })
                count += 1
            if count >= item_limit:
                break
        if count >= item_limit:
            break

    return data

def main():
    subreddits = ['ukpolitics', 'PoliticsUK', 'unitedkingdom', 'Scotland', 'Wales', 'northernireland', 'england', 'GreenParty', 'LeftWingUK', 'LabourUK', 'Labour', 'SNP', 'ScottishGreenParty', 'UKGreens', 'plaidcymru', 'RightWingUK', 'tories', 'reformuk', 'brexitpartyuk', 'brexit', 'TaxUK']
    keywords = [
        'Israel', 'Palestine', 'Israel-Palestine', 'Pro-Palestine', 'Pro-Israel',
        'Gaza', 'West Bank', 'Hamas', 'Ceasefire', 'Protest', 'Zionist','Zionism',
        'Antisemitist','Antisemitism', 'Boycott', 'Occupation', 'Annexation',
        'Israel-Palestine War','War', 'Israel-Palestine Conflict','Conflict',
        'Gaza Genocide','Genocide', 'Gaza Strip', 'Palestine Refugees', 'IDF',
        'Israel Defense Forces', 'PLO', 'Palestine Liberation Organization',
    ]

    csv_path = 'Israel-Palestine.csv'
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=['subreddit', 'type', 'keyword', 'id', 'author', 'title', 'body', 'created_utc']).to_csv(csv_path, index=False)

    # Load existing IDs to avoid duplicates
    existing_ids = pd.read_csv(csv_path)['id'].tolist()
    existing_ids_set = set(existing_ids)

    for subreddit in subreddits:
        for keyword in keywords:
            try:
                subreddit_data = collect_data(subreddit, keyword)
                df_new = pd.DataFrame([item for item in subreddit_data if item['id'] not in existing_ids_set])
                if not df_new.empty:
                    df_new.to_csv(csv_path, mode='a', header=False, index=False)
                    existing_ids_set.update(df_new['id'])
                print(f"Info for keyword '{keyword}' from subreddit '{subreddit}' added to {csv_path}")
            except Exception as e:
                print(f"Error collecting data for keyword '{keyword}' in subreddit '{subreddit}': {e}")
                time.sleep(60)  # Simple delay, adjust as needed based on the specific rate limit encountered

if __name__ == '__main__':
    main()
