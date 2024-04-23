import praw
import os
import pandas as pd
from datetime import datetime, timezone

reddit = praw.Reddit(
    client_id='vInV29b0TXkkpagkYMoPLQ',
    client_secret='VS-PBH-LXW_sXBbZWJvIKta5XeB6Yw',
    user_agent='adam'
)

def collect_data(subreddit, keyword):
    """Collect posts and comments from a specific subreddit based on a keyword."""
    data = []

    for submission in reddit.subreddit(subreddit).search(keyword, limit=None):
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
        
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
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

    return data

def main():
    subreddits = ['ukpolitics', 'PoliticsUK','unitedkingdom','Scotland','Wales', 'northernireland', 'england','GreenParty','LeftWingUK','LabourUK','Labour','SNP','ScottishGreenParty','UKGreens','plaidcymru','RightWingUK','tories','reformuk','brexitpartyuk','brexit','TaxUK']  # List of subreddits
    keywords = [
    'Israel', 'Palestine', 'Israel-Palestine', 'Pro-Palestine', 'Pro-Israel',
    'Gaza', 'West Bank', 'Hamas', 'Ceasefire', 'Protest', 'Zionist/Zionism',
    'Antisemitist/Antisemitism', 'Boycott', 'Occupation', 'Annexation',
    'Israel-Palestine War/War', 'Israel-Palestine Conflict/Conflict',
    'Gaza Genocide/Genocide', 'Gaza Strip', 'Palestine Refugees', 'IDF',
    'Israel Defense Forces', 'PLO', 'Palestine Liberation Organization'
    ]
  # List of keywords

    if not os.path.exists('reddit_data.csv'):
        pd.DataFrame(columns=['subreddit', 'type', 'keyword', 'id', 'author', 'title', 'body', 'created_utc']).to_csv('reddit_data.csv', index=False)

    for subreddit in subreddits:
        for keyword in keywords:
            subreddit_data = collect_data(subreddit, keyword)
            
            df_existing = pd.read_csv('reddit_data.csv')
            df_new = pd.DataFrame(subreddit_data)
            df = pd.concat([df_existing, df_new], ignore_index=True)
            
            df.to_csv('reddit_data.csv', index=False)
            print(f"Info for keyword '{keyword}' from subreddit '{subreddit}' added to reddit_data.csv")

if __name__ == '__main__':
    main()
