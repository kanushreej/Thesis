import pandas as pd
import os

def count_posts_comments(issue, data_dir):
    csv_path = os.path.join(data_dir, f"{issue}_data.csv")
    
    if not os.path.exists(csv_path):
        print(f"No data file found for issue: {issue}")
        return 0, 0

    df = pd.read_csv(csv_path, dtype={'id': str})

    total_posts = df[df['type'] == 'post'].shape[0]
    total_comments = df[df['type'] == 'comment'].shape[0]

    return total_posts, total_comments

# Update issue and local directory up to /Keyword Collection
issue = 'TaxationUK'  
data_dir = '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Subreddit Data/UK'

total_posts, total_comments = count_posts_comments(issue, data_dir)
print(f"Total number of posts: {total_posts}")
print(f"Total number of comments: {total_comments}")
print(f"Total number of data points: {total_comments + total_posts}")
