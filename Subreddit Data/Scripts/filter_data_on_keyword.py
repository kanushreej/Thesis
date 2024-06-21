import pandas as pd
import os

def remove_old_keywords_data(issue, base_dir, data_dir):
    """Remove data sourced from keywords not in the keyword file and deduplicate based on id."""
    csv_path = os.path.join(data_dir, f"{issue}_data.csv")
    keyword_file = os.path.join(base_dir, f"{issue}_final_keywords.csv")
    current_keywords = pd.read_csv(keyword_file)['Keyword'].tolist()

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, dtype={'id': str})
        df = df.drop_duplicates(subset='id')

        df_filtered = df[df['keyword'].isin(current_keywords)]
        df_filtered.to_csv(csv_path, index=False)
        print(f"Data for issue '{issue}' has been updated to only include current keywords and deduplicated.")
    else:
        print(f"No data found for issue '{issue}' at {csv_path}")

def main():
    base_dir = "/Users/kanushreejaiswal/Desktop/Thesis/Keyword Selection/Final"
    data_dir = "/Users/kanushreejaiswal/Desktop/Thesis/Subreddit Data/UK"
    issues = ['TaxationUK']

    for issue in issues:
        remove_old_keywords_data(issue, base_dir, data_dir)

if __name__ == '__main__':
    main()
