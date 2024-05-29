import pandas as pd
from glob import glob
import string

def clean_keyword(keyword):
    return keyword.translate(str.maketrans('', '', string.punctuation)).lower()

def aggregate_keywords(issue, base_dir):

    csv_paths = glob(f"{base_dir}/Models/**/top_keywords_{issue.lower()}.csv", recursive=True)     # Find all relevant CSV files
    all_keywords = pd.DataFrame()

    for path in csv_paths:
        df = pd.read_csv(path)

        if 'Keyword' in df.columns:
            df['Cleaned_Keyword'] = df['Keyword'].apply(clean_keyword)
            df.drop_duplicates(subset='Cleaned_Keyword', keep='first', inplace=True)
            df = df.head(20)
            df = df[['Keyword']]
        else:
            print(f"Warning: 'Keyword' column not found in {path}. Skipping.")
            continue

        all_keywords = pd.concat([all_keywords, df], ignore_index=True)
    
    all_keywords['Cleaned_Keyword'] = all_keywords['Keyword'].apply(clean_keyword)
    all_keywords.drop_duplicates(subset='Cleaned_Keyword', keep='first', inplace=True)
    all_keywords = all_keywords[['Keyword']]

    output_path = f"{base_dir}/ARI/Aggregated/aggregated_keywords_{issue}.csv"
    all_keywords.to_csv(output_path, index=False)

    return output_path

# Update issue and local directory up to /Keyword Collection
output_file_path = aggregate_keywords('TaxationUK', '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Keyword Selection')
print(f"Aggregated keywords are saved in: {output_file_path}")
