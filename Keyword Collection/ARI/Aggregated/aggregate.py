import pandas as pd
from glob import glob

def aggregate_keywords(issue, base_dir):
    # Find all relevant CSV files
    csv_paths = glob(f"{base_dir}/Models/**/top_keywords_{issue.lower()}.csv", recursive=True)
    all_keywords = pd.DataFrame()
    
    # Loop through each file path
    for path in csv_paths:
        df = pd.read_csv(path)

        if 'Keyword' in df.columns:
            df = df[['Keyword']]
        else:
            print(f"Warning: 'Keyword' column not found in {path}. Skipping.")
            continue

        all_keywords = pd.concat([all_keywords, df], ignore_index=True)
    
    all_keywords.drop_duplicates(subset='Keyword', keep='first', inplace=True)

    output_path = f"{base_dir}/ARI/Aggregated/{issue}.csv"
    all_keywords.to_csv(output_path, index=False)

    return output_path

# Update issue and local directory
output_file_path = aggregate_keywords('HealthcareUK', '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Keyword Collection')
print(f"Aggregated keywords are saved in: {output_file_path}")
