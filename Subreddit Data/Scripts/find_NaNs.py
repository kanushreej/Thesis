import pandas as pd
import os

original_file = '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Subreddit Data/US/TaxationUS_data.csv'  # PLEASE CHANGE FILENAME IF NEEDED

def prompt_overwrite(file_path):
    while True:
        response = input(f"Do you want to overwrite the original file {file_path}? (yes/no): ").strip().lower()
        if response in ['yes', 'no']:
            return response == 'yes'
        else:
            print("Please respond with 'yes' or 'no'.")

if os.path.exists(original_file):
    df = pd.read_csv(original_file, dtype=str)
    print(f"Loaded data with {len(df)} records.")
    
    print("Initial data head from the original file:")
    print(df.head())

    print("Number of NaN values in each column before conversion:")
    print(df.isna().sum())

    df.dropna(subset=['subreddit', 'keyword', 'created_utc', 'author'], inplace=True)
    print(f"Data after dropping rows with NaNs in crucial columns: {len(df)} records.")
    
    print("Number of NaN values in each column after dropping rows:")
    print(df.isna().sum())
    
    df.sort_values(by=['subreddit', 'keyword', 'created_utc'], inplace=True, na_position='last')
    df.reset_index(drop=True, inplace=True)

    if prompt_overwrite(original_file):
        df.to_csv(original_file, index=False)
        print(f"Cleaned and sorted data saved back to {original_file}.")
    else:
        print("Operation cancelled. The original file was not overwritten.")
else:
    raise FileNotFoundError(f"The original file {original_file} does not exist.")
