import pandas as pd
import os

original_file = '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Subreddit Data/UK/Brexit_data.csv'  # PLEASE CHANGE FILENAME IF NEEDED


if os.path.exists(original_file):
    df = pd.read_csv(original_file, dtype=str)
    print(f"Loaded data with {len(df)} records.")
    
    print("Initial data head from the original file:")
    print(df.head())

    print("Number of NaN values in each column before conversion:")
    print(df.isna().sum())

    df['created_utc'] = pd.to_datetime(df['created_utc'], errors='coerce')
    print("Converted created_utc to datetime:")
    print(df.head())

    print("Number of NaN values in each column after conversion:")
    print(df.isna().sum())

    df.dropna(subset=['subreddit', 'keyword', 'created_utc'], inplace=True)
    print(f"Data after dropping rows with NaNs in crucial columns: {len(df)} records.")
    
    print("Number of NaN values in each column after dropping rows:")
    print(df.isna().sum())
    
    df.sort_values(by=['subreddit', 'keyword', 'created_utc'], inplace=True, na_position='last')
    print("Data sorted:")
    print(df.head())
    
    df.reset_index(drop=True, inplace=True)
    print("Data index reset:")
    print(df.head())
    
    df.to_csv(original_file, index=False)
    print(f"Cleaned and sorted data saved back to {original_file}.")
else:
    raise FileNotFoundError(f"The original file {original_file} does not exist.")
