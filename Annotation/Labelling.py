import json
import os
import pandas as pd
import shutil


original_file = '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Subreddit Data/UK/Israel-Palestine/Israel-Palestine.csv'  # PLEASE CHANGE FILENAME IF NEEDED
copy_file = '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Annotation/UK/Adam/Israel-Palestine.csv' # PLEASE CHANGE FILENAME IF NEEDEd
progress_file = 'progress.json'

if not os.path.exists(copy_file):
    shutil.copy(original_file, copy_file)
    print("File copied successfully.")

df = pd.read_csv(copy_file)
pd.set_option('display.max_colwidth', None)

new_columns = [
    'pro_israel', 'pro_palestine',  
    'public_healthcare', 'private_healthcare',  
    'high_tax', 'low_tax',  
    'neutral', 'irrelevant'  
]

for column in new_columns:
    if column not in df.columns:
        df[column] = float('nan')

# Sorting data by subreddit and then keyword 
df.sort_values(by=['subreddit', 'keyword', 'created_utc'], inplace=True)
df.reset_index(drop=True, inplace=True)

df.to_csv(copy_file, index=False)

def load_progress():
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)["last_index"]
    return 0

last_index = load_progress()
start_index = last_index if last_index < len(df) else 0

def save_progress(last_index):
    with open(progress_file, 'w') as f:
        json.dump({"last_index": last_index}, f)


try:
    unique_keywords = df['keyword'].unique()
    max_keyword_index = len(unique_keywords) - 1
    keyword_index = 0
    
    while True:
        while True:
            row = df.iloc[start_index]
            if row['keyword'] == unique_keywords[keyword_index]:
                break
            start_index = (start_index + 1) % len(df)
        
        print(f"\nData Point: {start_index}")
        print(f"subreddit: {row['subreddit']}")
        print(f"type: {row['type']}")
        print(f"keyword: {row['keyword']}")
        print(f"id: {row['id']}")
        print(f"author: {row['author']}")
        print(f"title: {row['title']}")
        print(f"body: {row['body']}")
        print(f"created_utc: {row['created_utc']}")
        
        for column in new_columns:
            input_value = input(f"Value for {column} or 'q' to quit: ")
            if input_value.lower() == 'q':
                save_progress(start_index)
                print("Progress saved, exiting.")
                exit(0)
            try:
                new_value = int(input_value)
                df.at[start_index, column] = new_value
            except ValueError:
                print("Please enter a valid integer.")
        
        df.to_csv(copy_file, index=False)
        
        keyword_index = (keyword_index + 1) % (max_keyword_index + 1)
        start_index = (start_index + 1) % len(df)
        
except KeyboardInterrupt:
    save_progress(start_index)
    print("Progress saved, exiting due to interrupt.")
