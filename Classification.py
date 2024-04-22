import os
import pandas as pd
import shutil

original_file = 'reddit_data.csv' #NAMES!!!
copy_file = 'reddit_data_copy.csv'#NAMES!!!

if not os.path.exists(copy_file):
    shutil.copy(original_file, copy_file)
    print("File copied successfully.")

pd.set_option('display.max_colwidth', None)

df = pd.read_csv(copy_file)

new_columns = ['brexit', 'antibrexit', 'school shootings'] #collumns
for column in new_columns:
    if column not in df.columns:
        df[column] = float('nan')

df.to_csv(copy_file, index=False)

df = pd.read_csv(copy_file)

start_index = int(input(f"Enter the starting data point (0 to {len(df) - 1}): "))

start_index = max(0, min(start_index, len(df) - 1))

for index in range(start_index, start_index + 5):
    if index >= len(df):
        break 
    row = df.iloc[index]
    
    print(f"\nData Point: {index}")
    print(f"type: {row['type']}")
    print(f"keyword: {row['keyword']}")
    print(f"id: {row['id']}")
    print(f"author: {row['author']}")
    print(f"\ntitle: {row['title']}")
    print(f"\nbody: {row['body']}")
    print(f"\ncreated_utc: {row['created_utc']}")
    
    for column in new_columns:
        while True:
            input_value = input(f"Value for {column}: ")
            try:
                new_value = int(input_value)
                df.at[index, column] = new_value
                break
            except ValueError:
                print("Please enter a valid integer.")

    
    df.to_csv(copy_file, index=False)

print("All updates saved.")
