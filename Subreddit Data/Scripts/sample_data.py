import pandas as pd
import numpy as np

def sample_data(input_file, output_file):
    # Define the column names based on the provided structure
    column_names = [
        'subreddit', 'type', 'keyword', 'id', 'author', 
        'title', 'body', 'created_utc'
    ]

    # Load the data with specified column names
    df = pd.read_csv(input_file, names=column_names)

    # Group by 'keyword' and 'subreddit'
    grouped = df.groupby(['keyword', 'subreddit'])

    # Apply the sampling function, excluding grouping columns
    sampled_df = grouped.apply(lambda x: x.sample(frac=0.001, random_state=1, ignore_index=True))

    # Reset index to flatten the DataFrame
    sampled_df = sampled_df.reset_index(drop=True)

    # Save the sampled data to a new CSV file
    sampled_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = r'C:\Users\vshap\OneDrive\Desktop\work\code\python\Thesis\Thesis\Subreddit Data\UK\IsraelPalestineUK_data.csv'  # Replace with your input file path
    output_file = r'C:\Users\vshap\OneDrive\Desktop\work\code\python\Thesis\Thesis\Subreddit Data\Sample Data\UK\IsraelPalestineUK_data_0.1%.csv'  # Replace with your output file path
    sample_data(input_file, output_file)
