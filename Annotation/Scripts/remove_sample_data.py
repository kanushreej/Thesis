import pandas as pd

# File paths
sample_file_path = '/Users/kanushreejaiswal/Desktop/Thesis/Annotation/UK/Labelling data/TaxationUK_sample.csv'
data_file_path = '/Users/kanushreejaiswal/Desktop/Thesis/Subreddit Data/UK/TaxationUK_data.csv'

# Load the sample CSV file
sample_df = pd.read_csv(sample_file_path)

# Load the other CSV file (Brexit_data)
data_df = pd.read_csv(data_file_path)

# Get the IDs of the first 30 datapoints from the sample CSV file
removed_ids = sample_df.iloc[:120]['id']

# Remove rows in the Brexit_data CSV file that have IDs in the removed_ids list
filtered_data_df = data_df[~data_df['id'].isin(removed_ids)]

# Overwrite the Brexit_data file with the modified dataframe
filtered_data_df.to_csv(data_file_path, index=False)

print(f"Filtered data saved to {data_file_path}")
