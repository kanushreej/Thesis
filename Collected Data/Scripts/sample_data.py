import pandas as pd

# Load the CSV file
file_path = "/Users/kanushreejaiswal/Desktop/Thesis/Subreddit Data/UK/TaxationUK_data.csv"
data = pd.read_csv(file_path)

# Sample 400 random datapoints
sampled_data = data.sample(n=400, random_state=42)

# Save the sampled data to a new CSV file
output_file_path =  "/Users/kanushreejaiswal/Desktop/Thesis/Annotation/UK/Labelling data/Final Set/TaxationUK_sample.csv"
sampled_data.to_csv(output_file_path, index=False)

print(f'Sampled data saved to {output_file_path}')

