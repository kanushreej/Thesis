import pandas as pd
import random

# Load the CSV file
file_path = '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Subreddit Data/UK/Brexit_data.csv'  # Change this to your file path
df = pd.read_csv(file_path)

# Filter the dataframe for only "post" type
posts_df = df[df['type'] == 'post']

# Randomly select 30 datapoints
random_sample = posts_df.sample(n=30, random_state=1)

# Save the new dataframe to a new CSV file
new_file_path = '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Annotation/UK/Labelling data/Brexit_sample.csv'  # Change this to your desired output file path
random_sample.to_csv(new_file_path, index=False)
issue = ["ClimateChangeUS"]

print(f"New CSV file created: {new_file_path}")
