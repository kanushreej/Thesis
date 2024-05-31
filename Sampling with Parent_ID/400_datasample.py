import pandas as pd
import random

# Load the provided CSV file
input_file_path = '/Users/kanushreejaiswal/Desktop/IsraelPalestineUS_data.csv' 
data = pd.read_csv(input_file_path)

# Define a function to randomly collect data points and add them to another CSV file
def randomly_collect_data(input_df, output_file_path, num_samples=400):
    # Randomly sample data points from the input dataframe
    sampled_data = input_df.sample(n=num_samples)
    
    # Append the sampled data to the output CSV file
    sampled_data.to_csv(output_file_path, mode='a', index=False, header=not pd.io.common.file_exists(output_file_path))

# Specify the output file path
output_file_path = '/Users/kanushreejaiswal/Desktop/IsraelPalestineUS_400data.csv' 

# Call the function to randomly collect data points and add them to the output CSV file
randomly_collect_data(data, output_file_path, num_samples=400)

# Display the first few rows of the new output file to verify the result
output_data = pd.read_csv(output_file_path)
print("Data added")

