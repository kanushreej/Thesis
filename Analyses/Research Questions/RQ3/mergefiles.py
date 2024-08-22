import pandas as pd

# List of CSV files
csv_files = ['/Users/kanushreejaiswal/Desktop/MORETHESISSS/US_Labelled/ImmigrationUS_labelled.csv', '/Users/kanushreejaiswal/Desktop/MORETHESISSS/US_Labelled/ClimateChangeUS_labelled.csv', '/Users/kanushreejaiswal/Desktop/MORETHESISSS/US_Labelled/HealthcareUS_labelled.csv', '/Users/kanushreejaiswal/Desktop/MORETHESISSS/US_Labelled/IsraelPalestineUS_labelled.csv', '/Users/kanushreejaiswal/Desktop/MORETHESISSS/US_Labelled/TaxationUS_labelled.csv']

# List to store DataFrames
dataframes = []

# Loop through each file
for file in csv_files:
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Retain only 'subreddit', 'type', and 'author' columns
    df = df[['subreddit', 'type', 'author']]
    
    # Append the DataFrame to the list
    dataframes.append(df)

# Concatenate all DataFrames
merged_df = pd.concat(dataframes, ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('/Users/kanushreejaiswal/Desktop/MORETHESISSS/US_Labelled/type_file.csv', index=False)

print("Merged file created successfully.")
