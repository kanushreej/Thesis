import pandas as pd

# Load the data
df = pd.read_csv('/Users/kanushreejaiswal/Desktop/RQ4/combined_csv_file/usersUS_nr3_updated.csv')

# Print the number of data points before filtering
print(f"Total data points before filtering: {df.shape[0]}")

# Define the columns to filter
columns_to_filter = ['total_activity', 'total_karma']

# Filter out rows where the values are outside of two standard deviations from the mean
for column in columns_to_filter:
    mean = df[column].mean()
    std_dev = df[column].std()
    lower_bound = mean - 4 * std_dev
    upper_bound = mean + 4 * std_dev
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Print the number of data points after filtering
print(f"Total data points after filtering: {df.shape[0]}")

# Save the filtered data
df.to_csv('/Users/kanushreejaiswal/Desktop/RQ4/combined_csv_file/usersUS_nr3_preprocessed.csv', index=False)
