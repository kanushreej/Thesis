import pandas as pd

# Load the dataset
file_path = '/Users/kanushreejaiswal/Desktop/Thesis/Automated Annotation/Labelled Data/UK/all_labelled.csv'
data = pd.read_csv(file_path)

# Define the relevant columns for counting 0s and 1s
relevant_columns = [
    'pro_brexit', 'anti_brexit', 'pro_climateAction', 'anti_climateAction', 
    'pro_NHS', 'anti_NHS', 'pro_israel', 'pro_palestine', 
    'pro_company_taxation', 'pro_worker_taxation', 'neutral', 'irrelevant'
]

# Function to count 0s and 1s in each relevant column
def count_zeros_ones(df, columns):
    counts = {}
    for col in columns:
        counts[col] = {
            '0s': (df[col] == 0).sum(),
            '1s': (df[col] == 1).sum()
        }
    return counts

# Get the counts of 0s and 1s
counts = count_zeros_ones(data, relevant_columns)

# Print the counts
for column, count in counts.items():
    print(f"{column}:")
    print(f"  0s: {count['0s']}")
    print(f"  1s: {count['1s']}")
    print("ratio:",count['0s']/count['1s'] )