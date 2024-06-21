import pandas as pd

columns_to_evaluate = [
    'pro_brexit', 'anti_brexit', 'pro_climateAction', 'anti_climateAction', 
    'public_healthcare', 'private_healthcare', 'pro_israel', 'pro_palestine', 
    'increase_tax', 'decrease_tax', 'neutral', 'irrelevant'
]

columns_to_check = ['increase_tax', 'decrease_tax','irrelevant','neutral'] #change this

def process_files(file1_path, file2_path):
    # Load both CSV files
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Select rows 60 to 90 (inclusive)
    df1 = df1.iloc[0:121]
    df2 = df2.iloc[0:121]

    # Merge the two dataframes
    merged_df = pd.concat([df1, df2])

    # Filter out the irrelevant keywords and count their occurrences
    for column in columns_to_check:
        keywords = merged_df[merged_df[column] == 1]['keyword']
        keywords_count = keywords.value_counts()
        keywords_count_df = keywords_count.reset_index()
        keywords_count_df.columns = ['keyword', 'count']
        print(column)
        print(keywords_count_df)
        print()

# Paths to the CSV files
base_directory = "/Users/kanushreejaiswal/Desktop"  # Change this
issue = 'TaxationUK'  # Change this
moderator_name1 = "Kanushree"  # Change this
moderator_name2 = "Raphael"  # Change this

file1_path = f"{base_directory}/Thesis/Annotation/UK/{moderator_name1}/{issue}_labelled.csv"
file2_path = f"{base_directory}/Thesis/Annotation/UK/{moderator_name2}/{issue}_labelled.csv"

# Process the files
process_files(file1_path, file2_path)
