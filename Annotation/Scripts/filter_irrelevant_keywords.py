import pandas as pd

def process_files(file1_path, file2_path):
    # Load both CSV files
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Merge the two dataframes
    merged_df = pd.concat([df1, df2])

    # Filter out the irrelevant keywords and count their occurrences
    irrelevant_keywords = merged_df[merged_df['irrelevant'] == 1]['keyword']
    irrelevant_keywords_count = irrelevant_keywords.value_counts()

    # Display the irrelevant keywords along with their counts
    irrelevant_keywords_count_df = irrelevant_keywords_count.reset_index()
    irrelevant_keywords_count_df.columns = ['keyword', 'count']
    print(irrelevant_keywords_count_df)

# Paths to the CSV files
base_directory = "/Users/kanushreejaiswal/Desktop" # Change this
issue = 'HealthcareUK' # Change this
moderator_name1 = "Kanushree" # Change this
moderator_name2 = "Adam" # Change this

file1_path = f"{base_directory}/Thesis/Annotation/UK/{moderator_name1}/{issue}_labelled.csv"
file2_path = f"{base_directory}/Thesis/Annotation/UK/{moderator_name2}/{issue}_labelled.csv"

# Process the files
process_files(file1_path, file2_path)
