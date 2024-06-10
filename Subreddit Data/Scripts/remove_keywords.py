import pandas as pd

def remove_rows_by_keywords(file_path, keywords):
    # Load the CSV file
    data = pd.read_csv(file_path)
    
    # Convert keywords to a set for faster lookup
    keyword_set = set(keywords)
    
    # Filter out rows where the keyword is in the set of keywords to be removed
    filtered_data = data[~data['keyword'].isin(keyword_set)]
    
    # Overwrite the original CSV file with the filtered data
    filtered_data.to_csv(file_path, index=False)

# Example usage
file_path = '/Users/kanushreejaiswal/Desktop/Thesis/cleaned data/UK/TaxationUK_data.csv'
keywords_to_remove = ['pension', 'pensions']  # Replace with your keywords
remove_rows_by_keywords(file_path, keywords_to_remove)
