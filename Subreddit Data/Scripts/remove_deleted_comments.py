import pandas as pd

def filter_removed_deleted(input_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_file)
    
    # Function to filter rows containing '[removed]' or '[deleted]'
    def filter_body(text):
        if isinstance(text, str):  # Check if 'text' is a string
            return not ('[removed]' in text.lower() or '[deleted]' in text.lower())
        else:
            return True  # Keep rows where 'body' is not a string (handle NaN)
    
    # Apply the filter function to 'body' column
    df = df[df['body'].apply(filter_body)]
    
    # Overwrite the original file with the filtered DataFrame
    df.to_csv(input_file, index=False)
    
    print(f"Filtered CSV saved to {input_file}")

# Example usage:
input_file = '/Users/kanushreejaiswal/Desktop/Thesis/Annotation/UK/Labelling data/HealthcareUK_sample.csv'
filter_removed_deleted(input_file)
