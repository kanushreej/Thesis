import pandas as pd

def split_csv(input_file, output_file1, output_file2):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)
    
    # Calculate the number of rows in each split
    total_rows = len(df)
    split_size = total_rows // 2
    
    # Split the DataFrame into two parts
    df1 = df.iloc[:split_size]
    df2 = df.iloc[split_size:]
    
    # Save each part to a separate CSV file
    df1.to_csv(output_file1, index=False)
    df2.to_csv(output_file2, index=False)
    
    print(f"CSV file successfully split into {output_file1} and {output_file2}")

# Example usage:

input_csv = '/Users/kanushreejaiswal/Desktop/Thesis/Annotation/UK/Labelling data/Final Set/TaxationUK_sample.csv'
output_csv1 = '/Users/kanushreejaiswal/Desktop/Thesis/Annotation/UK/Labelling data/Set 1/TaxationUK_sampleset1.csv'
output_csv2 = '/Users/kanushreejaiswal/Desktop/Thesis/Annotation/UK/Labelling data/Set 2/TaxationUK_sampleset2.csv'

split_csv(input_csv, output_csv1, output_csv2)
