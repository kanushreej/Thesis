import pandas as pd
import os

def identify_keywords_for_consensus(base_directory, issue, threshold=0.5):

    combined_file_path = os.path.join(base_directory, 'Labelled/Combined', f'all_labelled_{issue}.csv')
    
    df = pd.read_csv(combined_file_path, index_col='Keyword')
    df['StdDev'] = df.std(axis=1)
    discrepant_keywords = df[df['StdDev'] >= threshold]
    
    discrepant_file_path = os.path.join(base_directory, 'Discrepent', f'discrepant_{issue}.csv')
    
    discrepant_keywords.to_csv(discrepant_file_path)
    
    print(f"Discrepant keywords for {issue} saved to: {discrepant_file_path}")
    return discrepant_keywords

base_directory = '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Keyword Selection/ARI'  
issue = 'TaxationUK'  
identify_keywords_for_consensus(base_directory, issue)
