import pandas as pd
import os

def label_keywords(issue, moderator, base_dir):

    aggregated_path = f"{base_dir}/Aggregated/aggregated_keywords_{issue}.csv"
    labeled_path = f"{base_dir}/Labelled/labelled_{issue}_{moderator}.csv"
    
    if os.path.exists(labeled_path):
        labeled_keywords = pd.read_csv(labeled_path)
        start_index = labeled_keywords['Relevant'].isna().idxmax() if labeled_keywords['Relevant'].isna().any() else len(labeled_keywords)
    else:
        all_keywords = pd.read_csv(aggregated_path)
        labeled_keywords = pd.DataFrame({
            'Keyword': all_keywords['Keyword'],
            'Relevant': pd.NA
        })
        start_index = 0  

    for index, row in labeled_keywords.iloc[start_index:].iterrows():
        while True:
            response = input(f"Is this keyword relevant to {issue}: \n{row['Keyword']}\nEnter 'y' for yes, 'n' for no, and 'q' to quit: ").strip().lower()
            if response == 'y':
                labeled_keywords.at[index, 'Relevant'] = 1
                break
            elif response == 'n':
                labeled_keywords.at[index, 'Relevant'] = 0
                break
            elif response == 'q':
                labeled_keywords.to_csv(labeled_path, index=False)
                print("Progress saved. Exiting...")
                return  # Exit 
            else:
                print("Invalid input. Please enter 'y' for yes, 'n' for no, or 'q' to quit.")
    
    labeled_keywords.to_csv(labeled_path, index=False)
    print("All keywords have been labeled and saved.")

base_directory = '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Keyword Selection/ARI'  # Update this to your actual directory path up to /ARI
label_keywords('ClimateChangeUS', 'Adam', base_directory)
