import pandas as pd
import os

def create_unified_dataframe(base_directory, issue):

    individual_directory = os.path.join(base_directory, 'Individual')
    ratings_dict = {}
    files_processed = 0  

    for file in os.listdir(individual_directory):
        if issue in file and file.startswith('labelled'):
            moderator_name = file.split('_')[2].split('.')[0]  
            file_path = os.path.join(individual_directory, file)
            try:
                df = pd.read_csv(file_path)
                df.set_index('Keyword', inplace=True)
                ratings_dict[moderator_name] = df['Relevant']  
                files_processed += 1
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    if files_processed == 0:
        raise ValueError("No files were processed. Check the directory path and file format.")

    all_ratings = pd.DataFrame(ratings_dict)

    combined_directory = os.path.join(base_directory, 'Combined')
    if not os.path.exists(combined_directory):
        os.makedirs(combined_directory)  
    
    output_filename = f'all_labelled_{issue}.csv'
    output_path = os.path.join(combined_directory, output_filename)
    all_ratings.to_csv(output_path)

    return all_ratings

base_directory = '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Keyword Selection/ARI/Labelled'  # Base path to the Labelled folder
issue = 'TaxationUK'  # Specify the issue
try:
    unified_df = create_unified_dataframe(base_directory, issue)
    print(f"Unified DataFrame created and saved as 'all_labelled_{issue}.csv' in the 'Combined' folder.")
    print(unified_df.head())
except Exception as error:
    print(f"An error occurred: {error}")
