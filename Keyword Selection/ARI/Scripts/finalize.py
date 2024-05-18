import pandas as pd
import os

def compile_final_keywords(issue, base_dir):
    labeled_dir = f"{base_dir}/ARI/Labelled"
    file_paths = [os.path.join(labeled_dir, f) for f in os.listdir(labeled_dir) if issue in f]
    data_frames = [pd.read_csv(file) for file in file_paths]
    
    combined = pd.concat(data_frames)
    grouped = combined.groupby('Keyword')['Relevant'].sum()
    final_keywords = grouped[grouped >= 3].index.tolist()
    
    final_path = f"{base_dir}/Final/{issue}_final_keywords.csv"
    pd.DataFrame(final_keywords, columns=['Keyword']).to_csv(final_path, index=False)
    print(f"Final keywords saved to {final_path}")

# Update issue and local directory up to /Keyword Collection
compile_final_keywords('HealthcareUK', '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Keyword Collection')
