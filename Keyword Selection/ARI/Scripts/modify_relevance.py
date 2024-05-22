import pandas as pd
import os

def modify_relevance(issue, moderator, base_dir):
    dir = f"{base_dir}/ARI/Labelled/Individual/labelled_{issue}_{moderator}.csv"
    if not os.path.exists(dir):
        print("No labeled data found for this moderator.")
        return
    
    data = pd.read_csv(dir)
    
    print("Enter 'done' when you finish modifications.")
    while True:
        keyword = input("Enter the keyword to modify: ").strip()
        if keyword == 'done':
            break
        if keyword in data['Keyword'].values:
            index = data[data['Keyword'] == keyword].index[0]
            new_value = input("Enter the new relevance (1 for relevant, 0 for not relevant): ")
            data.at[index, 'Relevant'] = int(new_value)
        else:
            print("Keyword not found.")
    
    data.to_csv(dir, index=False)
    print("Modifications saved.")

# Update issue and local directory up to /Keyword Collection
modify_relevance('Brexit', 'Adam', '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Keyword Selection')
