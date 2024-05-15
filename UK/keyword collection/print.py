import pandas as pd

# Load the CSV file
data = pd.read_csv('/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/cluster_keywords_top50.csv', header=None)

# Function to extract unique keywords from the string list
def extract_unique_keywords(keyword_string, num_keywords=10):
    # Remove extra characters and split the string into a list
    keywords_list = keyword_string.strip("[]").replace("'", "").split()
    unique_keywords = []
    seen = set()
    
    # Collect the first 'num_keywords' unique keywords
    for keyword in keywords_list:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)
            if len(unique_keywords) == num_keywords:
                break
    
    return unique_keywords

# Apply the function to each row and print results
for index, row in data.iterrows():
    unique_keywords = extract_unique_keywords(row[1])
    print(f"Cluster {row[0]}: {unique_keywords}")
