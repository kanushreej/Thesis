import pandas as pd

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to count stance values
def count_stance_values(data):
    stances = ['pro_brexit', 'anti_brexit', 'pro_climateAction', 'anti_climateAction',
               'pro_NHS', 'anti_NHS', 'pro_israel', 'pro_palestine',
               'pro_company_taxation', 'pro_worker_taxation', 'neutral', 'irrelevant']
    
    stance_counts = {}
    
    for stance in stances:
        stance_counts[stance] = data[stance].value_counts().to_dict()
    
    return stance_counts

# Load labeled data
file_path = '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Automated Annotation/Labelled Data/UK/all_labelled.csv'
data = load_data(file_path)

# Count stance values
stance_counts = count_stance_values(data)

# Print the results
for stance, counts in stance_counts.items():
    print(f"Stance: {stance}")
    for value, count in counts.items():
        print(f"Value {value}: {count}")
    print("\n")
