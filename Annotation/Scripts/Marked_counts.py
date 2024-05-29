import pandas as pd

file_path = 'HealthcareUS_labelled.csv' 
df = pd.read_csv(file_path)

new_columns = [
    'pro_immigration', 'anti_immigration',
    'pro_climateAction', 'anti_climateAction',
    'public_healthcare', 'private_healthcare',
    'pro_israel', 'pro_palestine',
    'increase_tax', 'decrease_tax',
    'neutral', 'irrelevant'
]

marked_counts = df[new_columns].notnull().sum().sum()
print("Total number of marked entries:", marked_counts)