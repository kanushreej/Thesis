import pandas as pd
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = '/Users/kanushreejaiswal/Desktop/Thesis/Automated Annotation/Labelled Data/UK/Brexit_labelled.csv'
df = pd.read_csv(file_path)

# Filter the columns of interest
columns_of_interest = ['pro_brexit', 'anti_brexit', 'neutral', 'irrelevant']
df_interest = df[columns_of_interest]

# Convert to integer type (if they are not already)
df_interest = df_interest.fillna(0).astype(int)

# Calculate counts of 0s and 1s for each column
counts = df_interest.apply(pd.Series.value_counts).fillna(0).astype(int)

# Identify majority and minority classes
majority_minority_info = {}
for column in columns_of_interest:
    count_0 = counts.at[0, column]
    count_1 = counts.at[1, column]
    majority_class = 0 if count_0 > count_1 else 1
    minority_class = 1 if majority_class == 0 else 0
    majority_count = max(count_0, count_1)
    minority_count = min(count_0, count_1)
    ratio = majority_count / minority_count if minority_count != 0 else 'Infinity'
    
    majority_minority_info[column] = {
        'majority_class': majority_class,
        'minority_class': minority_class,
        'majority_count': majority_count,
        'minority_count': minority_count,
        'ratio': ratio
    }

def resample_column_smote(data, majority_class, minority_class):
    smote = SMOTE(sampling_strategy=0.3, random_state=42)
    resampled_data, _ = smote.fit_resample(data.values.reshape(-1, 1), data)
    return pd.Series(resampled_data.flatten())

# Apply resampling to each column using SMOTE
resampled_columns = {}
for column, info in majority_minority_info.items():
    if info['ratio'] > (70/30):
        resampled_data = resample_column_smote(df_interest[column], 
                                               info['majority_class'], 
                                               info['minority_class'])
        resampled_columns[column] = resampled_data
        
        # Debugging: Print the counts after resampling
        print(f"Column: {column}")
        print(resampled_data.value_counts())

# Verify the length of resampled data
for column in resampled_columns.keys():
    print(f"Original length of '{column}': {len(df[column])}")
    print(f"Resampled length of '{column}': {len(resampled_columns[column])}")

# Adjust the length of the original dataframe to match resampled data if necessary
for column in resampled_columns.keys():
    resampled_len = len(resampled_columns[column])
    if len(df) < resampled_len:
        df = df.reindex(range(resampled_len))
    df[column] = resampled_columns[column].reset_index(drop=True)

# Save the modified dataframe to a new CSV file
output_file_path = '/Users/kanushreejaiswal/Desktop/Thesis/Automated Annotation/Balanced Dataset/UK/SMOTE/Brexit_SMOTEBalanced.csv'
df.to_csv(output_file_path, index=False)

# Verify the resampling results by reloading the saved file
df_reloaded = pd.read_csv(output_file_path)
reloaded_counts = df_reloaded[resampled_columns.keys()].apply(pd.Series.value_counts).fillna(0).astype(int)
print("Reloaded Data Counts:")
print(reloaded_counts)
