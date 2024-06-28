import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

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

def resample_column(data, majority_class, minority_class, minority_target, majority_target):
    # If the minority count is already greater than or equal to the target, no need to upsample
    if data.value_counts()[minority_class] >= minority_target:
        minority_data = data[data == minority_class]
    else:
        smote = SMOTE(sampling_strategy={minority_class: minority_target}, random_state=42)
        resampled_data, resampled_labels = smote.fit_resample(data.values.reshape(-1, 1), data)
        minority_data = pd.Series(resampled_data.flatten())[resampled_labels == minority_class]
    
    # Downsample the majority class to the target
    majority_data = data[data == majority_class]
    if len(majority_data) > majority_target:
        majority_data = resample(majority_data, replace=False, n_samples=majority_target, random_state=42)
    
    # Combine the downsampled majority data with the upsampled minority data
    final_resampled_data = pd.concat([majority_data, minority_data])
    
    return final_resampled_data.reset_index(drop=True)

# Apply resampling to each column
resampled_columns = {}
for column, info in majority_minority_info.items():
    resampled_data = resample_column(df_interest[column], 
                                     info['majority_class'], 
                                     info['minority_class'],
                                     120,  # Target for minority class
                                     280)  # Target for majority class
    resampled_columns[column] = resampled_data.reset_index(drop=True)

# Convert the dictionary of resampled columns back to a dataframe
resampled_df = pd.DataFrame(resampled_columns)

# Replace the relevant columns in the original dataframe with the resampled data
for column in resampled_columns.keys():
    df[column] = resampled_df[column]

# Save the modified dataframe to a new CSV file
output_file_path = '/Users/kanushreejaiswal/Desktop/Thesis/Automated Annotation/Balanced Dataset/UK/SMOTERUS/Brexit_SMOTERUSBalanced.csv'
df.to_csv(output_file_path, index=False)

# Verify the resampling results
resampled_counts = resampled_df.apply(pd.Series.value_counts).fillna(0).astype(int)
print(resampled_counts)
