import pandas as pd

# Load the data from the CSV file
file_path = '/Users/kanushreejaiswal/Desktop/RQ3/combined_csv_file/usersUK_nr8_preprocessed.csv'
data = pd.read_csv(file_path)

# Extract the relevant columns for analysis
activity_data = data[['user_id', 'total_karma', 'distance_to_center']]

# Define the cohorts
cohorts = [(0, 50000), (50000, 100000), (100000, 150000), (150000, 200000),(200000, 250000), (250000, 300000), (300000, 350000)]

# Function to assign cohorts
def assign_cohort(activity):
    for cohort in cohorts:
        if cohort[0] <= activity < cohort[1]:
            return f'{cohort[0]}-{cohort[1]}'
    return '100+'  # For any activity >= 100

# Assign cohorts to each user
activity_data['cohort'] = activity_data['total_karma'].apply(assign_cohort)

# Calculate average distance_to_center and distribution in each cohort
cohort_stats = activity_data.groupby('cohort')['distance_to_center'].agg(['mean', 'count']).reset_index()
cohort_stats.columns = ['Cohort', 'Average Distance to Cluster', 'Count']

# Print the cohort statistics
for index, row in cohort_stats.iterrows():
    print(f"Cohort: {row['Cohort']}")
    print(f"  Average Distance to Cluster: {row['Average Distance to Cluster']:.2f}")
    print(f"  Distribution (Count): {row['Count']}")
    print()

# If needed, display the dataframe for visualization
