import pandas as pd
from datetime import datetime

# Load the dataset
data = pd.read_csv('Analyses/User Data/Clustered/usersUS_nr3.csv')

# Convert UNIX timestamp to datetime
data['account_creation_date'] = pd.to_datetime(data['account_creation_date'], unit='s')

# Calculate current date
current_date = datetime.now()

# Calculate the number of days since account creation
data['days_since_creation'] = (current_date - data['account_creation_date']).dt.days

# Convert days to years
data['years_since_creation'] = data['days_since_creation'] / 365.25

# Print summary statistics
print("Summary statistics for account creation dates:")
print(data['years_since_creation'].describe())

# Define cohort boundaries correctly
cohort_boundaries = [0, 1, 5, 10, data['years_since_creation'].max()]

# Create cohorts with proper bin labels
cohort_labels = ['<1 year', '1-5 years', '5-10 years', '>10 years']
data['cohort'] = pd.cut(data['years_since_creation'], 
                        bins=cohort_boundaries, 
                        labels=cohort_labels,
                        right=False)

# Print cohort distribution
print("\nCohort distribution:")
print(data['cohort'].value_counts())

# Analyze mean distance to center by cohort
cohort_analysis = data.groupby('cohort')['distance_to_center'].mean()
print("\nMean distance to center by cohort:")
print(cohort_analysis)

# Calculate and print the total average distance to center
total_avg_distance_to_center = data['distance_to_center'].mean()
print("\nTotal average distance to center:")
print(total_avg_distance_to_center)
