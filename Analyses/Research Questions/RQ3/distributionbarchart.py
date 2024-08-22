import matplotlib.pyplot as plt
import pandas as pd

# Data
data = {
    'Cohort': ['0-20', '20-40', '40-60', '60-80', '80-100'],
    'Average Distance': [14.2, 13.77, 13.7, 13.73, 13.68],
    'Distribution': [169585, 4104, 1468, 592, 242]
}

new_data = {
    'Cohort': ['0-10', '10-20', '20-30', '30-40'],
    'Average Distance': [10.43, 10.34, 10.38, 10.48],
    'Distribution': [161076, 14907, 4833, 1596]
}

# Create DataFrame
df = pd.DataFrame(new_data)

# Plotting
fig, ax = plt.subplots()

# Bar plot for Distribution
ax.set_xlabel('Cohort')
ax.set_ylabel('Distribution')
ax.bar(df['Cohort'], df['Distribution'], color='green')
ax.tick_params(axis='y')

# Remove grid
ax.grid(False)

# Title and layout
plt.title('Distribution Across US Cohorts')
fig.tight_layout()  
plt.show()
