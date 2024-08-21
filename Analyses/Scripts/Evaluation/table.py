import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = r'C:\Users\vshap\OneDrive\Desktop\work\code\Thesis\Thesis\Analyses\User Data\Clustered Data\UK\cluster_analysis_nr8.csv'
df = pd.read_csv(file_path)

# Extracting the highest opinion for each cluster
highest_opinion_df = df[['cluster','cluster_size', 'nearest_cluster']]

highest_opinion_df.rename(columns={'cluster': 'Cluster','cluster_size': 'Cluster Size', 'nearest_cluster': 'Nearest Cluster'}, inplace=True)

highest_opinion_df.sort_values(by='Cluster', inplace=True)
# Setting up the plot
fig, ax = plt.subplots(figsize=(15, 10))

# Creating the table
table_data = highest_opinion_df.values
columns = highest_opinion_df.columns

# Plotting the table
table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')

# Enhancing the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.5, 1.5)  # Adjusted scaling for more space in cells

# Setting table colors
for (i, j), cell in table.get_celld().items():
    cell.set_edgecolor('black')
    cell.set_height(0.1)  # Increase cell height for more space
    cell.set_width(0.15)  # Increase cell width for more space
    if i == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#40466e')
    else:
        cell.set_facecolor('#f1f1f2')

# Adjusting the layout
ax.axis('off')

plt.title('Highest Opinion per Cluster', fontsize=16, weight='bold', color='#40466e')
plt.show()
