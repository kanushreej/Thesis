import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data file
file_path = '/Users/kanushreejaiswal/Desktop/RQ3/combined_csv_file/usersUS_nr3_preprocessed.csv'
data = pd.read_csv(file_path)

# Calculate the coefficient matrix for normalized columns
normalized_columns_of_interest = ['total_activity','total_karma','distance_to_center']
normalized_coefficient_matrix = data[normalized_columns_of_interest].corr()

# Define new names for the labels
new_names = {
    'total_activity': 'Total Activity',
    'total_karma': 'Total Karma',
    'distance_to_center': 'Distance to Centre'
}

# Rename the labels in the coefficient matrix
normalized_coefficient_matrix = normalized_coefficient_matrix.rename(index=new_names, columns=new_names)

# Display the coefficient matrix using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(normalized_coefficient_matrix, annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"size": 20})
plt.title('Pearsons Coefficient Matrix for US')

# Rotate the tick labels for better readability
plt.xticks(rotation=25, ha='right')
plt.yticks(rotation=0, ha='right')

plt.show()

# Print the coefficient matrix to the console
print("Coefficient Matrix for Normalized Total Posts, Comments, and Activity:")
print(normalized_coefficient_matrix)


'''normalized_columns_of_interest = ['total_posts', 'total_comments', 'total_activity','post_karma','comment_karma','total_karma','distance_to_center']
normalized_coefficient_matrix = data[normalized_columns_of_interest].corr()

# Define new names for the labels
new_names = {
    'total_posts': 'Total Posts',
    'total_comments': 'Total Comments',
    'total_activity': 'Posts + Comments',
    'post_karma': 'Post Karma',
    'comment_karma': 'Comment Karma',
    'total_karma': 'Post + Comment Karma',
    'distance_to_center': 'Distance to Center'
}'''