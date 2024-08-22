import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the data files
type_file_path = '/Users/kanushreejaiswal/Desktop/MORETHESISSS/US_Labelled/type_file.csv'
users_preprocessed_path = '/Users/kanushreejaiswal/Desktop/MORETHESISSS/USERS COMBINED/usersUK_preprocessed.csv'

type_file = pd.read_csv(type_file_path)
users_preprocessed = pd.read_csv(users_preprocessed_path)

# Calculate the number of posts and comments for each user in type_file
post_counts = type_file[type_file['type'] == 'post'].groupby('author').size().reset_index(name='total_posts')
comment_counts = type_file[type_file['type'] == 'comment'].groupby('author').size().reset_index(name='total_comments')

# Merge the counts with the users_preprocessed dataframe
users_preprocessed = users_preprocessed.merge(post_counts, how='left', left_on='username', right_on='author').drop(columns='author')
users_preprocessed = users_preprocessed.merge(comment_counts, how='left', left_on='username', right_on='author').drop(columns='author')

# Fill NaN values with 0
users_preprocessed['total_posts'] = users_preprocessed['total_posts'].fillna(0).astype(int)
users_preprocessed['total_comments'] = users_preprocessed['total_comments'].fillna(0).astype(int)

# Add a new column that is the sum of total posts and total comments
users_preprocessed['total_activity'] = users_preprocessed['total_posts'] + users_preprocessed['total_comments']
users_preprocessed['total_karma'] = users_preprocessed['comment_karma'] + users_preprocessed['post_karma']

# Normalize specified columns
scaler = MinMaxScaler()
columns_to_normalize = ['total_posts', 'total_comments', 'total_activity', 'post_karma', 'comment_karma', 'total_karma']

# Fit the scaler and transform the columns
normalized_values = scaler.fit_transform(users_preprocessed[columns_to_normalize])

# Create a DataFrame with the normalized values
normalized_df = pd.DataFrame(normalized_values, columns=['normalized_' + col for col in columns_to_normalize])

# Concatenate the original DataFrame with the normalized DataFrame
users_preprocessed = pd.concat([users_preprocessed, normalized_df], axis=1)

# Save the updated dataframe to a new CSV file
new_file_path = '/Users/kanushreejaiswal/Desktop/MORETHESISSS/USERS COMBINED/newUKfile.csv'
users_preprocessed.to_csv(new_file_path, index=False)

print(f"Updated file saved to {new_file_path}")
