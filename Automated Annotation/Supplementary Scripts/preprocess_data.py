import pandas as pd
import os

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path, dtype={'parent_id': str, 'body': str, 'title': str, 'id': str})

# Combine title and body
def combine_title_body(row):
    title = row['title'] if not pd.isna(row['title']) else ''
    body = row['body'] if not pd.isna(row['body']) else ''
    return f"{title} {body}".strip()

# Context Handling
def build_thread(data, comment_id):
    thread = ""
    try:
        parent_id = data.loc[data['id'] == comment_id, 'parent_id'].values[0]
    except IndexError:
        return thread

    while parent_id:
        if parent_id.startswith('t1_'):  # Comment
            parent_comment = data[data['id'] == parent_id[3:]]
            if not parent_comment.empty:
                parent_comment = parent_comment.iloc[0]
                parent_text = parent_comment['body'] if not pd.isna(parent_comment['body']) else ''
                thread = f"{parent_text}\n\n" + thread
                parent_id = parent_comment['parent_id']
            else:
                break
        elif parent_id.startswith('t3_'):  # Post
            parent_post = data[data['id'] == parent_id[3:]]
            if not parent_post.empty:
                parent_post = parent_post.iloc[0]
                parent_title = parent_post['title'] if not pd.isna(parent_post['title']) else ''
                parent_body = parent_post['body'] if not pd.isna(parent_post['body']) else ''
                thread = f"{parent_title}\n\n{parent_body}\n\n" + thread
            break
        else:
            break
    return thread

# Function to combine all contextual data for labeled data
def combine_all_contextual_data(directory_path):
    combined_data = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith('_data.csv'):
            file_path = os.path.join(directory_path, file_name)
            data = load_data(file_path)
            data['text'] = data.apply(combine_title_body, axis=1)
            combined_data.append(data)
    return pd.concat(combined_data, ignore_index=True)

# Function to preprocess labeled data and add context
def preprocess_labelled_data(data_file, context_directory, output_file):
    # Load labeled data
    data = load_data(data_file)
    
    # Combine title and body
    data['text'] = data.apply(combine_title_body, axis=1)
    
    # Remove rows where both title and body are empty
    data = data[data['text'].str.strip().astype(bool)]
    
    # Combine all contextual data
    context_data = combine_all_contextual_data(context_directory)

    # Apply context handling
    data['context'] = data.apply(lambda x: build_thread(context_data, x['id']) + x['text'] if x['type'] == 'comment' else x['text'], axis=1)

    # Save the processed data to a new file, retaining all columns including id
    data.to_csv(output_file, index=False)

# Function to preprocess unlabeled data and add context within the same issue
def preprocess_unlabelled_data(data_file, output_file):
    # Load unlabeled data
    data = load_data(data_file)
    
    # Combine title and body
    data['text'] = data.apply(combine_title_body, axis=1)
    
    # Remove rows where both title and body are empty
    data = data[data['text'].str.strip().astype(bool)]

    # Use the same data as context
    context_data = data.copy()

    # Apply context handling
    data['context'] = data.apply(lambda x: build_thread(context_data, x['id']) + x['text'] if x['type'] == 'comment' else x['text'], axis=1)

    # Save the processed data to a new file, retaining all columns including id
    data.to_csv(output_file, index=False)

# Preprocess labeled data
preprocess_labelled_data('/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Automated Annotation/Labelled Data/UK/Brexit_labelled.csv',
                         '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Collected Data/UK/Subreddit Data',
                         '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Automated Annotation/Labelled Data/UK/Brexit_labelled_with_context.csv')

# Preprocess unlabeled data
#preprocess_unlabelled_data('/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Collected Data/UK/Subreddit Data/Brexit_data.csv',
#                           '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Collected Data/UK/Subreddit Data/Brexit_data_with_context.csv')
