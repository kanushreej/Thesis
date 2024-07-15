import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import numpy as np

# nltk.download('punkt')
# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path, dtype={'parent_id': str, 'body': str, 'title': str, 'id': str})

# Function to clean text (stop before stop word removal)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    return text

# Function to clean and tokenize text
def clean_text(text):
    text = preprocess_text(text)
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    return tokens

# Combine title and body
def combine_title_body(row):
    title = row['title'] if not pd.isna(row['title']) else ''
    body = row['body'] if not pd.isna(row['body']) else ''
    combined_text = f"{title} {body}".strip()
    return clean_text(combined_text)

# Combine title and body without tokenization
def combine_title_body_raw(row):
    title = row['title'] if not pd.isna(row['title']) else ''
    body = row['body'] if not pd.isna(row['body']) else ''
    combined_text = f"{title} {body}".strip()
    return preprocess_text(combined_text)

# Context Handling
def build_thread(data, comment_id):
    thread = []
    thread_raw = []
    try:
        parent_id = data.loc[data['id'] == comment_id, 'parent_id'].values[0]
    except IndexError:
        return thread, thread_raw

    while parent_id:
        if parent_id.startswith('t1_'):  # Comment
            parent_comment = data[data['id'] == parent_id[3:]]
            if not parent_comment.empty:
                parent_comment = parent_comment.iloc[0]
                parent_text = parent_comment['body'] if not pd.isna(parent_comment['body']) else ''
                cleaned_parent_text = clean_text(parent_text)
                raw_parent_text = preprocess_text(parent_text)
                thread = cleaned_parent_text + thread
                thread_raw = [raw_parent_text] + thread_raw
                parent_id = parent_comment['parent_id']
            else:
                break
        elif parent_id.startswith('t3_'):  # Post
            parent_post = data[data['id'] == parent_id[3:]]
            if not parent_post.empty:
                parent_post = parent_post.iloc[0]
                parent_title = parent_post['title'] if not pd.isna(parent_post['title']) else ''
                parent_body = parent_post['body'] if not pd.isna(parent_post['body']) else ''
                cleaned_parent_title = clean_text(parent_title)
                cleaned_parent_body = clean_text(parent_body)
                raw_parent_title = preprocess_text(parent_title)
                raw_parent_body = preprocess_text(parent_body)
                thread = cleaned_parent_title + cleaned_parent_body + thread
                thread_raw = [raw_parent_title + ' ' + raw_parent_body] + thread_raw
            break
        else:
            break
    return thread, ' '.join(thread_raw)

# Function to combine all contextual data for labeled data
def combine_all_contextual_data(directory_path):
    combined_data = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith('_data.csv'):
            file_path = os.path.join(directory_path, file_name)
            data = load_data(file_path)
            data['text'] = data.apply(combine_title_body, axis=1)
            data['text_raw'] = data.apply(combine_title_body_raw, axis=1)
            combined_data.append(data)
    return pd.concat(combined_data, ignore_index=True)

# Train Word2Vec model
def train_word2vec_model(sentences):
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model

# Vectorize text using Word2Vec model
def vectorize_text(model, text):
    vectors = [model.wv[word] for word in text if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Function to preprocess labeled data and add context
def preprocess_labelled_data(data_file, context_directory, output_file):
    # Load labeled data
    data = load_data(data_file)
    
    # Combine title and body
    data['text'] = data.apply(combine_title_body, axis=1)
    data['text_raw'] = data.apply(combine_title_body_raw, axis=1)
    
    # Remove rows where both title and body are empty
    data = data[data['text'].apply(lambda x: len(x) > 0)]
    
    # Combine all contextual data
    context_data = combine_all_contextual_data(context_directory)

    # Collect all sentences for Word2Vec training
    all_sentences = data['text'].tolist() + context_data['text'].tolist()
    
    # Train Word2Vec model
    word2vec_model = train_word2vec_model(all_sentences)

    # Apply context handling
    def build_context(row):
        if row['type'] == 'comment':
            thread, thread_raw = build_thread(context_data, row['id'])
            return thread, thread_raw
        return [], ''

    context_result = data.apply(build_context, axis=1)
    data['context'], data['context_raw'] = zip(*context_result)

    # Vectorize text and context
    data['text_vector'] = data['text'].apply(lambda x: vectorize_text(word2vec_model, x))
    data['context_vector'] = data['context'].apply(lambda x: vectorize_text(word2vec_model, x))

    # Save the processed data to a new file, retaining all columns including id
    data.to_csv(output_file, index=False)

# Function to preprocess unlabeled data and add context within the same issue
def preprocess_unlabelled_data(data_file, output_file):
    # Load unlabeled data
    data = load_data(data_file)
    
    # Combine title and body
    data['text'] = data.apply(combine_title_body, axis=1)
    data['text_raw'] = data.apply(combine_title_body_raw, axis=1)
    
    # Remove rows where both title and body are empty
    data = data[data['text'].apply(lambda x: len(x) > 0)]

    # Use the same data as context
    context_data = data.copy()

    # Collect all sentences for Word2Vec training
    all_sentences = data['text'].tolist()
    
    # Train Word2Vec model
    word2vec_model = train_word2vec_model(all_sentences)

    # Apply context handling
    def build_context(row):
        if row['type'] == 'comment':
            thread, thread_raw = build_thread(context_data, row['id'])
            return thread, thread_raw
        return [], ''

    context_result = data.apply(build_context, axis=1)
    data['context'], data['context_raw'] = zip(*context_result)

    # Vectorize text and context
    data['text_vector'] = data['text'].apply(lambda x: vectorize_text(word2vec_model, x))
    data['context_vector'] = data['context'].apply(lambda x: vectorize_text(word2vec_model, x))

    # Save the processed data to a new file, retaining all columns including id
    data.to_csv(output_file, index=False)

# Preprocess labeled data
#preprocess_labelled_data('//Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Automated Annotation/Training Data/US/TaxationUS_labelled.csv',
#                         '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Collected Data/US/Subreddit Data',
#                         '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Automated Annotation/Training Data/US/TaxationUS_training.csv')

# Preprocess unlabeled data
preprocess_unlabelled_data('/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Collected Data/UK/Subreddit Data/Brexit_data.csv',
                            '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Analyses/Subbreddit Data/UK/Brexit_data_with_context.csv')
