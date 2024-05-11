import pandas as pd
import spacy
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import umap
import hdbscan

# Load data
df = pd.read_csv('/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/UK/keyword collection/ukpolitics.csv')

# Concatenate title and body columns
df['text'] = df['title'] + ' ' + df['body']
df['text'] = df['text'].fillna('')

# Clean the text: remove URLs, special characters, and normalize text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

df['text'] = df['text'].apply(clean_text)

# Load spaCy transformer model
nlp = spacy.load("en_core_web_trf")

# Initialize and fit the TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['text'])
tfidf_lookup = {key: vectorizer.idf_[value] for key, value in vectorizer.vocabulary_.items()}
vect = []

# Processing text with spaCy and applying TF-IDF weights
for doc in tqdm(nlp.pipe(df['text'], batch_size=10)):  # Adjust batch_size 
    weighted_doc_tensor = []
    for token in doc:
        word_text = token.text.lower()
        weight = tfidf_lookup.get(word_text, 0.5)  # Default weight if word not found
        weighted_doc_tensor.append(token.vector * weight if token.has_vector else np.zeros((768,)))
    if weighted_doc_tensor:  # Check if the document has content after processing
        vect.append(np.mean(weighted_doc_tensor, axis=0))
    else:
        vect.append(np.zeros((768,)))  # Ensure all vectors have the same dimension

vect = np.vstack(vect)

# Dimensionality Reduction
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine') 
embedding = reducer.fit_transform(vect)

# Clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom')
labels = clusterer.fit_predict(embedding)

# Attach clusters back to the DataFrame
df['cluster'] = labels

# Extracting key words for each cluster
cluster_keywords = {}
for cluster in set(labels):
    cluster_texts = df[df['cluster'] == cluster]['text'].values
    vectorized_texts = vectorizer.transform(cluster_texts)
    summed_vectors = np.sum(vectorized_texts, axis=0)
    terms = vectorizer.get_feature_names_out()
    sorted_terms = np.argsort(summed_vectors).flatten()[::-1]
    top_terms = [terms[idx] for idx in sorted_terms[:10]]
    cluster_keywords[cluster] = top_terms

print(cluster_keywords)
