import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import umap
import hdbscan
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Load and preprocess data
df = pd.read_csv('/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/UK/keyword collection/ukpolitics.csv')
df['text'] = df['title'].fillna('') + ' ' + df['body'].fillna('')
df['text'] = df['text'].str.replace(r'http\S+', '', regex=True)
df['text'] = df['text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
df['text'] = df['text'].fillna('')

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', min_df=0.01, max_df=0.7, ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(df['text'].values)
tfidf_lookup = {word: vectorizer.idf_[i] for word, i in vectorizer.vocabulary_.items()}

# Function to get BERT embeddings
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

# Lemmatizer
lemmatizer = WordNetLemmatizer()

# Process text with BERT and apply TF-IDF weights
vect = []
for text in tqdm(df['text'].values):
    lemmatized_text = ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text.lower())])
    embedding = get_bert_embeddings(lemmatized_text)
    words = tokenizer.tokenize(lemmatized_text)
    weighted_embedding = np.zeros(embedding.shape)  # Initialize with zeros to ensure same shape
    for word in words:
        weight = tfidf_lookup.get(word, 0.5)
        weighted_embedding += weight * embedding
    vect.append(weighted_embedding)

# Ensure all embeddings have the same shape before stacking
vect = np.vstack(vect)

# Dimensionality Reduction with UMAP
reducer = umap.UMAP(n_neighbors=25, metric='cosine')
embedding = reducer.fit_transform(vect)

# Clustering with HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=40, min_samples=1)
labels = clusterer.fit_predict(embedding)
df['cluster'] = labels

# Extract and print clusters and their keywords
cluster_keywords = {}
for cluster in set(labels):
    cluster_texts = df[df['cluster'] == cluster]['text'].values
    vectorized_texts = vectorizer.transform(cluster_texts)
    summed_vectors = np.sum(vectorized_texts, axis=0)
    terms = vectorizer.get_feature_names_out()
    sorted_terms = np.argsort(summed_vectors).flatten()[::-1]
    top_terms = [str(terms[idx]) for idx in sorted_terms[:50]]
    cluster_keywords[cluster] = top_terms

# Save cluster keywords with top 50 terms to a CSV file
cluster_df = pd.DataFrame.from_dict(cluster_keywords, orient='index')
cluster_df.to_csv('lemamtized_keywords.csv', header=False)

# Print each cluster's top 10 keywords
for cluster, keywords in cluster_keywords.items():
    print(f"Cluster {cluster}: {', '.join(keywords[:10])}")
