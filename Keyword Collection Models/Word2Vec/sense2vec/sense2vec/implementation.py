import pandas as pd
from sense2vec import Sense2Vec
from collections import Counter
import csv

# Load the sense2vec model
model_path = "/Users/kanushreejaiswal/Desktop/sense2vec/s2v_old"  # Ensure this path is correct
s2v = Sense2Vec().from_disk(model_path)

# Function to get similar words using sense2vec for multiple POS tags
def get_similar_words(word, s2v, pos_tags, n=250):
    similar_words = set()
    for pos in pos_tags:
        query = word + '|' + pos
        try:
            words = s2v.most_similar(query, n=n)
            similar_words.update([w[0].split('|')[0] for w in words])
        except ValueError:
            print(f"Skipping {query} as it is not found in the model.")
    return list(similar_words)

# List of parts of speech to consider
pos_tags = ["NOUN", "VERB", "ADJ", "ADV"]

# Get similar words for "Brexit" across different parts of speech
similar_words = get_similar_words("Tax", s2v, pos_tags)

print("Similar Words Retrieved:", similar_words)

# Read the CSV file
input_csv = "/Users/kanushreejaiswal/Desktop/sense2vec/csvfiles/TaxationUS.csv"  # Ensure this path is correct
df = pd.read_csv(input_csv)

# Initialize a counter to store the words and their frequencies
word_counter = Counter()

# Define a function to count the words in the text based on the similar words
def count_words_in_similar(text, similar_words):
    words = text.split()
    return [word for word in words if word in similar_words]

# Analyze each text in the "body" column and update the word counter
for text in df["body"].dropna():
    words = count_words_in_similar(text, similar_words)
    word_counter.update(words)

# Get the top 50 keywords based on their frequencies
top_keywords = word_counter.most_common(50)

# Write the top keywords to a new CSV file
output_csv = "/Users/kanushreejaiswal/Desktop/sense2vec/keywords/sense2vec_TaxationUS_keywords.csv"  # Ensure this path is correct
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Keyword", "Frequency"])
    writer.writerows(top_keywords)

print(f"Top 50 keywords have been written to {output_csv}")
