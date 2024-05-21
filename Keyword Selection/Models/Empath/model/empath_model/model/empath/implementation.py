import pandas as pd
from empath import Empath
from collections import Counter
import csv
import io
import contextlib

# Create an instance of the Empath lexicon
lexicon = Empath()

# Function to capture the printed output from create_category
def capture_create_category_output(lexicon, name, words, model, size):
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        lexicon.create_category(name, words, model=model, size=size)
        output = buf.getvalue()
    # Parse the output into a list
    category = output.strip().strip('[]').replace('"', '').split(', ')
    return category

# Create a new category related to "Israel-Palestine" using the "reddit" model
new_category = capture_create_category_output(lexicon, "Brexit", ["Brexit"], model="reddit", size=500)

print("New Category Created:", new_category)

# Read the CSV file
input_csv = "/Users/kanushreejaiswal/Desktop/empathmodel/empath/Brexit.csv" # Replace with the actual input CSV file path
df = pd.read_csv(input_csv)

# Initialize a counter to store the words and their frequencies
word_counter = Counter()

# Define a function to count the words in the text based on the new category
def count_words_in_category(text, category):
    words = text.split()
    return [word for word in words if word in category]

# Analyze each text in the "body" column and update the word counter
for text in df["body"].dropna():
    words = count_words_in_category(text, new_category)
    word_counter.update(words)

# Get the top 100 keywords based on their frequencies
top_keywords = word_counter.most_common(100)

# Write the top keywords to a new CSV file
output_csv = "/Users/kanushreejaiswal/Desktop/empathmodel/empath/top_keywords1.csv"  # Replace with the desired output CSV file path
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Keyword", "Frequency"])
    writer.writerows(top_keywords)

print(f"Top 100 keywords have been written to {output_csv}")
