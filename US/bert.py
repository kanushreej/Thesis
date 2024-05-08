import pandas as pd
from keybert import KeyBERT
from collections import defaultdict

def load_data(filepath):
    """Load the CSV file into a DataFrame."""
    return pd.read_csv(filepath)

def extract_keywords(texts, num_keywords=5):
    """Extract keywords from texts."""
    kw_model = KeyBERT()
    keywords = [kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', use_maxsum=True, nr_candidates=20, top_n=num_keywords) for text in texts]
    return [item for sublist in keywords for item in sublist]

def main():
    filepath = r'path_to_your_csv_file' #change it to ur own csv path
    df = load_data(filepath)

    df['body'] = df['body'].fillna('')

    all_keywords = extract_keywords(df['body'].tolist(), num_keywords=5)

    keyword_scores = defaultdict(float)
    for keyword, score in all_keywords:
        keyword_scores[keyword] += score

    top_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:50] # change the number so that it displays the first n data points

    print(top_keywords)

    pd.DataFrame(top_keywords, columns=['Keyword', 'Score']).to_csv('top_keywords.csv', index=False)

if __name__ == "__main__":
    main()
