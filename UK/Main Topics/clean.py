import pandas as pd
import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')  


df = pd.read_csv('/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/UK/keyword collection/ukpolitics.csv')

df['text'] = df['title'].fillna('') + ' ' + df['body'].fillna('')


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Function to clean, stem, and lemmatize text
def clean(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(stemmer.stem(word)) for word in words if word not in stop_words]
    return ' '.join(words)

df['text'] = df['text'].apply(clean)
df.to_csv('/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/UK/keyword collection/cleaned_ukpolitics.csv', index=False)