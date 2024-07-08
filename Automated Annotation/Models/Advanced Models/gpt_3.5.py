import openai
from openai import OpenAI
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Set your OpenAI API key
api_key = "sk-proj-2iUr5gCINMNZDYblwYYJT3BlbkFJfo1Vyw5Jj1mrjyyodbrN"

client = OpenAI(api_key=api_key)

# Load the dataset
file_path = '/Users/kanushreejaiswal/Desktop/Thesis/Automated Annotation/Training Data/UK/Brexit_training.csv'
brexit_data = pd.read_csv(file_path)

# Extract features and targets
X = brexit_data[['text_raw', 'context_raw']].fillna('')
y = brexit_data[['pro_brexit', 'anti_brexit', 'neutral', 'irrelevant']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to get predictions from GPT-3.5-turbo
def get_gpt35_turbo_predictions(texts):
    predictions = []
    for index, row in texts.iterrows():
        text = row['text_raw']
        context = row['context_raw']
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that classifies texts into categories."},
                    {"role": "user", "content": f"Make sure you do not add any extra comments, just classify the following text and its context into the categories pro_brexit, anti_brexit, neutral, and irrelevant.\n\nText: {text}\n\nContext: {context}\n\nCategories:\npro_brexit: 0 or 1\nanti_brexit: 0 or 1\nneutral: 0 or 1\nirrelevant: 0 or 1\n\nResponse:"}
                ],
                max_tokens=100,  # Allow enough tokens for a complete response
                temperature=0
            )
            response_text = response.choices[0].message.content.strip()

            # Extracting numerical values from the response text
            lines = response_text.split('\n')
            pred = [0, 0, 0, 0]  # Default to zeros

            for line in lines:
                line = line.strip().lower()  # Convert to lowercase and strip whitespace
                if 'pro_brexit' in line:
                    pred[0] = int(line.split(':')[1].strip())
                elif 'anti_brexit' in line:
                    pred[1] = int(line.split(':')[1].strip())
                elif 'neutral' in line:
                    pred[2] = int(line.split(':')[1].strip())
                elif 'irrelevant' in line:
                    pred[3] = int(line.split(':')[1].strip())

            # Ensure the prediction length is correct
            if len(pred) != 4:
                raise ValueError("Invalid prediction length")
        except Exception as e:
            #print(f"Error processing text: {text[:50]}... with context: {context[:50]}... Error: {e}")
            pred = [0, 0, 0, 0]
        predictions.append(pred)
    return np.array(predictions)

# Get predictions for the test set
y_pred = get_gpt35_turbo_predictions(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro', zero_division=0)
recall = recall_score(y_test, y_pred, average='micro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)

# Print metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

'''import openai
from openai import OpenAI

api_key = "sk-proj-2iUr5gCINMNZDYblwYYJT3BlbkFJfo1Vyw5Jj1mrjyyodbrN"

client = OpenAI(api_key=api_key)

# Replace with your actual API key and Organization ID

response = client.chat.completions.create(model="gpt-3.5-turbo",
messages=[
  {"role": "user", "content": "Say this is a test!"}
],
temperature=0.7)

print(response)'''
