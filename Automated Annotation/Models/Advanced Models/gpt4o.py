import openai
from openai import OpenAI
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import numpy as np
import ast
import time

# Set your OpenAI API key
api_key = "sk-proj-2iUr5gCINMNZDYblwYYJT3BlbkFJfo1Vyw5Jj1mrjyyodbrN"
client = OpenAI(api_key=api_key)

# Load the dataset
file_path = '/Users/kanushreejaiswal/Desktop/Thesis/Automated Annotation/Training Data/US/ImmigrationUS_training.csv'
brexit_data = pd.read_csv(file_path)

brexit_data_subset = brexit_data.iloc[:300]

# Extract features and targetsq
X = brexit_data_subset[['text_raw']]
y = brexit_data_subset[['pro_immigration', 'anti_immigration', 'neutral', 'irrelevant']]

# Prepare the training examples as a context for GPT-3.5-turbo
def prepare_training_context(X_train, y_train):
    training_context = "You are a helpful assistant that classifies texts into categories.\n\n"
    for text, labels in zip(X_train['text_raw'], y_train.values):
        training_context += f"Text: {text}\nCategories:\npro_immigration: {labels[0]}, anti_immigration: {labels[1]}, neutral: {labels[2]}, irrelevant: {labels[3]}\n\n"
    return training_context

def prepare_prediction_prompt(testing_text, training_context):
    prompt = training_context + "On the basis of the given data, classify the following texts into the categories pro_immigration, anti_immigration, neutral, and irrelevant.\n\n"
    for i, text in enumerate(testing_text):
        prompt += f"Text {i+1}: {text}\n"
    prompt += "\nCategories (provide in the format 'Text: [0 or 1, 0 or 1, 0 or 1, 0 or 1] where each element in the list refers to pro_immigration, anti_immigration, neutral, irrelevant make sure this each list is the only information you provide and do not write anything else'):\n"
    return prompt

# Define a function to get predictions from GPT-4o
def get_gpt4o_turbo_predictions(testing_text, training_context):
    prompt = prepare_prediction_prompt(testing_text, training_context)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that classifies texts into categories."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4096,  # Adjust this based on the length of your texts and expected response
        temperature=0
    )
    response_text = response.choices[0].message.content.strip()
    print(response_text)


    # Parse the response
    lines = response_text.split('\n')
    predictions = []
    for line in lines:
        if line.startswith("Text"):
            list_str = line.split(': ')[1]
            pred = ast.literal_eval(list_str)# Convert the string to a list
            predictions.append(pred)
    return np.array(predictions)


kf = KFold(n_splits=3, shuffle=True, random_state=40)
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    training_context = prepare_training_context(X_train, y_train)
    testing_text = X_test['text_raw'].tolist()

    y_pred = get_gpt4o_turbo_predictions(testing_text, training_context)
    
    y_test = y_test.astype(int)
    y_test_array = y_test[['pro_immigration', 'anti_immigration', 'neutral', 'irrelevant']].to_numpy()
    
    accuracy = accuracy_score(y_test_array, y_pred)
    precision = precision_score(y_test_array, y_pred, average='macro')
    recall = recall_score(y_test_array, y_pred, average='macro')
    f1 = f1_score(y_test_array, y_pred, average='macro')
    
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

print(f"Average Accuracy: {np.mean(accuracy_scores)}")
print(f"Average Precision: {np.mean(precision_scores)}")
print(f"Average Recall: {np.mean(recall_scores)}")
print(f"Average F1 Score: {np.mean(f1_scores)}")
