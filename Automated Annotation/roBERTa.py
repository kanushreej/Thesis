### What does it do? ###
#1. Load Data: Reads the labeled and unlabeled data from CSV files.
#2. Combine Text Columns: Combines the 'title' and 'body' columns into a single 'text' column for better context.
#3. Custom Dataset Class: Defines a custom dataset class RedditDataset to handle tokenization and label preparation.
#4. Metrics Function: Defines a compute_metrics function for evaluation.
#5. Initialize Tokenizer: Initializes the RoBERTa tokenizer.
#6. Split Labeled Data: Splits the labeled data into training and validation sets.
#7. K-Fold Cross-Validation: Performs 10-fold cross-validation to evaluate the model, storing the results.
#8. Print Average Metrics: Calculates and prints the average evaluation metrics from the cross-validation.
#9. Train on Full Dataset: Trains the final model on the entire labeled dataset.
#10. Predict on Unlabeled Data: Uses a random subset of 400 datapoints from the unlabeled data to predict labels and prints the evaluation metrics for this subset

import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load labeled and unlabeled data
base_directory = "C:/Users/rapha/Documents/CS_VU/Thesis"
moderator_name = "Raphael"
issue = "TaxationUK"
team = "UK"

labeled_df = pd.read_csv(f"{base_directory}/Thesis/Annotation/{team}/{moderator_name}/{issue}_labelled.csv")
unlabeled_df = pd.read_csv(f"{base_directory}/Thesis/Subreddit Data/{team}/{issue}_data.csv")

# Combine 'title' and 'body' for context
labeled_df['text'] = labeled_df['title'].fillna('') + ' ' + labeled_df['body'].fillna('')
unlabeled_df['text'] = unlabeled_df['title'].fillna('') + ' ' + unlabeled_df['body'].fillna('')

# Define your labels
label_columns = [
            'pro_brexit', 'anti_brexit', 'pro_climateAction', 'anti_climateAction',
            'pro_NHS', 'anti_NHS', 'pro_israel', 'pro_palestine',
            'pro_company_taxation', 'pro_worker_taxation', 'neutral', 'irrelevant'
        ]

class RedditDataset(Dataset):
    def __init__(self, df, tokenizer, label_columns, max_len=512):
        self.df = df
        self.tokenizer = tokenizer
        self.label_columns = label_columns
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text = self.df.iloc[index]['text']
        labels = self.df.iloc[index][self.label_columns].values.astype(int)
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions > 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Initialize the tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_length=512)

# Split labeled data
train_df, val_df = train_test_split(labeled_df, test_size=0.1, random_state=42, stratify=labeled_df[label_columns])

# K-Fold Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

results = []

for fold, (train_index, val_index) in enumerate(kf.split(labeled_df), 1):
    print(f"Processing fold {fold}...")
    train_df = labeled_df.iloc[train_index]
    val_df = labeled_df.iloc[val_index]
    
    train_dataset = RedditDataset(train_df, tokenizer, label_columns)
    val_dataset = RedditDataset(val_df, tokenizer, label_columns)
    
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(label_columns))
    
    training_args = TrainingArguments(
    output_dir=f'./results_fold_{fold}',
    num_train_epochs=3,                        # Set to 3-5 based on initial experiments
    per_device_train_batch_size=8,             # Start with 8 and adjust if needed
    per_device_eval_batch_size=16,             # Set to 16-32 based on memory availability
    warmup_steps=500,                          # Typical warmup steps
    weight_decay=0.01,                         # Regularization
    learning_rate=3e-5,                        # Set between 2e-5 and 5e-5 based on experiments
    gradient_accumulation_steps=2,             # Use 2 to simulate larger batch size if needed
    logging_dir=f'./logs_fold_{fold}',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    eval_result = trainer.evaluate()
    results.append(eval_result)

# Calculate and print average metrics
avg_results = {
    'accuracy': np.mean([result['eval_accuracy'] for result in results]),
    'f1': np.mean([result['eval_f1'] for result in results]),
    'precision': np.mean([result['eval_precision'] for result in results]),
    'recall': np.mean([result['eval_recall'] for result in results])
}

print("Average Results from K-Fold Cross-Validation:")
print(f"Accuracy: {avg_results['accuracy']:.4f}")
print(f"F1 Score: {avg_results['f1']:.4f}")
print(f"Precision: {avg_results['precision']:.4f}")
print(f"Recall: {avg_results['recall']:.4f}")

# Train on full labeled dataset
print("Training on the full labeled dataset...")
full_train_dataset = RedditDataset(labeled_df, tokenizer, label_columns)

model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(label_columns))

training_args = TrainingArguments(
    output_dir=f'./results_fold_{fold}',
    num_train_epochs=3,                        
    per_device_train_batch_size=8,            
    per_device_eval_batch_size=16,           
    warmup_steps=500,                          
    weight_decay=0.01,                      
    learning_rate=3e-5,                       
    gradient_accumulation_steps=2,            
    logging_dir=f'./logs_fold_{fold}',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=full_train_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# Use only a subset of the unlabeled data
print("Predicting on a subset of the unlabeled data...")
unlabeled_subset = unlabeled_df.sample(n=400, random_state=42)
unlabeled_dataset = RedditDataset(unlabeled_subset, tokenizer, label_columns)

# Predict labels for the subset of unlabeled data
predictions = trainer.predict(unlabeled_dataset)

# Apply threshold to predictions
pred_labels = (predictions.predictions > 0.5).astype(int)

# Calculate and print metrics for the unlabeled subset
precision, recall, f1, _ = precision_recall_fscore_support(unlabeled_subset[label_columns].values, pred_labels, average='macro')
accuracy = accuracy_score(unlabeled_subset[label_columns].values, pred_labels)

print("\nEvaluation Metrics on Unlabeled Data Subset:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
