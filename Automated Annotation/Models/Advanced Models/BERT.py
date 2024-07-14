import pandas as pd
from sklearn.model_selection import StratifiedKFold
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
import torch
import numpy as np
import os
import random
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# Ensure results directory exists
os.makedirs('./results', exist_ok=True)

# Load Data
labelled_data = pd.read_csv(r"C:\Users\vshap\OneDrive\Desktop\work\code\Thesis\Thesis\Automated Annotation\Training Data\US\TaxationUS_training.csv")

# Extract features and targets
X = labelled_data[['text_raw', 'context_raw']].fillna('')
y = labelled_data[['pro_middle_low_tax', 'pro_wealthy_corpo_tax', 'neutral', 'irrelevant']]

# Define Dataset Class
class StanceDataset(Dataset):
    def __init__(self, texts, contexts, labels, tokenizer, max_length):
        self.texts = texts
        self.contexts = contexts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        context = self.contexts[idx]
        combined_text = text + " " + context

        encoding = self.tokenizer(
            combined_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float).unsqueeze(0)
        return item

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare data for each stance
stances = ['pro_middle_low_tax', 'pro_wealthy_corpo_tax', 'neutral', 'irrelevant']

def compute_metrics(pred):
    labels = pred.label_ids
    preds = (pred.predictions > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Best hyperparameters from Optuna (adjusted for better generalization)
best_params = {
    'learning_rate': 1e-5,
    'num_train_epochs': 5,
    'per_device_train_batch_size': 4,
    'weight_decay': 0.01
}

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

train_loss_per_epoch = []
eval_loss_per_epoch = []

for stance in stances:
    labels = y[stance].values
    texts = X['text_raw'].tolist()
    contexts = X['context_raw'].tolist()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    for train_index, test_index in skf.split(texts, labels):
        train_texts = [texts[i] for i in train_index]
        test_texts = [texts[i] for i in test_index]
        train_contexts = [contexts[i] for i in train_index]
        test_contexts = [contexts[i] for i in test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        train_dataset = StanceDataset(train_texts, train_contexts, train_labels, tokenizer, max_length=256)
        test_dataset = StanceDataset(test_texts, test_contexts, test_labels, tokenizer, max_length=256)

        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=best_params['num_train_epochs'],
            per_device_train_batch_size=best_params['per_device_train_batch_size'],
            per_device_eval_batch_size=16,
            learning_rate=best_params['learning_rate'],
            warmup_steps=500,
            weight_decay=best_params['weight_decay'],
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            fp16=True,  # Enable mixed precision training
            logging_first_step=True,
            save_steps=200,
            eval_steps=200,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        trainer.train()
        eval_results = trainer.evaluate()

        # Collect the training and evaluation loss
        for log in trainer.state.log_history:
            if 'loss' in log:
                train_loss_per_epoch.append(log['loss'])
            if 'eval_loss' in log:
                eval_loss_per_epoch.append(log['eval_loss'])

        predictions = trainer.predict(test_dataset).predictions
        probabilities = torch.sigmoid(torch.tensor(predictions)).numpy()

        acc = accuracy_score(test_labels, (probabilities > 0.5).astype(int))
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels, (probabilities > 0.5).astype(int), average='binary', zero_division=0)

        accuracy_scores.append(acc)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

results = {
    'accuracy': np.mean(accuracy_scores),
    'precision': np.mean(precision_scores),
    'recall': np.mean(recall_scores),
    'f1_score': np.mean(f1_scores)
}

# Print the results
print("Evaluation Results:")
print(f"Accuracy: {results['accuracy']}")
print(f"Precision: {results['precision']}")
print(f"Recall: {results['recall']}")
print(f"F1 Score: {results['f1_score']}")

# Calculate average loss per epoch
num_epochs = best_params['num_train_epochs']
train_loss_per_epoch_avg = np.mean(np.reshape(train_loss_per_epoch, (num_epochs, -1)), axis=1)
eval_loss_per_epoch_avg = np.mean(np.reshape(eval_loss_per_epoch, (num_epochs, -1)), axis=1)

# Plot learning curves
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss_per_epoch_avg, label='Training Loss')
plt.plot(epochs, eval_loss_per_epoch_avg, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.show()
