import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def classify_issue(issue):
    stance_groups = {
        'immigration': ['pro_immigration', 'anti_immigration'],
        'climateAction': ['pro_climateAction', 'anti_climateAction'],
        'NHS': ['public_healthcare', 'private_healthcare'],
        'israel_palestine': ['pro_israel', 'pro_palestine'],
        'taxation': ['pro_middle_low_tax', 'pro_wealthy_corpo_tax']
    }

    if issue not in stance_groups:
        raise ValueError(f"Unknown issue: {issue}")

    targets = stance_groups[issue] + ['neutral', 'irrelevant']

    file_path = '/content/drive/MyDrive/colab_use/ImmigrationUS_training.csv'
    df = pd.read_csv(file_path)

    # Load GPT-Neo-2.7B model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

    # Add padding token
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))

    # Combine text and context
    df['text_combined'] = df['text_raw'].astype(str) + " " + df['context_raw'].astype(str)

    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df[targets].idxmax(axis=1))

    def generate_embeddings(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean of logits as embeddings
            embeddings = outputs.logits.mean(dim=1).detach().cpu().numpy()
        return embeddings

    # Generate text embeddings
    X = np.vstack(df['text_combined'].apply(lambda x: generate_embeddings(x)[0]))

    y = df['label'].values

    # Split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Using a more complex MLP classifier
    class SimpleClassifier(torch.nn.Module):
        def __init__(self, input_dim, num_classes):
            super(SimpleClassifier, self).__init__()
            self.fc1 = torch.nn.Linear(input_dim, 512)
            self.bn1 = torch.nn.BatchNorm1d(512)
            self.relu1 = torch.nn.ReLU()
            self.dropout1 = torch.nn.Dropout(0.3)
            self.fc2 = torch.nn.Linear(512, 256)
            self.bn2 = torch.nn.BatchNorm1d(256)
            self.relu2 = torch.nn.ReLU()
            self.dropout2 = torch.nn.Dropout(0.3)
            self.fc3 = torch.nn.Linear(256, num_classes)

        def forward(self, x):
            x = self.fc1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.dropout2(x)
            x = self.fc3(x)
            return x

    # Parameter settings
    input_dim = X_train.shape[1]
    num_classes = len(le.classes_)
    lr = 1e-3  # Adjust learning rate
    epochs = 50  # Increase number of epochs
    batch_size = 16

    # Data conversion to Tensor
    train_data = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_data = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Model, loss function and optimizer
    classifier = SimpleClassifier(input_dim, num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)  # Use L2 regularization
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce learning rate every 10 epochs

    # Train the model
    classifier.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = classifier(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()  # Update learning rate
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    # Evaluate the model
    classifier.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = classifier(batch_X)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(batch_y.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

classify_issue('immigration')
