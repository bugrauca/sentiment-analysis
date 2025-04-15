import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=100):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])
        tokens = tokens[:self.max_length]  # Truncate if too long
        tokens = [0] * (self.max_length - len(tokens)) + tokens  # Pad if too short

        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return tokens_tensor, label_tensor

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers=2, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        hidden_state = lstm_out[:, -1, :]
        output = self.fc(hidden_state)
        return output

def train_model(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    true_labels, predictions = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    #return true_labels, predictions
    accuracy = accuracy_score(true_labels, predictions)
    return total_loss / len(dataloader), accuracy, true_labels, predictions

if __name__ == "__main__":
    processed_data_path = "data/processed/preprocessed.csv"
    vectorizer_path = "models/vectorizer.pkl"
    model_save_path = "models/lstm_model.pt"
    metric_save_path = "results/metrics/"

    print("Loading vectorizer...")
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    print("Loading preprocessed data...")
    data = pd.read_csv(processed_data_path)
    data.dropna(subset=["text"], inplace=True)

    tokenizer_fn = vectorizer.build_analyzer()

    def tokenizer(text):
        tokens = tokenizer_fn(text)
        return [vectorizer.vocabulary_.get(token, 0) for token in tokens]

    print("Splitting data into train and test sets...")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data["text"], data["sentiment"], test_size=0.2, random_state=42
    )

    train_dataset = SentimentDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
    test_dataset = SentimentDataset(test_texts.tolist(), test_labels.tolist(), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    vocab_size = len(vectorizer.vocabulary_)
    embed_dim = 100
    hidden_dim = 256
    output_dim = 3

    print("Initializing model...")
    model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, output_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"Using device: {DEVICE}")

    print("Training model...")

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(10):
        train_loss = train_model(model, train_loader, optimizer, criterion)
        val_loss, val_accuracy, _, _ = evaluate_model(model, test_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")

    print("Evaluating model...")
    _, test_accuracy, true_labels, predictions = evaluate_model(model, test_loader, criterion)

    print("Saving model and metrics...")
    #os.makedirs("models", exist_ok=True)
    os.makedirs(metric_save_path, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    torch.save({'train_losses': train_losses,'val_losses': val_losses,'val_accuracies': val_accuracies}, metric_save_path + "lstm_training_history.pth")
    torch.save(true_labels, metric_save_path + "lstm_true_labels.pth")
    torch.save(predictions, metric_save_path + "lstm_predictions.pth")
    print(f"Metrics saved to {metric_save_path}")

    unique_classes = sorted(set(true_labels))
    num_classes = len(unique_classes)

    # Generate target names based on the number of classes
    default_target_names = ["Class " + str(i) for i in range(num_classes)]
    custom_target_names = ["Negative", "Neutral", "Positive"][:num_classes]  # Adjust to match detected classes

    print("\nAccuracy:", accuracy_score(true_labels, predictions))
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=custom_target_names))

    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predictions))


########## Evaluation ##########


# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=100):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])
        tokens = tokens[:self.max_length]
        tokens = [0] * (self.max_length - len(tokens)) + tokens

        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return tokens_tensor, label_tensor


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers=1, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        hidden_state = lstm_out[:, -1, :]
        output = self.fc(hidden_state)
        return output


def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    true_labels, predictions = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(true_labels, predictions)
    return total_loss / len(dataloader), accuracy, true_labels, predictions


if __name__ == "__main__":
    model_save_path="models/lstm_model.pt",
    processed_data_path="data/processed/preprocessed.csv",
    vectorizer_path="models/vectorizer.pkl",
    
    print("Loading vectorizer...")
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    print("Loading preprocessed data...")
    data = pd.read_csv(processed_data_path)
    data.dropna(subset=["text"], inplace=True)

    tokenizer_fn = vectorizer.build_analyzer()

    def tokenizer(text):
        tokens = tokenizer_fn(text)
        return [vectorizer.vocabulary_.get(token, 0) for token in tokens]

    print("Preparing test dataset...")
    test_texts, test_labels = data["text"].tolist(), data["sentiment"].tolist()
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=64)

    vocab_size = len(vectorizer.vocabulary_)
    embed_dim = 100
    hidden_dim = 256
    output_dim = 3

    print("Loading model...")
    model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, output_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_save_path))

    print("Evaluating model...")
    criterion = nn.CrossEntropyLoss()
    _, test_accuracy, true_labels, predictions = evaluate_model(model, test_loader, criterion)

    unique_classes = sorted(set(true_labels))
    num_classes = len(unique_classes)

    # Generate target names based on the number of classes
    default_target_names = ["Class " + str(i) for i in range(num_classes)]
    custom_target_names = ["Negative", "Neutral", "Positive"][:num_classes]  # Adjust to match detected classes

    print("\nAccuracy:", accuracy_score(true_labels, predictions))
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=custom_target_names))

    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predictions, labels=unique_classes))