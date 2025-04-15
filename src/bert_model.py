import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_scheduler
from sklearn.metrics import accuracy_score, classification_report
import torch.nn as nn
from torch.cuda.amp import autocast

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Dataset Class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)}
    
model_save_path = "/models/bert_model.pt"
output_save_path = "/results/metrics"
os.makedirs(output_save_path, exist_ok=True)

# Load Preprocessed Data
processed_data_path = "data/processed/preprocessed.csv"
print("Loading preprocessed data...")
data = pd.read_csv(processed_data_path)

# Verify dataset class distribution
print("Class distribution in the dataset:")
print(data["sentiment"].value_counts())

# Limit dataset size for testing
subset_size = 100000
data = data.sample(subset_size, random_state=42)

# Drop rows with missing values
data.dropna(subset=["text"], inplace=True)

# Prepare tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Prepare datasets and dataloaders
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data["text"], data["sentiment"], test_size=0.2, random_state=42
)

train_dataset = SentimentDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
test_dataset = SentimentDataset(test_texts.tolist(), test_labels.tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, num_workers=4, pin_memory=True)


print("Initializing BERT model...")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3).to(DEVICE)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
num_training_steps = len(train_loader) * 3  # Adjust for 3 epochs
warmup_steps = int(0.1 * num_training_steps)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

# Loss function
criterion = nn.CrossEntropyLoss()

# Mixed Precision Training
scaler = torch.GradScaler()

# Gradient Accumulation Steps
accumulation_steps = 2  # Simulates batch size 32

def train_model(model, dataloader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    optimizer.zero_grad()

    for i, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        with autocast():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / accumulation_steps  # Divide loss for accumulation

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * accumulation_steps  # Scale back for correct total
        progress_bar.set_postfix(loss=loss.item() * accumulation_steps)
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    predictions_list = []
    true_labels_list = []

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            predictions = torch.argmax(outputs.logits, dim=-1)
            predictions_list.extend(predictions.cpu().numpy())
            true_labels_list.extend(labels.cpu().numpy())

            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            progress_bar.set_postfix(loss=loss.item(), accuracy=correct / total)

    accuracy = correct / total
    return total_loss / len(dataloader), accuracy, predictions_list, true_labels_list

# Training loop with checkpoint saving
train_losses = []
val_losses = []
val_accuracies = []
best_accuracy = 0

print("Training model...")
for epoch in range(3):  # Train for 3 epochs
    print(f"Epoch {epoch + 1}")
    train_loss = train_model(model, train_loader, optimizer, scheduler, criterion)
    val_loss, val_accuracy, predictions, true_labels = evaluate_model(model, test_loader, criterion)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at epoch {epoch + 1}")

print("Evaluating model...")
model.load_state_dict(torch.load(model_save_path))
model.eval()

val_loss, val_accuracy, predictions, true_labels = evaluate_model(model, test_loader, criterion)

print("Saving outputs for visualization...")
outputs = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "val_accuracies": val_accuracies,
    "predictions": predictions,
    "true_labels": true_labels
}
torch.save(outputs, os.path.join(output_save_path, "bert_visualization_outputs.pth"))

accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy: {accuracy:.4f}")

# Adjust dynamically based on unique labels
unique_labels = sorted(list(set(true_labels + predictions)))

# Define the target names dynamically
target_names_full = ["Negative", "Neutral", "Positive"]
target_names = [target_names_full[i] for i in unique_labels]

print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=target_names))




######### Hyperparameters #########

# # Define hyperparameter search space
# search_space = {
#     "learning_rate": [2e-5, 3e-5, 5e-5],
#     "batch_size": [16, 32],
#     "num_epochs": [2, 3]
# }

# def hyperparameter_tuning(search_space, train_loader, val_loader):
#     """
#     Perform hyperparameter tuning to find the best configuration.
#     Args:
#         search_space (dict): Dictionary containing hyperparameter lists.
#         train_loader (DataLoader): DataLoader for training data.
#         val_loader (DataLoader): DataLoader for validation data.

#     Returns:
#         dict: Best hyperparameters and associated accuracy.
#     """
#     best_config = None
#     best_accuracy = 0.0

#     # Generate all combinations of hyperparameters
#     combinations = list(itertools.product(
#         search_space["learning_rate"], 
#         search_space["batch_size"], 
#         search_space["num_epochs"]
#     ))

#     for lr, batch_size, num_epochs in tqdm(combinations, desc="Tuning Hyperparameters"):
#         print(f"\nTesting configuration: LR={lr}, Batch Size={batch_size}, Epochs={num_epochs}")
        
#         # Update DataLoader with current batch size
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size)

#         # Initialize model, optimizer, and scheduler
#         model = BertForSequenceClassification.from_pretrained(
#             "bert-base-uncased",
#             num_labels=3
#         ).to(DEVICE)
#         optimizer = AdamW(model.parameters(), lr=lr)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
#         criterion = nn.CrossEntropyLoss()

#         # Train the model
#         for epoch in range(num_epochs):
#             train_loss = train_model(model, train_loader, optimizer, scheduler, criterion)
#             print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}")

#         # Validate the model
#         _, val_accuracy, _, _ = evaluate_model(model, val_loader, criterion)
#         print(f"Validation Accuracy: {val_accuracy:.4f}")

#         # Update the best configuration if the current one is better
#         if val_accuracy > best_accuracy:
#             best_accuracy = val_accuracy
#             best_config = {
#                 "learning_rate": lr,
#                 "batch_size": batch_size,
#                 "num_epochs": num_epochs
#             }
#             torch.save(model.state_dict(), "models/bert_model_best.pt")

#     print(f"\nBest Configuration: {best_config}")
#     print(f"Best Validation Accuracy: {best_accuracy:.4f}")
#     return best_config

# print("Starting hyperparameter tuning...")
# best_hyperparameters = hyperparameter_tuning(search_space, train_loader, val_loader)
# print(f"Best Hyperparameters: {best_hyperparameters}")