import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

lstm_training_history_path = "results/metrics/lstm_training_history.pth"
lstm_true_labels_path = "results/metrics/lstm_true_labels.pth"
lstm_predictions_path = "results/metrics/lstm_predictions.pth"

bert_visualization_outputs_path = "results/metrics/bert_visualization_outputs.pth"

output_dir = "results/plots"
os.makedirs(output_dir, exist_ok=True)

print("Loading LSTM outputs...")
lstm_training_history = torch.load(lstm_training_history_path)
lstm_true_labels = torch.load(lstm_true_labels_path)
lstm_predictions = torch.load(lstm_predictions_path)

print("Loading BERT outputs...")
bert_outputs = torch.load(bert_visualization_outputs_path)

# Extract necessary fields
lstm_train_losses = lstm_training_history.get("train_losses", [])
lstm_val_losses = lstm_training_history.get("val_losses", [])
lstm_val_accuracies = lstm_training_history.get("val_accuracies", [])

if not lstm_train_losses or not lstm_val_losses or not lstm_val_accuracies:
    raise ValueError("Training history metrics are empty. Please check the training script.")

bert_train_losses = bert_outputs.get("train_losses", [])
bert_val_losses = bert_outputs.get("val_losses", [])
bert_val_accuracies = bert_outputs.get("val_accuracies", [])
bert_predictions = bert_outputs.get("predictions", [])
bert_true_labels = bert_outputs.get("true_labels", [])

def plot_training_loss(train_losses, model_name):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss", marker="o")
    plt.title(f"{model_name} Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(output_dir, f"{model_name}_training_loss.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"{model_name} training loss plot saved at {save_path}")
    plt.close()

def plot_validation_metrics(val_losses, val_accuracies, model_name):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(val_losses) + 1)

    plt.plot(epochs, val_losses, label="Validation Loss", marker="o", color="red")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy", marker="s", color="green")

    plt.title(f"{model_name} Validation Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(output_dir, f"{model_name}_validation_metrics.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"{model_name} validation metrics plot saved at {save_path}")
    plt.close()

def plot_confusion_matrix(true_labels, predictions, model_name):
    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{model_name} Confusion Matrix")
    save_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"{model_name} confusion matrix saved at {save_path}")
    plt.close()

print("Generating LSTM visualizations...")
plot_training_loss(lstm_train_losses, model_name="LSTM")
plot_validation_metrics(lstm_val_losses, lstm_val_accuracies, model_name="LSTM")
plot_confusion_matrix(lstm_true_labels, lstm_predictions, model_name="LSTM")

print("Generating BERT visualizations...")
plot_training_loss(bert_train_losses, model_name="BERT")
plot_validation_metrics(bert_val_losses, bert_val_accuracies, model_name="BERT")
plot_confusion_matrix(bert_true_labels, bert_predictions, model_name="BERT")
