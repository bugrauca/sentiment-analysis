import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score


def train_naive_bayes(X, y, model_path):

    """
    Args:
        X (sparse matrix): TF-IDF features.
        y (Series or array): Sentiment labels.
        model_path (str): Path to save the trained model.

    Returns:
        None
    """

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Naïve Bayes model...")
    model = MultinomialNB()
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # wb: write binary mode, opens file at path in binary write mode
    print("Saving model...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    vectorizer_path = "models/vectorizer.pkl"
    processed_data_path = "data/processed/preprocessed.csv"
    model_save_path = "models/naive_bayes.pkl"

    print("Loading preprocessed data and vectorizer...")
    with open(vectorizer_path, "rb") as file:
        vectorizer = pickle.load(file)

    data = pd.read_csv(processed_data_path)

    # Drop rows with NaN values in 'text' column
    print("Cleaning missing values in text column...")
    data.dropna(subset=['text'], inplace=True)

    X = vectorizer.transform(data['text'])

    y = data['sentiment']

    train_naive_bayes(X, y, model_save_path)