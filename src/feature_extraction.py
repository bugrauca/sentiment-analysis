import os
import pickle
#from scipy.sparse import hstack
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(input_file, save_path, max_features=5000):

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Preprocessed data not found at {input_file}. Run preprocessing first!")

    print("Loading preprocessed data...")
    data = pd.read_csv(input_file)

    # Remove any unexpected NaN values from 'text' column
    data.dropna(subset=['text'], inplace=True)

    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_text = vectorizer.fit_transform(data['text'])  # TF-IDF features from text

    X = vectorizer.fit_transform(data['text'])
    y = data['sentiment']

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"TF-IDF vectorizer saved to {save_path}")

    return X, y

if __name__ == "__main__":
    input_path = "data/processed/preprocessed.csv"
    vectorizer_path = "models/vectorizer.pkl"
    extract_features(input_path, vectorizer_path)