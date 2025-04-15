import os
import re
import pandas as pd
import emoji

def clean_text(text):
    # Remove links, mentions, hashtags, and special characters
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = text.strip().lower()
    return text

def load_and_prepare_data(raw_file, processed_file):

    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"Raw dataset not found at {raw_file}. Please download it first!")

    print("Loading raw data...")
    cols = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    data = pd.read_csv(raw_file, encoding='latin-1', names=cols)

    # labels: 0 = negative, 2 = positive
    data['sentiment'] = data['sentiment'].replace({0: 0, 4: 2})

    print("Cleaning data...")
    data['text'] = data['text'].apply(clean_text)

    # Drop rows with NaN values in the 'text' column
    data.dropna(subset=['text'], inplace=True)

    print("Saving preprocessed data...")
    os.makedirs(os.path.dirname(processed_file), exist_ok=True)
    #data.to_csv(processed_file, index=False)  # Save all columns, including metadata
    data[['sentiment', 'text']].to_csv(processed_file, index=False)
    print(f"Preprocessed data saved to {processed_file}")


if __name__ == "__main__":
    raw_file_path = "data/raw/sentiment140.csv"
    processed_file_path = "data/processed/preprocessed.csv"
    load_and_prepare_data(raw_file_path, processed_file_path)
