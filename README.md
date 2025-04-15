# Sentiment Analysis Project

## Overview

This project aims to perform sentiment analysis on social media posts, classifying their sentiment into predefined categories (Negative, Neutral, Positive). It employs advanced Natural Language Processing (NLP) techniques and machine learning models, including LSTM, BERT, and Naive Bayes, to analyze and categorize textual data. The analysis leverages the Sentiment140 dataset for model training and evaluation. The project spans the full pipeline, from data preprocessing to model evaluation, ensuring a comprehensive approach to sentiment classification.

## Project Structure

```
Sentiment-Analysis-Project/
|
|-- data/
|   |-- raw/                     # Raw dataset file
|   |-- processed/               # Preprocessed dataset
|
|-- models/
|   |-- bert_model.pt            # Saved BERT model
|   |-- lstm_model.pt            # Saved LSTM model
|   |-- naive_bayes.pkl          # Saved Naive Bayes model
|   |-- vectorizer.pkl           # TF-IDF vectorizer for LSTM
|
|-- results/
|   |-- metrics/                 # Metrics and outputs for evaluation
|
|-- notebook/
|   |-- sa_lstm_bert.ipynb       # Jupyter notebook for LSTM and Bert models training
|
|-- src/
|   |-- bert_model.py            # BERT model training and evaluation script
|   |-- lstm_model.py            # LSTM model training and evaluation script
|   |-- naive_bayes_model.py     # Naive Bayes model training and evaluation script
|   |-- data_preparation.py      # Data preprocessing script
|   |-- feature_extraction.py    # TF-IDF vectorizer script
|   |-- visualization.py         # Model output visualization script
|
|-- README.md                    # Project documentation (this file)
|-- requirements.txt             # Required Python packages
```

## Requirements

To run this project, the following dependencies are required:

- Python 3.8+
- NumPy
- Pandas
- scikit-learn
- PyTorch
- Transformers (Hugging Face)
- NLTK
- tqdm
- Matplotlib
- seaborn
- Git LFS (for large files)
- Google Colab (Optional)

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Cloning the Repository

This repository uses Git LFS to store large files such as the dataset CSVs and the BERT model weights.  
To clone the project properly:

```bash
git lfs install
git clone https://github.com/bugrauca/sentiment-analysis.git
```

## Dataset

The Sentiment140 dataset is used for training and evaluation. This dataset contains labeled tweets, where each tweet is associated with a sentiment label:

- 0: Negative
- 2: Neutral
- 4: Positive

The dataset is preprocessed to map these labels to [0, 1, 2] and is stored in the `data/processed/` directory.
Sentiment140 dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140) and be placed under the `data/raw/` folder.

## Models

### Naive Bayes

The Naive Bayes model is implemented in `naive_bayes_model.py`. It uses a TF-IDF vectorizer to extract features from the preprocessed text data and applies the Naive Bayes algorithm for sentiment classification. This model provides a lightweight and interpretable baseline for comparison.

### LSTM

The LSTM (Long Short-Term Memory) model is implemented in `lstm_model.py`. It uses an embedding layer, an LSTM layer, and a fully connected output layer. The model is trained using preprocessed data tokenized with a custom tokenizer based on a fitted vectorizer.

### BERT

The BERT (Bidirectional Encoder Representations from Transformers) model is implemented in `bert_model.py`. The Hugging Face Transformers library is used to fine-tune a pre-trained BERT model for sentiment classification. This model leverages a more sophisticated tokenizer and uses mixed precision training for efficiency.

## Features

### Preprocessing

- Text cleaning (removal of URLs, mentions, hashtags, special characters)
- Lowercasing
- Tokenization
- Stopword removal
- Label transformation (mapping sentiment labels to [0, 1, 2])

### Feature Extraction

- TF-IDF vectorization for Naive Bayes
- Embedding layer for LSTM
- Pre-trained BERT tokenizer for BERT

### Model Training and Evaluation

- Multiple models implemented and compared for performance.

### Results

- Evaluation metrics such as accuracy, precision, recall, and F1-score.

## How to Run

### Preprocessing

Ensure that the dataset is preprocessed and stored in `data/processed/preprocessed.csv`. Preprocessing includes cleaning text, tokenization, and label transformation. Use external preprocessing scripts if necessary.

### Extracting Features

To generate features for the models, run:

```bash
python scripts/feature_extraction.py
```

### Training the Naive Bayes Model

To train and evaluate the Naive Bayes model, run:

```bash
python src/naive_bayes_model.py
```

This script will:

- Train the Naive Bayes model on the preprocessed dataset.
- Save the trained model to `models/naive_bayes.pkl`.
- Save evaluation metrics and outputs to `results/metrics/`.

### Training the LSTM Model

To train and evaluate the LSTM model, run:

```bash
python src/lstm_model.py
```

This script will:

- Train the LSTM model on the preprocessed dataset.
- Save the trained model to `models/lstm_model.pt`.
- Save evaluation metrics and outputs to `results/metrics/`.

The model can also be trained using Google Colab for faster training and outcome. In the directory `notebook/sa_lstm_bert.ipynb` has sections divided for models. `Final LSTM Model and Evaluation` can be run for training and evaluation. (Optional)

### Training the BERT Model

To train and evaluate the BERT model, run:

```bash
python src/bert_model.py
```

This script will:

- Fine-tune the BERT model on the preprocessed dataset.
- Save the trained model to `models/bert_model.pt`.
- Save evaluation metrics and outputs to `results/metrics/`.

The model can also be trained using Google Colab for faster training and outcome. In the directory `notebook/sa_lstm_bert.ipynb` has sections divided for models. `Final BERT Model and Evaluation` can be run for training and evaluation. (Optional)

## Evaluation

Each script generates the following evaluation metrics:

- Accuracy
- Classification Report
- Confusion Matrix

Metrics and visualization outputs are saved in the `results/metrics/` directory. These can be used for further analysis and reporting.

## Visualization

To generate the visualizations of metrics:

```bash
python src/visualization.py
```

This script will output the plots to the `results/plots/` directory for further analysis or inclusion in reports.

## Acknowledgments

This project is built as part of a university module and uses the [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) dataset. The implementation is powered by PyTorch, scikit-learn, and Hugging Face Transformers libraries.
