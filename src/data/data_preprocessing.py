import os
import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text):
    """Lemmatize each word in the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(y) for y in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    Text = [i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_numbers(text):
    """Remove all digits from the text."""
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    """Convert all words in the text to lowercase."""
    text = text.split()
    text = [y.lower() for y in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations and extra whitespace from the text."""
    text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', "")
    text = re.sub(r'\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    """Set text to NaN if sentence has fewer than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df):
    """Apply all preprocessing steps to the 'text' column of the DataFrame."""
    df['text'] = df['text'].apply(lower_case)
    df['text'] = df['text'].apply(remove_stop_words)
    df['text'] = df['text'].apply(removing_numbers)
    df['text'] = df['text'].apply(removing_punctuations)
    df['text'] = df['text'].apply(removing_urls)
    df['text'] = df['text'].apply(lemmatization)
    return df

def normalized_sentence(sentence):
    """Apply all preprocessing steps to a single sentence."""
    sentence = lower_case(sentence)
    sentence = remove_stop_words(sentence)
    sentence = removing_numbers(sentence)
    sentence = removing_punctuations(sentence)
    sentence = removing_urls(sentence)
    sentence = lemmatization(sentence)
    return sentence

if __name__ == "__main__":
    import sys
    # Load raw train and test data
    train_data = pd.read_csv("data/raw/train.csv")
    test_data = pd.read_csv("data/raw/test.csv")

    # Rename 'content' column to 'text' if necessary
    for df in [train_data, test_data]:
        if 'content' in df.columns and 'text' not in df.columns:
            df.rename(columns={'content': 'text'}, inplace=True)

    # Check for 'text' column in both train and test data
    if 'text' not in train_data.columns or 'text' not in test_data.columns:
        print("ERROR: 'text' column not found in train.csv or test.csv. Columns found in train.csv:", train_data.columns.tolist())
        print("Columns found in test.csv:", test_data.columns.tolist())
        sys.exit(1)

    # Normalize train and test data
    train_data = normalize_text(train_data)
    test_data = normalize_text(test_data)

    # Save processed data to CSV files
    os.makedirs("data/processed", exist_ok=True)  # Ensure the directory exists
    train_data.to_csv("data/processed/train.csv", index=False)
    test_data.to_csv("data/processed/test.csv", index=False)