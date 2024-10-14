import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob, Word
import re

def load_data(file_path):
    return pd.read_csv(file_path)

def normalize_case(df, column):
    df[column] = df[column].str.lower()
    return df

def remove_punctuation(df, column):
    df[column] = df[column].str.replace('[^\w\s]', '')
    return df

def remove_numbers(df, column):
    df[column] = df[column].apply(lambda x: re.sub(r'\b\d+\b', '', x))
    return df

def remove_stopwords(df, column):
    sw = set(stopwords.words('english'))
    df[column] = df[column].apply(lambda x: " ".join(word for word in str(x).split() if word not in sw))
    return df

def remove_rare_words(df, column, min_count=1):
    temp_df = pd.Series(' '.join(df[column]).split()).value_counts()
    drops = temp_df[temp_df <= min_count]
    df[column] = df[column].apply(lambda x: " ".join(word for word in x.split() if word not in drops))
    return df

def tokenize(df, column):
    df[column] = df[column].apply(lambda x: word_tokenize(x))
    return df

def lemmatize(df, column):
    lemmatizer = WordNetLemmatizer()
    df[column] = df[column].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in (x if isinstance(x, list) else x.split())]))
    return df

def preprocess_data(df, column):
    df = normalize_case(df, column)
    df = remove_punctuation(df, column)
    df = remove_numbers(df, column)
    df = remove_stopwords(df, column)
    df = remove_rare_words(df, column)
    df = tokenize(df, column)
    df = lemmatize(df, column)
    return df

if __name__ == "__main__":   
    file_path = "wiki_data.csv"  
    df = load_data(file_path)

    df = preprocess_data(df, 'text')

    print(df['text'].head())

    df.to_csv("preprocessed_data.csv", index=False)