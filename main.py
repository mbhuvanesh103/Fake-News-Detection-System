
import os
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import joblib

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

RANDOM_STATE = 42


def load_and_label(fake_path, true_path, text_columns=None):
    """Load CSVs and create a combined DataFrame with label column.
    label: 0 = fake, 1 = true
    text_columns (list): columns to combine for text. If None, tries ['title','text']
    """
    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")

    df_fake['label'] = 0
    df_true['label'] = 1

    if text_columns is None:
        # prefer columns 'title' and 'text' commonly present in Kaggle dataset
        text_columns = [c for c in ['title', 'text'] if c in df_fake.columns and c in df_true.columns]

    if not text_columns:
        # fallback: use the first text-like column other than label
        def pick_text_col(df):
            for c in df.columns:
                if df[c].dtype == object and c.lower() not in ('label',):
                    return c
            return None

        c1 = pick_text_col(df_fake)
        c2 = pick_text_col(df_true)
        if c1 and c2 and c1 == c2:
            text_columns = [c1]
        else:
            raise ValueError("Could not auto-detect text columns. Provide text_columns explicitly.")

    # combine columns into single 'content' field
    def combine_text(row):
        parts = [str(row[c]) for c in text_columns if c in row and pd.notna(row[c])]
        return ' '.join(parts).strip()

    df_fake['content'] = df_fake.apply(combine_text, axis=1)
    df_true['content'] = df_true.apply(combine_text, axis=1)

    df = pd.concat([df_fake[['content', 'label']], df_true[['content', 'label']]], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def basic_eda(df):
    print("Total samples:", len(df))
    print(df['label'].value_counts())
    print("Missing content:", df['content'].isna().sum())
    df['length'] = df['content'].apply(lambda x: len(str(x).split()))
    print(df.groupby('label')['length'].describe())


def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text)
    # lower
    text = text.lower()
    # remove urls
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # remove html tags
    text = re.sub(r'<.*?>', ' ', text)
    # remove non-alphanumeric characters (keep spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    text = clean_text(text)
    if not text:
        return ''
    try:
        nltk.download('punkt')      # standard tokenizer
        nltk.download('stopwords')  # you already have this
        nltk.download('wordnet')
        nltk.download('punkt_tab')
        tokens = nltk.word_tokenize(text)
    except Exception:
        return ''
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1 and not t.isnumeric()]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)



def prepare_data(df, test_size=0.2):
    df['content_clean'] = df['content'].apply(preprocess_text)
    X = df['content_clean'].values
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


def build_vectorizer(max_features=20000, ngram_range=(1, 2)):
    vect = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    return vect


def train_and_evaluate(X_train, X_test, y_train, y_test, vectorizer, models=None):
    if models is None:
        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            'MultinomialNB': MultinomialNB(),
            'LinearSVC': LinearSVC(max_iter=5000, random_state=RANDOM_STATE)
        }

    # fit vectorizer
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        probs = None
        try:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_test_vec)[:, 1]
            elif hasattr(model, 'decision_function'):
                probs = model.decision_function(X_test_vec)
        except Exception:
            probs = None

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc = roc_auc_score(y_test, probs) if probs is not None else None

        print(f"{name} -> accuracy: {acc:.4f}, precision: {prec:.4f}, recall: {rec:.4f}, f1: {f1:.4f}, roc_auc: {roc}")
        print('\nClassification report:\n', classification_report(y_test, preds, digits=4))

        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()

        results[name] = {
            'model': model,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': roc if roc is not None else -1
        }

    return results


def save_artifacts(best_model, vectorizer, out_dir='artifacts'):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(out_dir, 'best_model.joblib'))
    joblib.dump(vectorizer, os.path.join(out_dir, 'tfidf_vectorizer.joblib'))
    print('Saved model and vectorizer to', out_dir)


def predict_text(texts, model, vectorizer):
    if isinstance(texts, str):
        texts = [texts]
    cleaned = [preprocess_text(t) for t in texts]
    vec = vectorizer.transform(cleaned)
    preds = model.predict(vec)
    try:
        probs = model.predict_proba(vec)[:, 1]
    except Exception:
        try:
            probs = model.decision_function(vec)
        except Exception:
            probs = None
    return list(zip(texts, preds, probs))


def find_best_model(results):
    # choose by highest f1, tie-breaker accuracy
    best = None
    best_score = -1
    for name, info in results.items():
        score = info['f1']
        if score > best_score:
            best_score = score
            best = (name, info)
    return best


def main(fake_path, true_path, out_dir='artifacts'):
    print('Loading data...')
    df = load_and_label(fake_path, true_path)
    basic_eda(df)

    print('\nPreprocessing & splitting...')
    X_train, X_test, y_train, y_test = prepare_data(df)

    print('\nBuilding vectorizer...')
    vectorizer = build_vectorizer(max_features=20000, ngram_range=(1, 2))

    print('\nTraining models...')
    results = train_and_evaluate(X_train, X_test, y_train, y_test, vectorizer)

    # pick best
    best_name, best_info = find_best_model(results)
    print(f"\nBest model: {best_name} with F1={best_info['f1']:.4f}")

    # save model and vectorizer
    save_artifacts(best_info['model'], vectorizer, out_dir=out_dir)

    # example predictions
    examples = [
        "President signs new education reform bill.",
        "Scientists confirm the ancient cure that instantly cures all diseases!"
    ]
    loaded_vectorizer = vectorizer  # already fitted
    loaded_model = best_info['model']
    preds = predict_text(examples, loaded_model, loaded_vectorizer)
    print('\nExample predictions:')
    for text, pred, prob in preds:
        label = 'REAL' if pred == 1 else 'FAKE'
        print(f"{label} | prob/conf: {prob} | {text}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fake News Detection - train and save model')
    parser.add_argument('--fake', type=str, default='/mnt/data/Fake.csv', help='Path to Fake.csv')
    parser.add_argument('--true', type=str, default='/mnt/data/True.csv', help='Path to True.csv')
    parser.add_argument('--out', type=str, default='artifacts', help='Output directory for model and vectorizer')
    args = parser.parse_args()

    main(args.fake, args.true, out_dir=args.out)
