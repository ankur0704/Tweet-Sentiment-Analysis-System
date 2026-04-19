"""
train.py
--------
Training script for the Tweet Sentiment Analysis System.

Loads the NLTK twitter_samples dataset, preprocesses tweets, extracts TF-IDF
features, trains a Logistic Regression classifier, evaluates it, and persists
the model artefacts to the models/ directory.

Usage
-----
    cd <project_root>
    python src/train.py

Artefacts produced
------------------
    models/sentiment_model.pkl    – trained LogisticRegression
    models/tfidf_vectorizer.pkl   – fitted TfidfVectorizer
"""
import os
import sys
import joblib
import pandas as pd
from nltk.corpus import twitter_samples
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Make sure 'src/' is on the path when run as `python src/train.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_tweet

# ---------------------------------------------------------------------------
# Configuration constants (easy to adjust for experiments)
# ---------------------------------------------------------------------------

RANDOM_STATE   = 42
TEST_SIZE      = 0.20
TFIDF_MAX_FEAT = 5000
LR_MAX_ITER    = 1000

# ---------------------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    """Load and label the NLTK twitter_samples corpus."""
    print("Loading NLTK twitter_samples dataset...")
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')

    df_pos = pd.DataFrame({'tweet': positive_tweets, 'sentiment': 1})  # 1 = Positive
    df_neg = pd.DataFrame({'tweet': negative_tweets, 'sentiment': 0})  # 0 = Negative
    df = pd.concat([df_pos, df_neg], ignore_index=True)

    print(f"  Total samples: {len(df)}  (pos={len(df_pos)}, neg={len(df_neg)})")
    return df


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply NLP preprocessing to all tweets."""
    print("Preprocessing tweets ...")
    df = df.copy()
    df['clean_tweet'] = df['tweet'].apply(preprocess_tweet)
    # Drop rows where preprocessing produced an empty string
    df = df[df['clean_tweet'].str.strip() != ''].reset_index(drop=True)
    print(f"  Samples after cleaning: {len(df)}")
    return df


def build_and_train(X_train, y_train):
    """Fit the TF-IDF vectorizer and Logistic Regression model."""
    print(f"Fitting TF-IDF vectorizer (max_features={TFIDF_MAX_FEAT}) ...")
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEAT)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    print("Training Logistic Regression model ...")
    model = LogisticRegression(
        solver='saga',         # better suited for large sparse TF-IDF matrices
        C=1.0,
        max_iter=LR_MAX_ITER,
        random_state=RANDOM_STATE
    )
    model.fit(X_train_tfidf, y_train)
    return model, vectorizer


def evaluate(model, vectorizer, X_test, y_test) -> float:
    """Print evaluation metrics and return test accuracy."""
    X_test_tfidf = vectorizer.transform(X_test)
    predictions  = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, predictions)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(classification_report(y_test, predictions, target_names=['Negative', 'Positive']))
    return acc


def save_artefacts(model, vectorizer, base_dir: str) -> None:
    """Persist trained model and vectorizer to disk."""
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    model_path      = os.path.join(models_dir, 'sentiment_model.pkl')
    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')

    joblib.dump(model,      model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"\nArtefacts saved:")
    print(f"  Model      → {model_path}")
    print(f"  Vectorizer → {vectorizer_path}")


def train_model() -> None:
    """End-to-end training pipeline."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    df = load_dataset()
    df = preprocess_dataset(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_tweet'], df['sentiment'],
        test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model, vectorizer = build_and_train(X_train, y_train)
    evaluate(model, vectorizer, X_test, y_test)
    save_artefacts(model, vectorizer, base_dir)
    print("\nTraining complete.")


if __name__ == "__main__":
    train_model()
