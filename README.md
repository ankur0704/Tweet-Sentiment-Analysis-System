# Tweet Sentiment Analysis System

A mini-project that classifies social-media text (tweets) as **Positive**, **Negative**, or **Neutral** using classical NLP preprocessing and a Logistic Regression classifier trained on the NLTK `twitter_samples` corpus.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Technology Stack](#3-technology-stack)
4. [System Architecture](#4-system-architecture)
5. [NLP Preprocessing Pipeline](#5-nlp-preprocessing-pipeline)
6. [Machine Learning Model](#6-machine-learning-model)
7. [Web Application Interface](#7-web-application-interface)
8. [Setup & Installation](#8-setup--installation)
9. [Running the Project](#9-running-the-project)
10. [Model Performance](#10-model-performance)
11. [Design Decisions](#11-design-decisions)
12. [Known Limitations & Future Improvements](#12-known-limitations--future-improvements)

---

## 1. Project Overview

**Problem Statement:** Design a sentiment analysis system that processes social media text (tweets) and classifies opinions into positive, negative, or neutral categories using NLP preprocessing and machine learning techniques.

### What the System Does

| Input | Output |
|---|---|
| Any raw text / tweet | Sentiment label (Positive / Negative / Neutral) |
| | Confidence score (0 – 100 %) |
| | Probability breakdown (positive vs. negative) |
| | Session history of past analyses |

---

## 2. Project Structure

```
nlp_minipro_a4/
│
├── app.py                    # Streamlit web application (main entry point)
├── requirements.txt          # Python dependencies with version pins
│
├── src/
│   ├── preprocess.py         # NLP text cleaning & normalisation pipeline
│   └── train.py              # Model training, evaluation & artefact saving
│
├── models/                   # Auto-created during training
│   ├── sentiment_model.pkl   # Serialised LogisticRegression classifier
│   └── tfidf_vectorizer.pkl  # Serialised TfidfVectorizer
│
└── .venv/                    # Python virtual environment (not committed)
```

---

## 3. Technology Stack

| Layer | Library | Purpose |
|---|---|---|
| Language | Python 3.9+ | Core runtime |
| NLP | NLTK | Tokenisation, stop words, stemming |
| Feature Extraction | scikit-learn `TfidfVectorizer` | Convert text → numeric vectors |
| ML Model | scikit-learn `LogisticRegression` | Binary sentiment classifier |
| Serialisation | joblib | Save / load model artefacts |
| Web UI | Streamlit | Interactive, browser-based interface |
| Data Wrangling | pandas, numpy | Dataset handling during training |

---

## 4. System Architecture

```
                        ┌─────────────────────────────────────┐
                        │         TRAINING PIPELINE            │
                        │  (run once: python src/train.py)     │
                        │                                      │
      NLTK twitter       │  load_dataset()                      │
      _samples corpus  ──►  preprocess_dataset()               │
      (10 k tweets)      │  build_and_train()                   │
                        │  evaluate()                          │
                        │  save_artefacts()                    │
                        └────────────┬────────────────────────-┘
                                     │ .pkl files
                                     ▼
                        ┌─────────────────────────────────────┐
                        │       INFERENCE (app.py)             │
                        │                                      │
      User types text ──► preprocess_tweet()                  │
                        │  vectorizer.transform()              │
                        │  model.predict_proba()               │
                        │  Neutral threshold check (≥ 65 %)   │
                        │         │                            │
                        │         ▼                            │
                        │  Streamlit UI renders result card,   │
                        │  confidence bars & session history   │
                        └─────────────────────────────────────┘
```

---

## 5. NLP Preprocessing Pipeline

**File:** `src/preprocess.py`  
**Public function:** `preprocess_tweet(tweet: str) → str`

Each tweet passes through the following ordered steps:

| Step | Operation | Example |
|---|---|---|
| 1 | Remove URLs (`https?://…`) | `"Check https://t.co/abc"` → `"Check "` |
| 2 | Remove @mentions | `"@jack hello"` → `" hello"` |
| 3 | Remove stock tickers (`$AAPL`) | `"Bought $TSLA"` → `"Bought "` |
| 4 | Strip legacy RT prefix | `"RT Great tweet"` → `"Great tweet"` |
| 5 | Remove `#` symbol (keep word) | `"#NLP rocks"` → `"NLP rocks"` |
| 6 | TweetTokenizer | Lowercases, reduces repeated chars (`soooo` → `soo`) |
| 7 | Stop-word & punctuation filter | Removes `the`, `is`, `,`, `.` … |
| 8 | Porter Stemming | `"loving"` → `"love"`, `"amazing"` → `"amaz"` |

### Performance Design
All heavy objects (`TweetTokenizer`, `PorterStemmer`, stop-word set) are instantiated **once at module import time** as private module-level singletons, avoiding repeated construction overhead during batch preprocessing.

---

## 6. Machine Learning Model

**File:** `src/train.py`

### Dataset
- **Source:** `nltk.corpus.twitter_samples`
- **Size:** 10,000 tweets (5,000 positive + 5,000 negative)
- **Split:** 80% train / 20% test (stratified by default via sklearn)

### Feature Extraction
```
TfidfVectorizer(max_features=5_000)
```
TF-IDF (Term Frequency – Inverse Document Frequency) scores each token by how distinctive it is across the corpus, giving more weight to rare but informative words.

### Classifier
```
LogisticRegression(max_iter=1_000)
```
A fast, interpretable linear model well-suited to high-dimensional sparse text features. Despite its simplicity it achieves strong baseline accuracy on polarity classification tasks.

### Neutral Inference
The underlying dataset only contains positive / negative labels. The **Neutral** class is inferred at runtime:

> If `max(P(positive), P(negative)) < 0.65`, the system returns **Neutral**.

The 0.65 threshold is configurable via `NEUTRAL_THRESHOLD` in `app.py`.

### Training Pipeline Functions

| Function | Responsibility |
|---|---|
| `load_dataset()` | Fetch tweets from NLTK corpus, attach labels |
| `preprocess_dataset()` | Apply `preprocess_tweet()` to each row, drop empty results |
| `build_and_train()` | Fit vectorizer and classifier |
| `evaluate()` | Print accuracy & full classification report |
| `save_artefacts()` | Persist `.pkl` files to `models/` |

---

## 7. Web Application Interface

**File:** `app.py`  
**Framework:** Streamlit  
**Entry point:** `streamlit run app.py`

### Features

| Feature | Details |
|---|---|
| Text input area | Multi-line tweet input with placeholder guidance |
| Sentiment result card | Glassmorphic card with colour-coded label + emoji |
| Confidence score | Percentage displayed inside result card |
| Probability bars | Side-by-side `st.progress` bars for Positive / Negative |
| Analysis history | Scrollable session log of all analysed texts with "Clear" button |
| Google Fonts | Inter typeface for premium typography |
| Dark theme | Linear gradient `#0d1117 → #161b22` background |

### Inference Flow in `app.py`

```python
result = predict_sentiment(user_text)
#  └─ preprocess_tweet(text)     # clean
#  └─ vectorizer.transform(...)  # TF-IDF features
#  └─ model.predict_proba(...)   # class probabilities
#  └─ threshold check            # Neutral if max_prob < 0.65
```

---

## 8. Setup & Installation

### Prerequisites
- Python 3.9 or higher
- `pip`

### Steps

```bash
# 1. Clone / navigate to project root
cd /path/to/nlp_minipro_a4

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate it
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Train the model (downloads NLTK data automatically)
python src/train.py
```

---

## 9. Running the Project

### Step 1 — Train the model (once)
```bash
source .venv/bin/activate
python src/train.py
```

Expected output:
```
Loading NLTK twitter_samples dataset...
  Total samples: 10000  (pos=5000, neg=5000)
Preprocessing tweets ...
Fitting TF-IDF vectorizer (max_features=5000) ...
Training Logistic Regression model ...
Test Accuracy: 0.7425
...
Artefacts saved:
  Model      → models/sentiment_model.pkl
  Vectorizer → models/tfidf_vectorizer.pkl
Training complete.
```

### Step 2 — Launch the web app
```bash
streamlit run app.py
```

Navigate to **http://localhost:8501** in your browser.

---

## 10. Model Performance

Evaluated on 2,000 held-out tweets (20% of corpus):

| Metric | Negative | Positive | Macro Avg |
|---|---|---|---|
| Precision | 0.72 | 0.77 | 0.74 |
| Recall | 0.78 | 0.71 | 0.74 |
| F1-Score | 0.75 | 0.74 | 0.74 |
| **Accuracy** | — | — | **74.25 %** |

> A 74 % accuracy on binary tweet polarity with a simple TF-IDF + Logistic Regression baseline is consistent with published benchmarks on the NLTK `twitter_samples` corpus (typical range: 70–80 %).

---

## 11. Design Decisions

| Decision | Rationale |
|---|---|
| Logistic Regression over Naïve Bayes | LR with TF-IDF generally outperforms MultinomialNB on balanced datasets because it optimises log-loss directly and handles feature correlations better. |
| Porter Stemmer over Lemmatizer | Faster and sufficient for polarity analysis; semantic nuance from lemmatization doesn't meaningfully improve bag-of-words models. |
| TweetTokenizer over `word_tokenize` | Designed for social media: handles emoticons, repeated characters (`hahaha`), and contractions without false splits. |
| Neutral via threshold, not training data | The NLTK corpus has no neutral label. Threshold-based detection is a principled heuristic used in industry when neutral examples are unavailable. |
| Streamlit for UI | Minimal boilerplate, native support for progress bars and session state; ideal scope for a mini-project demo. |
| Module-level NLP singletons | Avoids re-constructing `PorterStemmer` and `TweetTokenizer` on each of the ~8,000 training calls — reduces train time noticeably. |

---

## 12. Known Limitations & Future Improvements

### Current Limitations
- **Binary training data only** — the Neutral class is heuristic (probability threshold), not learned.
- **No negation handling** — "not good" may be misclassified because stemming destroys the negative context.
- **English only** — stop-word list and stemmer are English-specific.
- **No emoji semantics** — emojis are dropped during tokenisation rather than converted to sentiment signals.

### Suggested Future Improvements

| Improvement | Benefit |
|---|---|
| Add a labelled Neutral dataset (e.g., SemEval 2017) and retrain as 3-class | Proper Neutral predictions |
| Replace Logistic Regression with fine-tuned `twitter-roberta-base-sentiment` (HuggingFace) | +10–15 % accuracy |
| Add VADER as a rule-based secondary signal | Better on slang / emojis |
| Implement negation-aware tokenisation | Fixes "not happy" misclassifications |
| Add `pytest` unit tests for `preprocess_tweet` | Prevents regressions |
| Add batch CSV upload in the UI | Analyse multiple tweets at once |
| Cache preprocessing with `functools.lru_cache` for repeated inputs | Faster repeated queries |
