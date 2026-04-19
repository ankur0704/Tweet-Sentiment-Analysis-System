# Tweet Sentiment Analysis System — Deep Dive Explanation

## Table of Contents
1. [Big Picture](#1-big-picture)
2. [What Does "Sentiment Analysis" Mean?](#2-what-does-sentiment-analysis-mean)
3. [Data: Where It All Starts](#3-data-where-it-all-starts)
4. [Step-by-Step: NLP Preprocessing Pipeline](#4-step-by-step-nlp-preprocessing-pipeline)
5. [Step-by-Step: Feature Extraction (TF-IDF)](#5-step-by-step-feature-extraction-tf-idf)
6. [Step-by-Step: Machine Learning Model](#6-step-by-step-machine-learning-model)
7. [Neutral Sentiment — A Special Design Choice](#7-neutral-sentiment--a-special-design-choice)
8. [The Web Application (app.py)](#8-the-web-application-apppy)
9. [End-to-End Data Flow](#9-end-to-end-data-flow)
10. [Glossary of All NLP/ML Techniques Used](#10-glossary-of-all-nlpml-techniques-used)

---

## 1. Big Picture

The system has **two separate phases**:

```
PHASE 1 — TRAINING  (run once, offline)
────────────────────────────────────────────────────
Raw Tweets ──► NLP Cleaning ──► TF-IDF Features ──► Train Model ──► Save .pkl

PHASE 2 — INFERENCE  (runs every time user submits text)
────────────────────────────────────────────────────
User Text ──► NLP Cleaning ──► TF-IDF Transform ──► Predict ──► Show Result
```

Phase 1 happens in `src/train.py`. Phase 2 happens inside `app.py` every time you click "Analyze Sentiment".

---

## 2. What Does "Sentiment Analysis" Mean?

Sentiment analysis (also called **opinion mining**) is the task of automatically determining the **emotional tone** of a piece of text. For tweets, the three target classes are:

| Class | Meaning | Example |
|---|---|---|
| **Positive** | Expresses approval, happiness, praise | "This is absolutely amazing! 🤩" |
| **Negative** | Expresses criticism, sadness, anger | "Worst experience of my life 😠" |
| **Neutral** | Factual, no emotional signal | "Going to the grocery store today." |

This is a **text classification** problem — given an input string, output one of N categories.

---

## 3. Data: Where It All Starts

**Dataset: `nltk.corpus.twitter_samples`**

This is a built-in NLTK corpus with **10,000 real tweets** scraped from Twitter:

```
positive_tweets.json  →  5,000 tweets labelled Positive  ( label = 1 )
negative_tweets.json  →  5,000 tweets labelled Negative  ( label = 0 )
```

The dataset is **perfectly balanced** (equal positive and negative), so the model isn't biased toward one class. There are **no neutral tweets** in this dataset — that is handled separately (see Section 7).

**Train/Test Split:**
```
80%  →  8,000 tweets  →  used to TRAIN the model
20%  →  2,000 tweets  →  used to TEST / evaluate the model
```

---

## 4. Step-by-Step: NLP Preprocessing Pipeline

> **File:** `src/preprocess.py` — function `preprocess_tweet(tweet)`

Raw tweets are messy. Before feeding text to a machine learning model, it must be cleaned and standardised. Here is every step applied:

---

### Step 1 — Remove URLs
```
"Check this out https://t.co/abc123"
               ↓  re.sub(r'https?://\S+', '', tweet)
"Check this out "
```
**Why?** URLs carry no sentiment signal. They waste vocabulary slots in the TF-IDF matrix.

---

### Step 2 — Remove @mentions
```
"@elonmusk this is great!"
               ↓  re.sub(r'@[A-Za-z0-9_]+', '', tweet)
" this is great!"
```
**Why?** Names of accounts don't indicate positive or negative sentiment.

---

### Step 3 — Remove Stock Tickers
```
"Bought $TSLA today"
               ↓  re.sub(r'\$\w+', '', tweet)
"Bought  today"
```
**Why?** Financial tickers are not meaningful sentiment words.

---

### Step 4 — Strip "RT" Retweet Prefix
```
"RT @user Great news everyone!"
               ↓  re.sub(r'^RT\s+', '', tweet)
"@user Great news everyone!"
```
**Why?** `RT` is a Twitter convention for "retweet" — not sentiment.

---

### Step 5 — Remove `#` Symbol (keep the word)
```
"#amazing product"
               ↓  re.sub(r'#', '', tweet)
"amazing product"
```
**Why?** The `#` is a platform symbol; the word behind it (e.g. "amazing") is the real sentiment word we want to keep.

---

### Step 6 — TweetTokenizer (NLTK)
> **NLP Technique: Twitter-aware Tokenisation**

```python
from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
tokens = tokenizer.tokenize("This is AMAZING!!!! <3")
# → ['this', 'is', 'amazin', '!', '<3']
```

`TweetTokenizer` is smarter than a generic word tokenizer because:
- `preserve_case=False` → lowercases everything ("Amazing" → "amazing")
- `strip_handles=True` → removes any leftover @handles
- `reduce_len=True` → collapses repeated chars ("sooooo" → "soo", "!!!!" → "!!")

This is critical for social media text where grammar is informal.

---

### Step 7 — Stop-Word Removal
> **NLP Technique: Stop-Word Filtering**

```
tokens = ['this', 'is', 'amazing', 'the', 'best']
stop_words = {'this', 'is', 'the', 'a', 'an', 'and', ... }  # 178 English words

result = ['amazing', 'best']  ← only meaningful words kept
```

**Why?** Stop words like "the", "is", "a", "and" appear in every sentence regardless of sentiment — they add noise but no signal. Removing them helps the model focus on opinion-bearing words.

---

### Step 8 — Porter Stemming
> **NLP Technique: Stemming**

```
"loving"    → "love"
"amazingly" → "amaz"
"happiness" → "happi"
"running"   → "run"
```

**Why?** The same root word appears in many forms. Without stemming, "love", "loved", "loving", "lovely" would be treated as 4 completely different features. Stemming reduces them all to "love", making the vocabulary more compact and the model more generalisable.

**Porter Stemmer** uses a rule-based algorithm (a series of suffix-stripping rules like "-ing → ''", "-ness → ''") — it is fast and works very well for English polarity classification.

---

### Full Preprocessing Example:

```
Input:   "I absolutely LOVE this new product! Check it out https://t.co/xyz #amazing @brand"

Step 1:  "I absolutely LOVE this new product! Check it out  #amazing @brand"   ← URL removed
Step 2:  "I absolutely LOVE this new product! Check it out  #amazing "         ← @brand removed
Step 5:  "I absolutely LOVE this new product! Check it out  amazing "           ← # removed
Step 6:  ['i', 'absolut', 'love', 'this', 'new', 'product', '!', 'check', 'it', 'out', 'amaz']
Step 7:  ['absolut', 'love', 'new', 'product', 'check', 'amaz']                ← stopwords gone
Step 8:  ['absolut', 'love', 'new', 'product', 'check', 'amaz']                ← stems applied

Output:  "absolut love new product check amaz"
```

---

## 5. Step-by-Step: Feature Extraction (TF-IDF)

> **NLP/ML Technique: TF-IDF Vectorisation**

Computers cannot work with raw strings — they need **numbers**. TF-IDF converts a collection of text documents into a matrix of numbers.

### What is TF-IDF?

**TF = Term Frequency** — how often a word appears in *this* tweet.  
**IDF = Inverse Document Frequency** — how rare the word is across *all* tweets.

```
TF-IDF(word, tweet) = TF × IDF
```

A word that appears often in *one tweet* but rarely across *all tweets* gets a **high score** — it is considered distinctive and informative. A word that appears in every tweet (like "the") gets a **low score** — it is not useful for classification.

### In Practice:

```python
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
```

- `max_features=5000` → keep only the 5,000 most important words in the vocabulary
- Each tweet becomes a **vector of 5,000 numbers** (most are 0 — called a sparse matrix)
- The vectorizer is `fit` on training data only (learns vocabulary from training) and `transform`ed on test/inference data

### Visualised:

```
                 "love"  "amaz"  "worst"  "great"  "terribl" ... (5000 words)
tweet_1 →  [  0.42,    0.31,    0.00,    0.00,    0.00,   ... ]
tweet_2 →  [  0.00,    0.00,    0.55,    0.00,    0.48,   ... ]
tweet_3 →  [  0.00,    0.00,    0.00,    0.60,    0.00,   ... ]
```

Each row is one tweet, each column is one word's TF-IDF score.

---

## 6. Step-by-Step: Machine Learning Model

> **ML Technique: Logistic Regression**

### Why Logistic Regression?

Logistic Regression is a **linear classifier** — it learns a weight (importance score) for each of the 5,000 TF-IDF word features. Words like "love", "amazing", "great" get **positive weights**. Words like "hate", "terrible", "worst" get **negative weights**.

```
score =  w₁ × tfidf("love")
       + w₂ × tfidf("amaz")
       + w₃ × tfidf("hate")
       + ... (5000 terms)
       + bias
```

Then it applies the **sigmoid function** to convert this raw score into a probability:

```
P(Positive) = sigmoid(score) = 1 / (1 + e^(-score))
P(Negative) = 1 - P(Positive)
```

### Training:

During training `model.fit(X_train_tfidf, y_train)` — the model adjusts the 5,000 weights to **minimise the error** on the training tweets using gradient descent with the `saga` solver (optimised for large sparse matrices).

### Why not a Neural Network?

For a **bag-of-words** representation (which TF-IDF is), Logistic Regression is extremely competitive. A neural network would require much more data, computation, and tuning. For a text classification mini-project with 10k tweets and TF-IDF features, Logistic Regression typically achieves 70–80% accuracy — on par with simple neural networks.

### Model Evaluation (on 2,000 held-out test tweets):

```
Test Accuracy: 74.25%

              Precision  Recall  F1-Score  Support
  Negative       0.72    0.78     0.75      988
  Positive       0.77    0.71     0.74     1012

  Macro Avg      0.74    0.74     0.74     2000
```

- **Precision 0.77 (Positive):** When the model says "Positive", it's right 77% of the time.
- **Recall 0.78 (Negative):** Of all actual negative tweets, the model catches 78%.
- **F1-Score:** Harmonic mean of precision and recall — a balanced metric.

---

## 7. Neutral Sentiment — A Special Design Choice

The training dataset only has Positive and Negative labels — there are **no neutral tweets**. So how do we detect neutral text?

### The Threshold Heuristic:

```python
NEUTRAL_THRESHOLD = 0.65

if max(P_positive, P_negative) < 0.65:
    sentiment = "Neutral"
elif P_positive > P_negative:
    sentiment = "Positive"
else:
    sentiment = "Negative"
```

**Intuition:** If the model is not confident enough to commit to either polarity (both probabilities are close to 50%), the text is likely factual/neutral — it just doesn't contain strong sentiment words in either direction.

For example: `"Going to the grocery store"` → after preprocessing → barely any opinion words remain → the model returns ~50%/50% → classified as **Neutral**.

---

## 8. The Web Application (app.py)

> **Framework: Streamlit**

Streamlit turns a Python script into an interactive web page with minimal boilerplate.

### Key Components:

| UI Element | What it does |
|---|---|
| **Text area** | User types/pastes a tweet |
| **Analyze button** | Triggers the inference pipeline |
| **Result card** | Glassmorphic card showing colour-coded sentiment + emoji |
| **Confidence score** | % shown inside the card (e.g. "Confidence Score: 84%") |
| **Probability bars** | Side-by-side progress bars for Positive vs. Negative probabilities |
| **History panel** | Session log of all analysed texts in this browser session |

### `@st.cache_resource` — Model Loading:
The model and vectorizer are loaded from `.pkl` files **only once** per session. This is critical — loading a model file on every button click would be very slow.

---

## 9. End-to-End Data Flow

```
USER submits: "I absolutely love this movie! It was fantastic 🎬"
    │
    ▼
preprocess_tweet()
    │ Remove URLs / mentions / hashtags
    │ TweetTokenizer → ['i', 'absolut', 'love', 'movi', 'fantast']
    │ Remove stop words → ['absolut', 'love', 'movi', 'fantast']
    │ Porter Stem → ['absolut', 'love', 'movi', 'fantast']
    │ Join → "absolut love movi fantast"
    ▼
vectorizer.transform(["absolut love movi fantast"])
    │ Looks up each word in the 5,000-word TF-IDF vocabulary
    │ Output: sparse vector [0, 0, 0.42, 0, 0, 0.38, 0, ...]
    ▼
model.predict_proba(vector)
    │ Applies learned weights
    │ Output: [0.08, 0.92]  →  P(Negative)=8%, P(Positive)=92%
    ▼
Threshold check: max(0.08, 0.92) = 0.92  ≥  0.65
    │ prediction = 1 (Positive)
    ▼
UI renders:
    ┌──────────────────────────────┐
    │   Detected Sentiment:        │
    │   🤩  Positive               │
    │   Confidence Score: 92%      │
    └──────────────────────────────┘
    🟢 Positive: 92%  ████████████░░░
    🔴 Negative:  8%  █░░░░░░░░░░░░░░
```

---

## 10. Glossary of All NLP/ML Techniques Used

| Technique | Where Used | What It Does |
|---|---|---|
| **Regex cleaning** | `preprocess.py` | Pattern-based removal of URLs, mentions, tickers, RT prefix |
| **Tokenisation** | `preprocess.py` | Splits a sentence into individual word units (tokens) |
| **TweetTokenizer** | `preprocess.py` | Twitter-specific tokeniser: handles slang, emoticons, repeated chars |
| **Stop-word removal** | `preprocess.py` | Filters out uninformative common words (the, is, a, …) |
| **Stemming (Porter)** | `preprocess.py` | Reduces words to their root form (loving → love) |
| **TF-IDF Vectorisation** | `train.py` | Converts text into a numeric matrix weighted by word importance |
| **Sparse matrix representation** | `train.py` (sklearn) | Efficiently stores TF-IDF vectors where most values are 0 |
| **Train/Test Split** | `train.py` | Holds out 20% of data to evaluate generalisation |
| **Logistic Regression** | `train.py` | Linear classifier that learns word weights for polarity prediction |
| **SAGA Solver** | `train.py` | Stochastic Average Gradient solver, optimal for large sparse data |
| **Sigmoid / Probability Calibration** | `model.predict_proba` | Converts raw scores to calibrated class probabilities 0–1 |
| **Confidence Threshold (Neutral)** | `app.py` | Heuristic to infer Neutral when model is uncertain (< 65%) |
| **Model Serialisation (joblib)** | `train.py` / `app.py` | Saves trained model to disk; loads it without retraining |
| **Caching (`@st.cache_resource`)** | `app.py` | Loads model once per session for fast repeated inference |
| **Session State** | `app.py` | Persists analysis history across button clicks in Streamlit |
