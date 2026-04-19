"""
preprocess.py
-------------
NLP preprocessing pipeline for tweet text.

Provides a single public function, `preprocess_tweet`, which cleans raw
tweet strings into stemmed, normalised tokens ready for feature extraction.

Steps applied:
    1. Remove URLs (http/https)
    2. Remove @mentions and stock-market tickers ($AAPL)
    3. Strip legacy "RT" retweet prefix
    4. Strip '#' symbol while keeping the word
    5. Tokenise with NLTK TweetTokenizer (handles emoticons & slang)
    6. Remove English stop words and punctuation
    7. Apply Porter Stemming
"""
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

__all__ = ["preprocess_tweet"]

# ---------------------------------------------------------------------------
# NLTK data bootstrap
# ---------------------------------------------------------------------------

def _download_nltk_data() -> None:
    """Download required NLTK corpora/models if not already present."""
    resources = {
        'corpora/stopwords':       'stopwords',
        'tokenizers/punkt':        'punkt',
        'tokenizers/punkt_tab':    'punkt_tab',
        'corpora/twitter_samples': 'twitter_samples',
    }
    for path, pkg in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

_download_nltk_data()

# ---------------------------------------------------------------------------
# Module-level singletons (instantiated once, reused on every call)
# ---------------------------------------------------------------------------

_TOKENIZER   = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
_STOP_WORDS  = set(stopwords.words('english'))
_STEMMER     = PorterStemmer()
_PUNCTUATION = set(string.punctuation)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_tweet(tweet: str) -> str:
    """
    Clean and normalise a raw tweet string.

    Parameters
    ----------
    tweet : str
        Raw tweet text (may include URLs, mentions, hashtags, etc.).

    Returns
    -------
    str
        Space-joined string of stemmed, filtered tokens. Returns an empty
        string when no meaningful tokens remain after cleaning.

    Examples
    --------
    >>> preprocess_tweet("I love @twitter! Check https://t.co/xyz #NLP")
    'love check nlp'
    """
    if not isinstance(tweet, str) or not tweet.strip():
        return ""

    # 1. Remove URLs
    tweet = re.sub(r'https?://\S+', '', tweet)
    # 2. Remove @mentions
    tweet = re.sub(r'@[A-Za-z0-9_]+', '', tweet)
    # 3. Remove stock-market tickers ($AAPL)
    tweet = re.sub(r'\$\w+', '', tweet)
    # 4. Strip legacy retweet prefix (must be at start of string)
    tweet = re.sub(r'^RT\s+', '', tweet)
    # 5. Remove '#' symbol but keep the word
    tweet = re.sub(r'#', '', tweet)

    # 6. Tokenise using TweetTokenizer
    tokens = _TOKENIZER.tokenize(tweet)

    # 7. Filter stop words / punctuation, then stem
    clean_tokens = [
        _STEMMER.stem(token)
        for token in tokens
        if token not in _STOP_WORDS and token not in _PUNCTUATION
    ]

    return ' '.join(clean_tokens)
