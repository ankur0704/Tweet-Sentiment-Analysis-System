"""
app.py
------
Streamlit web application for Tweet Sentiment Analysis.

Loads the pre-trained model artefacts and exposes an interactive interface
for users to classify any piece of text as Positive, Negative, or Neutral.

Run with:
    streamlit run app.py
"""
import os
import sys
import joblib
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from preprocess import preprocess_tweet

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Tweet Sentiment AI",
    page_icon="✨",
    layout="centered"
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark glassmorphic design
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0d1117, #161b22);
        color: #c9d1d9;
    }
    .header-style {
        text-align: center;
        background: -webkit-linear-gradient(45deg, #ff7b00, #ff007b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .subheader-style {
        text-align: center;
        font-size: 1.1rem;
        color: #8b949e;
        margin-bottom: 2rem;
    }
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        margin-top: 2rem;
        transition: transform 0.3s;
    }
    .result-card:hover { transform: translateY(-5px); }
    .sentiment-positive { color: #2ea043; font-size: 2rem; font-weight: bold; }
    .sentiment-negative { color: #f85149; font-size: 2rem; font-weight: bold; }
    .sentiment-neutral  { color: #58a6ff; font-size: 2rem; font-weight: bold; }
    .stTextArea textarea {
        background-color: rgba(0,0,0,0.2) !important;
        color: white !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        font-size: 1.1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 1px #58a6ff !important;
    }
    .stButton button {
        background: linear-gradient(90deg, #238636, #2ea043) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .stButton button:hover {
        background: linear-gradient(90deg, #2ea043, #3fb950) !important;
        box-shadow: 0 4px 15px rgba(46, 160, 67, 0.4) !important;
    }
    /* history table styling */
    .history-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.4rem;
        background: rgba(255,255,255,0.04);
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<h1 class="header-style">Tweet Sentiment AI</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subheader-style">Analyze the emotion behind social media text using Machine Learning</p>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# Model loading (cached — runs only once per session)
# ---------------------------------------------------------------------------

# Neutral threshold: if max probability < NEUTRAL_THRESHOLD, classify as Neutral
NEUTRAL_THRESHOLD = 0.65

@st.cache_resource
def load_models():
    """Load the saved model and vectorizer from the models/ directory."""
    base = os.path.dirname(os.path.abspath(__file__))
    model_path      = os.path.join(base, 'models', 'sentiment_model.pkl')
    vectorizer_path = os.path.join(base, 'models', 'tfidf_vectorizer.pkl')
    model      = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

try:
    model, vectorizer = load_models()
except FileNotFoundError:
    st.error("⚠️ Model files not found. Please run `python src/train.py` first.")
    st.stop()

# ---------------------------------------------------------------------------
# Session state — analysis history
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts: {text, sentiment, confidence}


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

_SENTIMENT_META = {
    "Positive": dict(css="sentiment-positive", emoji="🤩"),
    "Negative": dict(css="sentiment-negative", emoji="😠"),
    "Neutral":  dict(css="sentiment-neutral",  emoji="😐"),
}

def predict_sentiment(text: str) -> dict:
    """
    Run the full inference pipeline on raw text.

    Returns a dict with keys: sentiment, emoji, css, prob_pos, prob_neg, confidence.
    """
    clean = preprocess_tweet(text)
    features      = vectorizer.transform([clean])
    prediction    = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    prob_neg, prob_pos = float(probabilities[0]), float(probabilities[1])
    max_prob            = max(prob_neg, prob_pos)

    if max_prob < NEUTRAL_THRESHOLD:
        sentiment  = "Neutral"
        confidence = max_prob
    elif prediction == 1:
        sentiment  = "Positive"
        confidence = prob_pos
    else:
        sentiment  = "Negative"
        confidence = prob_neg

    meta = _SENTIMENT_META[sentiment]
    return dict(
        sentiment=sentiment, confidence=confidence,
        prob_pos=prob_pos, prob_neg=prob_neg,
        **meta
    )


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

tweet_input = st.text_area(
    "✍️ Paste a tweet or sentence here...",
    height=150,
    placeholder="e.g. Just bought the new product and it is absolutely amazing!"
)

if st.button("Analyze Sentiment ✨"):
    if not tweet_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing semantics..."):
            result = predict_sentiment(tweet_input)

        # --- Result card ---
        st.markdown(f"""
        <div class="result-card">
            <p style="color:#8b949e; font-size:1.2rem; margin-bottom:0.5rem;">Detected Sentiment:</p>
            <div class="{result['css']}">{result['emoji']} {result['sentiment']}</div>
            <p style="color:#8b949e; margin-top:1rem; font-size:0.9rem;">
                Confidence Score: {result['confidence']:.1%}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # --- Probability distribution bars ---
        st.markdown("<br><p style='text-align:center; color:#8b949e;'>Probability Distribution</p>",
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"🟢 **Positive:** {result['prob_pos']:.1%}")
            st.progress(result['prob_pos'])
        with col2:
            st.write(f"🔴 **Negative:** {result['prob_neg']:.1%}")
            st.progress(result['prob_neg'])

        # --- Save to history ---
        st.session_state.history.insert(0, {
            "text":       tweet_input[:80] + ("…" if len(tweet_input) > 80 else ""),
            "sentiment":  result['sentiment'],
            "confidence": result['confidence'],
            "emoji":      result['emoji'],
        })

# ---------------------------------------------------------------------------
# Analysis history panel
# ---------------------------------------------------------------------------
if st.session_state.history:
    st.divider()
    col_title, col_clear = st.columns([4, 1])
    with col_title:
        st.markdown("### 🕑 Analysis History")
    with col_clear:
        if st.button("Clear", key="clear_history"):
            st.session_state.history = []
            st.rerun()

    for entry in st.session_state.history:
        badge_color = {"Positive": "#2ea043", "Negative": "#f85149", "Neutral": "#58a6ff"}.get(
            entry["sentiment"], "#8b949e"
        )
        st.markdown(f"""
        <div class="history-row">
            <span style="flex:1; color:#c9d1d9; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                {entry['text']}
            </span>
            <span style="margin-left:1rem; color:{badge_color}; font-weight:600; white-space:nowrap;">
                {entry['emoji']} {entry['sentiment']} ({entry['confidence']:.0%})
            </span>
        </div>
        """, unsafe_allow_html=True)
