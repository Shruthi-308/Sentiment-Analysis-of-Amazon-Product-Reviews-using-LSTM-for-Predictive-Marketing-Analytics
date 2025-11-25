# app.py
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import get_custom_objects
import requests

# ----------------- CONFIG -----------------
MAX_LEN = 100  # must match what you used during training
MODEL_FILES = ["lstm_sentiment_model.keras", "lstm_sentiment_model.h5"]  # order: prefer .keras
TOKENIZER_FILE = "tokenizer.pickle"
LABELENC_FILE = "label_encoder.pickle"

# If you keep models in repo, you don't need these. If you want to download at runtime
# set True and replace the URLs below with raw GitHub or direct download links.
AUTO_DOWNLOAD_IF_MISSING = False
DOWNLOAD_URLS = {
    "lstm_sentiment_model.keras": "https://raw.githubusercontent.com/<USER>/<REPO>/main/lstm_sentiment_model.keras",
    "lstm_sentiment_model.h5": "https://raw.githubusercontent.com/<USER>/<REPO>/main/lstm_sentiment_model.h5",
    "tokenizer.pickle": "https://raw.githubusercontent.com/<USER>/<REPO>/main/tokenizer.pickle",
    "label_encoder.pickle": "https://raw.githubusercontent.com/<USER>/<REPO>/main/label_encoder.pickle",
}
# ------------------------------------------

st.set_page_config(page_title="Amazon Sentiment (LSTM)", layout="centered")

st.title("🔎 Sentiment Analysis — Amazon Reviews (LSTM)")
st.write("Type a review or upload a CSV to get predictions. CSV should contain a text column (default `Product_Review`).")

# ---------- Helper: download file if missing ----------
def download_file(url: str, out_path: str):
    """Download a file streaming (simple)."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return out_path

def ensure_files_present():
    """Ensure model/tokenizer/label encoder are present. Downloads if configured and missing."""
    missing = []
    # check model
    found_model = None
    for fname in MODEL_FILES:
        if Path(fname).exists():
            found_model = fname
            break
    if not found_model:
        missing.append(MODEL_FILES[0])  # prefer first
    if not Path(TOKENIZER_FILE).exists():
        missing.append(TOKENIZER_FILE)
    if not Path(LABELENC_FILE).exists():
        missing.append(LABELENC_FILE)

    if missing and AUTO_DOWNLOAD_IF_MISSING:
        st.info("Some model artifacts are missing on disk — attempting to download...")
        for m in missing:
            url = DOWNLOAD_URLS.get(m)
            if not url:
                st.error(f"No download URL configured for {m}. Please add it in app.py.")
                continue
            try:
                with st.spinner(f"Downloading {m} ..."):
                    download_file(url, m)
                    st.success(f"Downloaded {m}")
            except Exception as e:
                st.error(f"Failed to download {m}: {e}")
    # After attempted download, re-check model file
    found_model = None
    for fname in MODEL_FILES:
        if Path(fname).exists():
            found_model = fname
            break
    return found_model is not None

# ---------- Load artifacts with caching ----------
@st.cache_resource
def load_artifacts():
    # ensure presence (may download if configured)
    ok = ensure_files_present()
    if not ok:
        raise FileNotFoundError(
            "Model or artifacts not found. Put model and pickles in the repo or enable AUTO_DOWNLOAD_IF_MISSING."
        )

    # choose model file
    model_path = None
    for fname in MODEL_FILES:
        if Path(fname).exists():
            model_path = fname
            break

    if model_path is None:
        raise FileNotFoundError("Model file not found. Expected one of: " + ", ".join(MODEL_FILES))

    # load model
    model = load_model(model_path)

    # load tokenizer
    with open(TOKENIZER_FILE, "rb") as f:
        tokenizer = pickle.load(f)

    # load label encoder
    with open(LABELENC_FILE, "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder

def preprocess_texts(texts, tokenizer, maxlen=MAX_LEN):
    seq = tokenizer.texts_to_sequences(texts)
    pad = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
    return pad

def predict_sentiment(texts, model, tokenizer, label_encoder):
    X = preprocess_texts(texts, tokenizer)
    probs = model.predict(X, verbose=0)
    preds = np.argmax(probs, axis=1)
    labels = label_encoder.inverse_transform(preds)
    confidences = np.max(probs, axis=1)
    return labels, confidences, probs

# ---------- Main UI ----------
# Try to load artifacts
try:
    with st.spinner("Loading model and artifacts..."):
        model, tokenizer, label_encoder = load_artifacts()
    st.success("Model and artifacts loaded.")
except Exception as e:
    st.error(f"Could not load model/artifacts: {e}")
    st.stop()

# Sidebar info
st.sidebar.header("Info")
st.sidebar.write(f"Model file loaded: **{model.name if hasattr(model,'name') else 'Keras model'}**")
st.sidebar.write(f"Max tokens (padding length): **{MAX_LEN}**")
st.sidebar.write("If your model files are not in the repo, set `AUTO_DOWNLOAD_IF_MISSING=True` and add raw URLs in `DOWNLOAD_URLS`.")

# Single prediction UI
st.header("Single review prediction")
col1, col2 = st.columns([3,1])
with col1:
    user_text = st.text_area("Enter product review text", height=140, value="")
with col2:
    st.write(" ")
    if st.button("Predict single"):
        if not user_text.strip():
            st.warning("Please enter a review text first.")
        else:
            with st.spinner("Predicting..."):
                labels, confidences, probs = predict_sentiment([user_text], model, tokenizer, label_encoder)
            lab = labels[0]
            conf = confidences[0]
            # badge like UI
            if lab.lower().startswith("positive"):
                st.success(f"✅ Predicted: **{lab}**  — Confidence: {conf:.2f}")
            elif lab.lower().startswith("neutral"):
                st.info(f"ℹ️ Predicted: **{lab}**  — Confidence: {conf:.2f}")
            else:
                st.error(f"❌ Predicted: **{lab}**  — Confidence: {conf:.2f}")

            # show probability distribution as dataframe and bar chart
            prob_df = pd.DataFrame(probs[0].reshape(1,-1), columns=label_encoder.classes_).T
            prob_df.columns = ["Probability"]
            st.bar_chart(prob_df)

st.markdown("---")

# Batch prediction
st.header("Batch prediction (CSV)")
st.write("Upload a CSV with a text column. Default column name: `Product_Review`.")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
col_name = st.text_input("Column name containing reviews", value="Product_Review")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} rows.")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        df = None

    if df is not None:
        if col_name not in df.columns:
            st.error(f"Column '{col_name}' not found. Available: {list(df.columns)}")
        else:
            if st.button("Run batch predictions"):
                texts = df[col_name].astype(str).tolist()
                with st.spinner("Predicting batch..."):
                    labels, confidences, probs = predict_sentiment(texts, model, tokenizer, label_encoder)
                df["Predicted_Sentiment"] = labels
                df["Confidence"] = confidences
                st.success("Batch predictions finished — showing top rows:")
                st.dataframe(df[[col_name, "Predicted_Sentiment", "Confidence"]].head(10))

                # download button
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV with predictions", data=csv, file_name="predictions.csv", mime="text/csv")

                # class distribution chart
                dist = df["Predicted_Sentiment"].value_counts().reindex(label_encoder.classes_, fill_value=0)
                st.subheader("Predicted class distribution")
                st.bar_chart(dist)

st.markdown("---")
st.write("Demo sample predictions:")
demo_reviews = [
    "This product is excellent and works perfectly!",
    "Battery life is very poor and disappointing.",
    "It's okay, not too good and not too bad."
]
if st.button("Show demo predictions"):
    labels, confidences, probs = predict_sentiment(demo_reviews, model, tokenizer, label_encoder)
    demo_df = pd.DataFrame({
        "Review": demo_reviews,
        "Predicted_Sentiment": labels,
        "Confidence": np.round(confidences, 3)
    })
    st.table(demo_df)

st.markdown("---")
st.write("If you see errors during deploy: check that your model and pickles are in the repo root, or enable downloading by setting `AUTO_DOWNLOAD_IF_MISSING=True` and populating `DOWNLOAD_URLS` with raw file links (GitHub raw links work for small files).")
