# app.py (temporary debug — paste this whole file)
import streamlit as st
import sys, traceback, importlib
from pathlib import Path
import pickle

st.set_page_config(page_title="DEBUG: Sentiment App", layout="wide")
st.title("🔧 Debug mode — show error & environment info")

# Python + important package versions
st.subheader("Python & versions")
st.write(sys.version)
for pkg in ("streamlit","tensorflow","numpy","pandas","sklearn","requests"):
    try:
        m = importlib.import_module(pkg)
        st.write(f"- {pkg}: {getattr(m,'__version__','(no __version__)')}")
    except Exception as e:
        st.write(f"- {pkg}: NOT INSTALLED ({e})")

# list files in repo root
st.subheader("Files in repo (top-level)")
root = Path.cwd()
files = [p.name for p in root.iterdir()]
st.write(files)

# Check expected artifact files
st.subheader("Check artifact files")
expected = ["lstm_sentiment_model.keras", "lstm_sentiment_model.h5", "tokenizer.pickle", "label_encoder.pickle"]
for e in expected:
    p = Path(e)
    st.write(f"{e}: {'FOUND' if p.exists() else 'MISSING'}; size: {p.stat().st_size if p.exists() else 'N/A'}")

# Try to load artifacts and show full traceback
st.subheader("Attempt to load model + pickles (click button)")
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def attempt_load():
    try:
        # find a model file
        model_file = None
        for cand in ["lstm_sentiment_model.keras","lstm_sentiment_model.h5"]:
            if Path(cand).exists():
                model_file = cand
                break
        if not model_file:
            raise FileNotFoundError("No model file found among: lstm_sentiment_model.keras / .h5")

        st.write(f"Trying to load model: {model_file}")
        if model_file.endswith(".keras") or model_file.endswith(".h5"):
            m = load_model(model_file)
            st.write("Model loaded - showing summary (may be truncated):")
            try:
                m.summary()
                st.write("Model summary printed to logs (check Streamlit logs if not visible).")
            except Exception:
                st.write("Model loaded but summary() raised (that's fine).")
        # tokenizer
        if not Path("tokenizer.pickle").exists():
            raise FileNotFoundError("tokenizer.pickle missing")
        with open("tokenizer.pickle","rb") as f:
            tok = pickle.load(f)
        st.write("Tokenizer loaded. Example word_index size:", len(getattr(tok,"word_index",{})))
        # label encoder
        if not Path("label_encoder.pickle").exists():
            raise FileNotFoundError("label_encoder.pickle missing")
        with open("label_encoder.pickle","rb") as f:
            le = pickle.load(f)
        st.write("Label encoder loaded. Classes:", getattr(le,"classes_", "unknown"))
        st.success("All artifacts loaded successfully.")
    except Exception:
        tb = traceback.format_exc()
        st.error("Exception occurred — full traceback below:")
        st.code(tb, language="text")
        # save for convenience
        Path("last_traceback.txt").write_text(tb)
        st.write("Traceback saved to last_traceback.txt")

if st.button("Run artifact loader now"):
    attempt_load()

st.markdown("---")
st.write("When you see the traceback, copy & paste it here. I will give the exact single change to fix the app.")
