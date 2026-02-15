import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# NLTK SETUP

nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)

nltk.data.path.append(nltk_data_path)

# Ensure required tokenizer + stopwords resources
required_resources = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
]

for path, name in required_resources:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(name, download_dir=nltk_data_path)

# GLOBAL OBJECTS

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))


# TEXT PREPROCESSING

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    filtered = []
    for word in tokens:
        if word.isalnum() and word not in stop_words:
            filtered.append(ps.stem(word))

    return " ".join(filtered)

# LOAD MODEL & VECTORIZER

with open("vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# STREAMLIT UI

st.title("Spam SMS Detector")

input_sms = st.text_area("Enter your message")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message")
    else:
        transformed = transform_text(input_sms)
        vector_input = tfidf.transform([transformed])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 Message is Spam")
        else:
            st.success("Message is Not Spam")

