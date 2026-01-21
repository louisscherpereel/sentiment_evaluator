import streamlit as st
from models import (VaderModel,DistilbertModel)

st.set_page_config(page_title="Sentiment Evaluator", layout="centered")
st.title("Sentiment Evaluator")
st.write("Analyze the sentiment of a movie review using different NLP models.")


# Models
@st.cache_resource
def load_models():
    return {"VADER": VaderModel(),"DistilBERT": DistilbertModel()}

models = load_models()


# UI
review_text = st.text_area(
    "Enter a movie review:",
    height=200,
    placeholder="Paste a movie review here..."
)

model_name = st.selectbox(
    "Choose a sentiment model:",
    list(models.keys())
)

analyze = st.button("Analyze sentiment")


# Prediction
if analyze:
    if not review_text.strip():
        st.warning("Please enter some text.")
    else:
        model = models[model_name]
        label, score = model.predict([review_text])[0]

        st.subheader("Result")
        st.write(f"**Sentiment:** {label.capitalize()}")
        st.write(f"**Confidence score:** {score:.3f}")

        if label == "positive":
            st.success("Positive sentiment")
        else:
            st.error("Negative sentiment")