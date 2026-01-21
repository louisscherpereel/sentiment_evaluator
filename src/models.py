import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")

class VaderModel:
    """Rule-based sentiment model (VADER)."""

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def predict(self, texts):
        results = []
        for text in texts:
            score = self.analyzer.polarity_scores(text)["compound"]
            label = "positive" if score >= 0.0 else "negative"
            results.append((label, score))
        return results
    

class DistilbertModel:
    """Transformer-based sentiment model (DistilBERT)."""

    def __init__(self):
        self.pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # Use CPU
        )

    def predict(self, texts):
        outputs = self.pipeline(texts, truncation=True, max_length=512)
        return [(o["label"].lower(), o["score"]) for o in outputs]
