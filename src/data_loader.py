import pandas as pd
import re
from pathlib import Path

def clean_text(text):
    """Remove HTML tags and extra whitespace."""
    text = re.sub(r"<br\s*/?>", " ", text)
    return text.strip()

def load_data(path):
    """Load IMDB reviews dataset, semicolon-separated."""
    path = Path(path).resolve()
    df = pd.read_csv(path, sep=";", encoding="latin-1")
    df["review"] = df["review"].apply(clean_text)
    df["sentiment"] = df["sentiment"].str.lower()
    return df