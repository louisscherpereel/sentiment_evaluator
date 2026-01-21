from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from src.models import VaderModel, DistilbertModel

app = FastAPI(
    title="Sentiment Evaluator API",
    description="API for sentiment analysis using a pre-trained transformer model"
)

MODELS = {
    "vader": VaderModel(),
    "distilbert": DistilbertModel()
}
DEFAULT_MODEL = "distilbert"

class ReviewRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    sentiment: str
    score: float
    model_used: str


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: ReviewRequest, model_name: str = Query(DEFAULT_MODEL, description="Model to use for prediction")):
    """
    Predict sentiment for a single review.
    """

    if model_name not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not supported. Choose from {list(MODELS.keys())}"
        )
    model = MODELS[model_name]
    label, score = model.predict([request.text])[0]

    return {
        "sentiment": label,
        "score": score,
        "model_used": model_name
    }