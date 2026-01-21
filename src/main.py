from pathlib import Path
from src.data_loader import load_data
from src.models import VaderModel, DistilbertModel
from src.evaluate import benchmark_model

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data/IMDB-movie-reviews.csv"
OUTPUT_PATH = BASE_DIR / "output/predictions.csv"

def main():
    df = load_data(DATA_PATH)
    texts = df["review"].tolist()
    true_labels = df["sentiment"].tolist()

    vader = VaderModel()
    distilbert = DistilbertModel()

    # Benchmark models
    vader_results = benchmark_model(vader, texts, true_labels)
    distilbert_results = benchmark_model(distilbert, texts, true_labels)

    # Print results
    print(f"{'Model':<10} {'Accuracy':<10} {'Recall':<10} {'Precision':<10} {'F1-Score':<10} {'Runtime (s)':<12}")

    vader_metrics = vader_results["metrics"]
    print(f"{'VADER':<10} {vader_metrics['accuracy']:<10.3f} {vader_metrics['recall']:<10.3f} "
        f"{vader_metrics['precision']:<10.3f} {vader_metrics['f1']:<10.3f} {vader_metrics['runtime_seconds']:<12.3f}")
    
    distilbert_metrics = distilbert_results["metrics"]
    print(f"{'DistilBERT':<10} {distilbert_metrics['accuracy']:<10.3f} {distilbert_metrics['recall']:<10.3f} "
        f"{distilbert_metrics['precision']:<10.3f} {distilbert_metrics['f1']:<10.3f} {distilbert_metrics['runtime_seconds']:<12.3f}")

    # Save predictions from selected model (distilbert)
    df["predicted_sentiment"] = [p[0] for p in distilbert_results["predictions"]]
    df["sentiment_score"] = [p[1] for p in distilbert_results["predictions"]]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Predictions saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()