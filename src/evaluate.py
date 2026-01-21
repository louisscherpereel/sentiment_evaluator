import time
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def benchmark_model(model, texts, true_labels):
    """
    Benchmark a sentiment model using multiple evaluation metrics.
    """

    # Run inference
    start_time = time.time()
    predictions = model.predict(texts)
    pred_labels = [p[0] for p in predictions]
    runtime = time.time() - start_time

    # Metrics
    metrics = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels, pos_label="positive"),
        "recall": recall_score(true_labels, pred_labels, pos_label="positive"),
        "f1": f1_score(true_labels, pred_labels, pos_label="positive"),
        "confusion_matrix": confusion_matrix(
            true_labels,
            pred_labels,
            labels=["positive", "negative"]
        ),
        "runtime_seconds": runtime,
        "samples_per_second": len(texts) / max(runtime, 1e-8)
    }

    return {
        "metrics": metrics,
        "predictions": predictions
    }