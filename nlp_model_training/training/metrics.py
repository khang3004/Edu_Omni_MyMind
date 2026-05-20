import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """
    Computes standard classification evaluation metrics.

    Args:
        logits (np.ndarray): Output predictions from the model (shape: num_samples, num_labels).
        labels (np.ndarray): True target class indices (shape: num_samples).

    Returns:
        Dict[str, float]: Dictionary of calculated metrics.
    """
    # Obtain predicted class indices by taking the argmax along the last dimension
    predictions = np.argmax(logits, axis=-1)

    # Calculate global accuracy score
    accuracy = accuracy_score(labels, predictions)

    # Calculate precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    # Also get macro-F1 (extremely common in NLP benchmarks for class imbalance)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
    }
