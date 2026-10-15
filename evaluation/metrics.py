import numpy as np


def calculate_statistics(scores):
    return {
        "samples": len(scores),
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "p1": float(np.percentile(scores, 1)),
        "p5": float(np.percentile(scores, 5)),
        "p10": float(np.percentile(scores, 10)),
    }


def calculate_thresholds(validation_scores):
    return {
        "1%": float(np.percentile(validation_scores, 1)),
        "5%": float(np.percentile(validation_scores, 5)),
        "10%": float(np.percentile(validation_scores, 10)),
    }


def calculate_anomaly_metrics(probabilities, threshold):
    anomalies = probabilities < threshold

    return {
        "samples": len(probabilities),
        "anomalies": int(anomalies.sum()),
        "normal": int((~anomalies).sum()),
        "anomaly_rate": float(np.mean(anomalies)),
    }


def calculate_temporal_statistics(
    timestamps,
    probabilities,
    injection_interval
):
    if injection_interval is None:
        return None

    injection_start, injection_end = injection_interval

    before = probabilities[
        timestamps < injection_start
    ]

    during = probabilities[
        (timestamps >= injection_start) &
        (timestamps <= injection_end)
    ]

    after = probabilities[
        timestamps > injection_end
    ]

    return {
        "injection_start": injection_start,
        "injection_end": injection_end,
        "BEFORE": calculate_statistics(before),
        "DURING": calculate_statistics(during),
        "AFTER": calculate_statistics(after),
    }

def calculate_classification_metrics(
    timestamps,
    probabilities,
    injection_interval,
    threshold
):
    if injection_interval is None:
        return None

    injection_start, injection_end = injection_interval

    # Ground truth:
    # 1 = attack, 0 = benign
    y_true = (
        (timestamps >= injection_start) &
        (timestamps <= injection_end)
    )

    # Prediction:
    # 1 = anomaly, 0 = normal
    y_pred = probabilities < threshold

    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    total = tp + tn + fp + fn

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    accuracy = (tp + tn) / total if total else 0.0
    balanced_accuracy = (recall + specificity) / 2

    return {
        "samples": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "fpr": fpr,
        "tpr": recall,
        "balanced_accuracy": balanced_accuracy,
    }
