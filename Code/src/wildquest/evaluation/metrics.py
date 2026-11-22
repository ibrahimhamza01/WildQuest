"""Classification metrics from saved predictions or a confusion matrix."""

from dataclasses import dataclass
from math import fsum

from .predictions import load_species, read_predictions


def confusion_metrics(confusion):
    """Compute macro metrics and row-wise per-class metrics from a square confusion matrix.

    Rows are true class IDs and columns are predicted class IDs.
    All matrix classes contribute to macro averages. Undefined ratios are zero.
    """
    n = len(confusion)
    if n == 0 or any(len(row) != n for row in confusion):
        raise ValueError("Confusion matrix must be nonempty and square")
    total = sum(map(sum, confusion))
    if total == 0:
        raise ValueError("Cannot calculate metrics on an empty prediction set")
    per_class = []
    for i in range(n):
        true_positive = confusion[i][i]
        support = sum(confusion[i])
        predicted_count = sum(confusion[r][i] for r in range(n))
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({
            "class_id": i,
            "support": support,
            "predicted_count": predicted_count,
            "correct": true_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    diagonal = sum(confusion[i][i] for i in range(n))
    return {
        "top1_accuracy": diagonal / total,
        "macro_precision": sum(v["precision"] for v in per_class) / n,
        "macro_recall": sum(v["recall"] for v in per_class) / n,
        "macro_f1": sum(v["f1"] for v in per_class) / n,
        "correct": diagonal,
        "total": total,
    }, per_class


@dataclass
class EvaluationResult:
    """Metrics and counts for the configured species, ordered by class ID."""

    species: tuple[str, ...]
    metrics: dict
    per_class: list[dict]
    confusion_matrix: list[list[int]]
    confidence_analysis: dict | None


def evaluate_predictions(path):
    """Evaluate a prediction CSV with the frozen WildQuest class mapping."""
    species = load_species()
    class_ids = range(len(species))
    confusion = [[0 for _ in class_ids] for _ in class_ids]
    has_probabilities = False
    top3_correct = 0
    confidence_source = None
    confidence_groups = {True: [], False: []}

    for true_id, pred_id, confidence, probabilities in read_predictions(path, len(species)):
        confusion[true_id][pred_id] += 1
        if probabilities is not None:
            has_probabilities = True
            # Keep the exported prediction first among tied maxima, then use ID order.
            ranked = sorted(class_ids, key=lambda i: (probabilities[i], i == pred_id, -i),
                            reverse=True)
            top3_correct += true_id in ranked[:3]
        if confidence is not None:
            confidence_source = "confidence"
        elif probabilities is not None:
            confidence_source = "predicted_class_probability"
            confidence = probabilities[pred_id]
        if confidence is not None:
            confidence_groups[true_id == pred_id].append(confidence)

    metrics, per_class = confusion_metrics(confusion)
    if has_probabilities:
        metrics["top3_accuracy"] = top3_correct / metrics["total"]

    confidence_analysis = None
    if confidence_source is not None:
        confidence_analysis = {"source": confidence_source}
        for label, values in (("correct", confidence_groups[True]),
                              ("incorrect", confidence_groups[False])):
            confidence_analysis[label] = {
                "count": len(values),
                "mean_confidence": fsum(values) / len(values) if values else None,
            }
    return EvaluationResult(species, metrics, per_class, confusion, confidence_analysis)
