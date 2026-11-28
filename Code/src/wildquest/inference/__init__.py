"""Inference interfaces for WildQuest models."""

from .classifier import ClassificationResult, EfficientNetClassifier, SpeciesPrediction
from .pipeline import AnimalPrediction, InferenceError, InferencePipeline, PhotoInferenceResult

__all__ = [
    "AnimalPrediction",
    "ClassificationResult",
    "EfficientNetClassifier",
    "InferenceError",
    "InferencePipeline",
    "PhotoInferenceResult",
    "SpeciesPrediction",
]
