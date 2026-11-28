"""Run animal detection and crop classification on a single photograph."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

from PIL import Image

from wildquest.detection import Detection, crop_animal, prepare_image
from wildquest.detection.crops import ImageSource, is_valid_pixel_box

from .classifier import ClassificationResult

InferenceStatus = Literal["ok", "no_detection", "partial", "error"]
InferenceStage = Literal["input", "detector", "crop", "classifier"]


class _Detector(Protocol):
    def detect(self, image: Image.Image) -> list[Detection]: ...


class _Classifier(Protocol):
    def predict(self, crop: Image.Image) -> ClassificationResult: ...


@dataclass(frozen=True)
class AnimalPrediction:
    """The detector record and species predictions for the same crop."""

    detection: Detection
    classification: ClassificationResult

    def to_dict(self) -> dict:
        """Return the animal's box and scores as JSON-compatible values."""
        primary = self.classification.primary
        return {
            "box_xyxy": list(self.detection.box),
            "category": self.detection.category,
            "detector_confidence": self.detection.confidence,
            "class_id": primary.class_id,
            "scientific_name": primary.scientific_name,
            "classification_confidence": primary.confidence,
            "top3": [asdict(item) for item in self.classification.top3],
        }


@dataclass(frozen=True)
class InferenceError:
    """A failed stage and, for crop-level failures, a zero-based detector index."""

    stage: InferenceStage
    error_type: str
    message: str
    detection_index: int | None = None


@dataclass(frozen=True)
class PhotoInferenceResult:
    """A photograph's outcome and any successful predictions in detector order.

    Partial results contain both predictions and errors. The oriented size is
    (width, height) in the same pixel grid as the half-open XYXY boxes.
    """

    status: InferenceStatus
    oriented_size: tuple[int, int] | None = None
    predictions: tuple[AnimalPrediction, ...] = ()
    errors: tuple[InferenceError, ...] = ()

    def to_dict(self) -> dict:
        """Return a JSON-compatible representation of this result."""
        return {
            "status": self.status,
            "oriented_size": list(self.oriented_size) if self.oriented_size is not None else None,
            "predictions": [item.to_dict() for item in self.predictions],
            "errors": [asdict(error) for error in self.errors],
        }


class InferencePipeline:
    """Reuse one detector and one loaded classifier across photographs."""

    def __init__(self, detector: _Detector, classifier: _Classifier):
        self._detector = detector
        self._classifier = classifier

    def predict(self, image: ImageSource) -> PhotoInferenceResult:
        """Detect and classify each animal in a path or in-memory PIL photograph."""
        try:
            if not isinstance(image, (str, Path, Image.Image)):
                raise TypeError("Expected a filesystem path or PIL image")
            source = prepare_image(image)
            if source.width == 0 or source.height == 0:
                raise ValueError("Photograph must have positive width and height")
        except Exception as exc:
            location = f" {image!s}" if isinstance(image, (str, Path)) else ""
            error = InferenceError("input", type(exc).__name__, f"Could not read photograph{location}: {exc}")
            return PhotoInferenceResult("error", errors=(error,))

        try:
            detections = self._detector.detect(source)
            if not isinstance(detections, list):
                raise TypeError("Detector must return a list of Detection records")
        except Exception as exc:
            error = InferenceError("detector", type(exc).__name__, str(exc))
            return PhotoInferenceResult("error", source.size, errors=(error,))

        if not detections:
            return PhotoInferenceResult("no_detection", source.size)

        predictions = []
        errors = []
        for index, detection in enumerate(detections):
            if not isinstance(detection, Detection):
                errors.append(InferenceError("detector", "TypeError", "Expected a Detection record", index))
                continue
            try:
                if not is_valid_pixel_box(detection.box, source.size):
                    raise ValueError(f"Invalid detector box for a {source.width}×{source.height} image: {detection.box!r}")
                crop = crop_animal(source, detection.box)
                if crop is None or crop.width == 0 or crop.height == 0:
                    raise ValueError(f"Detector box did not yield a nonempty crop: {detection.box}")
            except Exception as exc:
                errors.append(InferenceError("crop", type(exc).__name__, str(exc), index))
                continue

            try:
                classification = self._classifier.predict(crop)
                if not isinstance(classification, ClassificationResult):
                    raise TypeError("Classifier must return a ClassificationResult")
            except Exception as exc:
                errors.append(InferenceError("classifier", type(exc).__name__, str(exc), index))
                continue
            predictions.append(AnimalPrediction(detection, classification))

        if not errors:
            status: InferenceStatus = "ok"
        else:
            status = "partial" if predictions else "error"
        return PhotoInferenceResult(status, source.size, tuple(predictions), tuple(errors))
