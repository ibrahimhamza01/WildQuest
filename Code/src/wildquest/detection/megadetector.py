"""Adapt SPARROW-Engine MegaDetector predictions to WildQuest boxes."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from .crops import ImageSource, PixelBox, _clip_box, _prepare_image

MODEL_ID = "MDV6-yolov10-e"


@dataclass(frozen=True)
class Detection:
    """An animal confidence score and half-open XYXY box in oriented source pixels."""

    category: str
    confidence: float
    box: PixelBox


def _sparrow_predictions(image: Image.Image, threshold: float):
    import sparrow_engine

    # Use the same oriented pixels for both input types and for cropping.
    with TemporaryDirectory(prefix="wildquest-detection-") as directory:
        path = Path(directory) / "image.png"
        image.save(path, format="PNG")
        results = sparrow_engine.detect(path, model=MODEL_ID, threshold=threshold)

    if len(results) != 1:
        raise RuntimeError("SPARROW-Engine did not return exactly one image result")
    result = results[0]
    if result.image_size != image.size:
        raise RuntimeError("SPARROW-Engine reported unexpected image dimensions")
    return result.detections


class MegaDetector:
    """Detect animals using a locally provisioned SPARROW-Engine model.

    Set SPARROW_ENGINE_MODEL_DIR before the first inference to choose the model
    directory. SPARROW-Engine initializes and loads the model on demand.
    """

    def __init__(self, confidence_threshold: float = 0.20):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.confidence_threshold = confidence_threshold

    def detect(self, image: ImageSource) -> list[Detection]:
        """Return every valid animal detection at or above the confidence threshold."""
        source = _prepare_image(image)
        predictions = _sparrow_predictions(source, self.confidence_threshold)
        detections = []
        for prediction in predictions:
            if prediction.label != "animal":
                continue
            confidence = prediction.confidence
            if not self.confidence_threshold <= confidence <= 1.0:
                continue
            try:
                bbox = prediction.bbox
                coordinates = (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max)
            except AttributeError:
                continue
            box = _clip_box(coordinates, source.size, normalized=True)
            if box is not None:
                detections.append(Detection("animal", confidence, box))
        return detections
