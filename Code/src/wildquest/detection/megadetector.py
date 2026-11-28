"""Adapt SPARROW-Engine MegaDetector predictions to WildQuest boxes."""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from .crops import ImageSource, PixelBox, _clip_box, prepare_image

MODEL_ID = "MDV6-yolov10-e"
DEFAULT_CONFIDENCE_THRESHOLD = 0.20


def read_model_bundle(bundle: Path) -> tuple[Path, dict, Path]:
    """Return the manifest path, model table, and local weights path."""
    if bundle.name != MODEL_ID:
        raise ValueError(f"--model-bundle must point to the unpacked {MODEL_ID} ONNX directory")
    manifest = bundle / "manifest.toml"
    with manifest.open("rb") as handle:
        model = tomllib.load(handle).get("model", {})
    if not isinstance(model, dict) or model.get("id") != MODEL_ID or model.get("format") != "onnx":
        raise ValueError(f"--model-bundle must point to the unpacked {MODEL_ID} ONNX directory")
    filename = model.get("file")
    if not isinstance(filename, str) or not filename:
        raise ValueError("Model manifest is missing its ONNX file path")
    onnx_file = (bundle / filename).resolve()
    if not onnx_file.is_relative_to(bundle) or onnx_file.suffix != ".onnx":
        raise ValueError("Model manifest must point to an ONNX file inside the bundle")
    if not onnx_file.is_file():
        raise FileNotFoundError(f"Detector ONNX weights not found: {onnx_file}")
    return manifest, model, onnx_file


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

    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.confidence_threshold = confidence_threshold

    def detect(self, image: ImageSource) -> list[Detection]:
        """Return every valid animal detection at or above the confidence threshold."""
        source = prepare_image(image)
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
