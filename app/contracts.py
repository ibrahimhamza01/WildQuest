"""The response format the UI expects from an inference provider, and its parser.

A provider returns a plain dict (JSON-compatible) so a real backend (issue #11)
can replace the mock without changing the UI:

    {
      "status": "ok" | "no_animal" | "error",
      "message": "optional text, required for errors",
      "is_mock": true,
      "detections": [
        {
          "box": [x_min, y_min, x_max, y_max],      # normalized to 0..1
          "detector_confidence": 0.97,               # MegaDetector score
          "predictions": [                           # classifier output, any order
            {"class_id": 17, "confidence": 0.94},
            {"class_id": 9, "confidence": 0.03}
          ]
        }
      ]
    }

`parse_prediction` validates that dict and turns it into objects the UI renders.
"""

from dataclasses import dataclass
from typing import Optional

from .species import SPECIES

# Matches the provisional review threshold in Code/configs/gamification.json (#12).
REVIEW_THRESHOLD = 0.70
TOP_K = 3
STATUSES = ("ok", "no_animal", "error")


class ResponseError(ValueError):
    """The provider returned something the UI cannot display."""


@dataclass(frozen=True)
class Guess:
    class_id: int
    common_name: str
    scientific_name: str
    confidence: float


@dataclass(frozen=True)
class AnimalResult:
    box: tuple            # (x_min, y_min, x_max, y_max), normalized
    detector_confidence: float
    top: tuple            # up to TOP_K Guess objects, highest confidence first

    @property
    def best(self) -> Guess:
        return self.top[0]

    @property
    def needs_review(self) -> bool:
        return self.best.confidence < REVIEW_THRESHOLD


@dataclass(frozen=True)
class PredictionResult:
    status: str           # "ok", "no_animal" or "error"
    animals: tuple
    message: Optional[str]
    is_mock: bool


def _number(value, name, low=0.0, high=1.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResponseError(f"{name} must be a number, got {value!r}")
    if not low <= value <= high:
        raise ResponseError(f"{name} must be between {low} and {high}, got {value!r}")
    return float(value)


def _guess(raw, where):
    if not isinstance(raw, dict):
        raise ResponseError(f"{where} must be an object")
    class_id = raw.get("class_id")
    if isinstance(class_id, bool) or not isinstance(class_id, int) \
            or not 0 <= class_id < len(SPECIES):
        raise ResponseError(f"{where}.class_id must be an integer 0-{len(SPECIES) - 1}")
    species = SPECIES[class_id]
    confidence = _number(raw.get("confidence"), f"{where}.confidence")
    return Guess(class_id, species.common_name, species.scientific_name, confidence)


def _box(raw, where):
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ResponseError(f"{where}.box must be [x_min, y_min, x_max, y_max]")
    x0, y0, x1, y1 = (_number(v, f"{where}.box") for v in raw)
    if x1 <= x0 or y1 <= y0:
        raise ResponseError(f"{where}.box must have x_max > x_min and y_max > y_min")
    return (x0, y0, x1, y1)


def _animal(raw, index):
    where = f"detections[{index}]"
    if not isinstance(raw, dict):
        raise ResponseError(f"{where} must be an object")
    predictions = raw.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ResponseError(f"{where}.predictions must be a non-empty list")
    guesses = [_guess(p, f"{where}.predictions[{i}]") for i, p in enumerate(predictions)]
    guesses.sort(key=lambda g: g.confidence, reverse=True)
    return AnimalResult(
        box=_box(raw.get("box"), where),
        detector_confidence=_number(raw.get("detector_confidence"),
                                    f"{where}.detector_confidence"),
        top=tuple(guesses[:TOP_K]),
    )


def parse_prediction(response) -> PredictionResult:
    """Validate a provider response. Raises ResponseError if it is malformed."""
    if not isinstance(response, dict):
        raise ResponseError("Response must be an object")
    status = response.get("status")
    if status not in STATUSES:
        raise ResponseError(f"status must be one of {STATUSES}, got {status!r}")
    message = response.get("message")
    is_mock = bool(response.get("is_mock", False))

    if status == "error":
        if not isinstance(message, str) or not message.strip():
            raise ResponseError("An error response needs a message")
        return PredictionResult("error", (), message, is_mock)

    detections = response.get("detections", [])
    if not isinstance(detections, list):
        raise ResponseError("detections must be a list")
    animals = tuple(_animal(d, i) for i, d in enumerate(detections))
    if status == "no_animal" and animals:
        raise ResponseError("status is no_animal but detections were returned")
    if not animals:
        status = "no_animal"
    return PredictionResult(status, animals, message, is_mock)