"""Load the frozen EfficientNet-B0 classifier and predict species from animal crops."""

import hashlib
import json
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from wildquest.classifier.dataset import image_transforms
from wildquest.classifier.model import build_model

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
_CLASS_COUNT = 25
_CLASS_KEYS = tuple(str(class_id) for class_id in range(_CLASS_COUNT))


@dataclass(frozen=True)
class SpeciesPrediction:
    """A frozen class ID and scientific name with its softmax probability."""

    class_id: int
    scientific_name: str
    confidence: float


@dataclass(frozen=True)
class ClassificationResult:
    """One crop's best match and three descending predictions, including the best."""

    primary: SpeciesPrediction
    top3: tuple[SpeciesPrediction, ...]


def _resolve_device(device: str | torch.device) -> torch.device:
    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(
            f"Invalid classifier device {device!r}. Use 'cpu', 'cuda', or 'cuda:N'."
        ) from exc
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported classifier device {resolved}. Use CPU or CUDA.")
    if resolved.type == "cuda":
        if isinstance(device, str) and device.startswith("cuda:") and str(resolved) != device:
            raise ValueError(f"CUDA device index in {device!r} is out of range")
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {resolved} is unavailable. Use 'cpu' for inference.")
        if resolved.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        if not 0 <= resolved.index < torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {resolved} is unavailable. Its index is out of range.")
    return resolved


def _species_from_mapping(mapping: object, source: str) -> tuple[str, ...]:
    if not isinstance(mapping, dict) or mapping.keys() != {"id_to_label", "label_to_id"}:
        raise ValueError(f"{source} must contain id_to_label and label_to_id")

    id_to_label = mapping["id_to_label"]
    label_to_id = mapping["label_to_id"]
    if not isinstance(id_to_label, dict) or id_to_label.keys() != set(_CLASS_KEYS):
        raise ValueError(f"{source} must define contiguous class IDs 0 through {_CLASS_COUNT - 1}")
    names = tuple(id_to_label[key] for key in _CLASS_KEYS)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{source} must define a scientific name for every class ID")
    expected_reverse = {name: class_id for class_id, name in enumerate(names)}
    if (len(expected_reverse) != _CLASS_COUNT or not isinstance(label_to_id, dict)
            or any(type(class_id) is not int for class_id in label_to_id.values())
            or label_to_id != expected_reverse):
        raise ValueError(f"{source} must map unique scientific names back to their class IDs")
    return names


def _load_checkpoint(path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Classifier checkpoint not found: {path}") from exc
    except (OSError, EOFError, pickle.UnpicklingError, RuntimeError, ValueError, IndexError) as exc:
        raise ValueError(f"Cannot load classifier checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Classifier checkpoint {path} must contain a metadata dictionary")

    for field in ("class_mapping", "manifest_sha256", "model_state_dict"):
        if field not in checkpoint:
            raise ValueError(f"Classifier checkpoint {path} is missing {field}")
    return checkpoint


class EfficientNetClassifier:
    """Load a local training checkpoint once and classify in-memory PIL crops."""

    def __init__(self, checkpoint_path: str | Path, *, device: str | torch.device = "cpu"):
        self._device = _resolve_device(device)
        path = Path(checkpoint_path)
        checkpoint = _load_checkpoint(path)

        mapping_path = _CONFIG_DIR / "class_mapping.json"
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        self._species = _species_from_mapping(mapping, f"Repository class mapping {mapping_path}")
        checkpoint_species = _species_from_mapping(
            checkpoint["class_mapping"], f"Checkpoint class mapping ({path})"
        )
        if checkpoint_species != self._species:
            raise ValueError(f"Checkpoint class mapping differs from repository mapping: {path}")

        manifest = _CONFIG_DIR / "dataset_manifest.csv"
        actual_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        checkpoint_sha = checkpoint["manifest_sha256"]
        if not isinstance(checkpoint_sha, str) or checkpoint_sha != actual_sha:
            raise ValueError(f"Checkpoint manifest hash differs from frozen local manifest: {path}")

        state_dict = checkpoint["model_state_dict"]
        if not isinstance(state_dict, Mapping) or any(not isinstance(key, str) for key in state_dict):
            raise ValueError(
                f"Classifier checkpoint {path} must contain a model state dictionary with string keys"
            )
        model = build_model(num_classes=_CLASS_COUNT, pretrained=False)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"Incompatible EfficientNet-B0 model state in {path}: {exc}") from exc
        self._model = model.to(self._device).eval()
        self._transform = image_transforms("val")

    def predict(self, crop: Image.Image) -> ClassificationResult:
        """Classify one already-extracted crop with the training validation transform."""
        if not isinstance(crop, Image.Image):
            raise TypeError("Classifier crop must be a PIL image")
        if crop.width == 0 or crop.height == 0:
            raise ValueError("Classifier crop must have nonzero width and height")
        if crop.mode != "RGB":
            crop = crop.convert("RGB")

        image = self._transform(crop).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            logits = self._model(image)
            if (not isinstance(logits, torch.Tensor) or logits.shape != (1, _CLASS_COUNT)
                    or not logits.is_floating_point()):
                raise RuntimeError(
                    f"Classifier must return floating-point logits of shape (1, {_CLASS_COUNT})"
                )
            if not torch.isfinite(logits).all():
                raise RuntimeError("Classifier returned non-finite logits")
            probabilities = logits[0].softmax(dim=0)
            confidences, class_ids = probabilities.topk(3, sorted=True)
            top3 = tuple(
                SpeciesPrediction(class_id, self._species[class_id], confidence)
                for class_id, confidence in zip(class_ids.tolist(), confidences.tolist())
            )
        return ClassificationResult(primary=top3[0], top3=top3)
