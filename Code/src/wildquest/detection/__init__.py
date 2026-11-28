"""Standalone detection and cropping for wildlife photographs."""

from .crops import crop_animal, prepare_image
from .megadetector import Detection, MegaDetector

__all__ = ["Detection", "MegaDetector", "crop_animal", "prepare_image"]
