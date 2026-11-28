"""Prepare source pixels and crop bounded, half-open pixel boxes."""

from collections.abc import Sequence
from math import ceil, floor, isfinite
from pathlib import Path

from PIL import Image, ImageOps

ImageSource = str | Path | Image.Image
PixelBox = tuple[int, int, int, int]


def prepare_image(image: ImageSource) -> Image.Image:
    """Decode a photograph as RGB pixels in its EXIF-corrected orientation."""
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")
    with Image.open(image) as source:
        return prepare_image(source)


def _clip_box(
    box: Sequence[float], size: tuple[int, int], *, normalized: bool = False
) -> PixelBox | None:
    """Convert an XYXY box to a valid half-open integer box, or return None."""
    if not isinstance(box, Sequence):
        return None
    try:
        left, top, right, bottom = box
        if not all(
            not isinstance(value, bool) and isfinite(value)
            for value in (left, top, right, bottom)
        ):
            return None
    except (TypeError, ValueError, OverflowError):
        return None

    width, height = size
    x_limit, y_limit = (1, 1) if normalized else size
    left, top = max(0, left), max(0, top)
    right, bottom = min(x_limit, right), min(y_limit, bottom)
    if normalized:
        left, right = left * width, right * width
        top, bottom = top * height, bottom * height
    if left >= right or top >= bottom:
        return None
    return floor(left), floor(top), ceil(right), ceil(bottom)


def is_valid_pixel_box(box: object, size: tuple[int, int]) -> bool:
    """Check that an integer XYXY box is valid without clipping or rounding."""
    return (isinstance(box, tuple) and len(box) == 4
            and all(type(coordinate) is int for coordinate in box)
            and _clip_box(box, size) == box)


def crop_animal(image: ImageSource, box: Sequence[float]) -> Image.Image | None:
    """Crop an EXIF-oriented RGB image, returning None for an invalid box.

    The box is (left, top, right, bottom) in source pixels with exclusive
    right and bottom edges. The crop keeps the source resolution.
    """
    source = prepare_image(image)
    clipped_box = _clip_box(box, source.size)
    return source.crop(clipped_box) if clipped_box is not None else None
