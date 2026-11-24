"""Image helpers for the UI: load uploads safely and draw detection boxes."""

import io

from PIL import Image, ImageDraw, ImageFont, ImageOps

BOX_COLORS = ("#E4572E", "#17BEBB", "#FFC914", "#76B041", "#8C4FFF", "#FF7AB6")


class ImageLoadError(ValueError):
    """The upload could not be read as an image."""


def load_image(data: bytes) -> Image.Image:
    """Open an uploaded photo, apply its EXIF rotation, and return it as RGB."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            return ImageOps.exif_transpose(im).convert("RGB")
    except Exception as exc:  # Pillow raises several exception types for bad files
        raise ImageLoadError("This file could not be read as an image.") from exc


def box_color(index: int) -> str:
    return BOX_COLORS[index % len(BOX_COLORS)]


def draw_boxes(image: Image.Image, animals) -> Image.Image:
    """Return a copy of the image with a numbered box for each detected animal."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    width, height = out.size
    line = max(2, round(min(width, height) / 150))
    font = ImageFont.load_default(size=max(12, round(min(width, height) / 20)))
    for i, animal in enumerate(animals):
        x0, y0, x1, y1 = animal.box
        rect = (x0 * width, y0 * height, x1 * width, y1 * height)
        color = box_color(i)
        draw.rectangle(rect, outline=color, width=line)
        label = f" {i + 1} "
        text_box = draw.textbbox((rect[0], rect[1]), label, font=font)
        draw.rectangle(text_box, fill=color)
        draw.text((rect[0], rect[1]), label, fill="black", font=font)
    return out