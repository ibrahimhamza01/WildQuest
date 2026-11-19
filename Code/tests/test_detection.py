"""CPU-only tests of animal detection and cropping with a mocked engine."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wildquest.detection import Detection, MegaDetector, crop_animal


def prediction(box, confidence=0.9, label="animal"):
    return SimpleNamespace(
        label=label,
        confidence=confidence,
        bbox=SimpleNamespace(**dict(zip(("x_min", "y_min", "x_max", "y_max"), box))),
    )


class CropTests(unittest.TestCase):
    def test_crop_keeps_source_pixels_and_excludes_right_and_bottom(self):
        image = Image.new("RGB", (5, 4))
        for y in range(image.height):
            for x in range(image.width):
                image.putpixel((x, y), (x, y, 0))

        crop = crop_animal(image, (-1.0, 1.25, 2.25, 3))

        self.assertEqual(crop.size, (3, 2))
        for y in range(crop.height):
            for x in range(crop.width):
                self.assertEqual(crop.getpixel((x, y)), (x, y + 1, 0))

    def test_invalid_boxes_do_not_produce_crops(self):
        image = Image.new("RGB", (5, 4))
        invalid_boxes = (
            None,
            (0, 0, 1),
            (0, 0, 1, 1, 2),
            {0: 0, 1: 0, 2: 1, 3: 1},
            ("0", 0, 1, 1),
            (False, 0, 1, 1),
            (0, 0, float("nan"), 1),
            (0, 0, 1, float("inf")),
            (2.1, 0, 2.1, 2),
            (2, 0, 1, 1),
            (0, 2, 1, 1),
            (5, 0, 8, 1),
            (0, -2, 1, -1),
        )
        for box in invalid_boxes:
            with self.subTest(box=box):
                self.assertIsNone(crop_animal(image, box))


class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (10, 6))
        self.backend = Mock()
        engine = SimpleNamespace(detect=self.backend)
        patcher = patch.dict(sys.modules, {"sparrow_engine": engine})
        patcher.start()
        self.addCleanup(patcher.stop)

    def set_predictions(self, *predictions):
        self.backend.return_value = [
            SimpleNamespace(image_size=self.image.size, detections=list(predictions))
        ]

    def test_normalized_boxes_convert_and_clip_all_animals(self):
        self.set_predictions(
            prediction((0.125, 0.25, 0.625, 0.75), confidence=0.8),
            prediction((-0.25, 0.5, 1.2, 1.1), confidence=0.9),
        )

        detections = MegaDetector().detect(self.image)

        self.assertEqual(
            detections,
            [
                Detection("animal", 0.8, (1, 1, 7, 5)),
                Detection("animal", 0.9, (0, 3, 10, 6)),
            ],
        )
        self.assertEqual(
            self.backend.call_args.kwargs, {"model": "MDV6-yolov10-e", "threshold": 0.20}
        )

    def test_only_animals_at_or_above_threshold_are_retained(self):
        box = (0, 0, 1, 1)
        self.set_predictions(
            prediction(box, label="person"),
            prediction(box, label="car"),
            prediction(box, confidence=0.349),
            prediction(box, confidence=float("nan")),
            prediction(box, confidence=0.35),
        )

        detections = MegaDetector(confidence_threshold=0.35).detect(self.image)

        self.assertEqual(detections, [Detection("animal", 0.35, (0, 0, 10, 6))])
        self.assertEqual(self.backend.call_args.kwargs["threshold"], 0.35)

    def test_malformed_and_empty_boxes_are_skipped(self):
        self.set_predictions(
            prediction((0, 0, 1)),
            prediction((0, 0, float("inf"), 1)),
            prediction((0, 0, "1", 1)),
            prediction((0.55, 0, 0.55, 1)),
            prediction((0.57, 0, 0.55, 1)),
            prediction((1.1, 0, 1.5, 1)),
            prediction((0, 0, 1, 1)),
        )

        self.assertEqual(
            MegaDetector().detect(self.image), [Detection("animal", 0.9, (0, 0, 10, 6))]
        )

    def test_empty_detections_return_an_empty_list(self):
        self.set_predictions()
        self.assertEqual(MegaDetector().detect(self.image), [])

    def test_exif_orientation_matches_for_path_and_in_memory_inputs(self):
        image = Image.new("RGB", (3, 2))
        image.putpixel((0, 0), (255, 0, 0))
        image.getexif()[274] = 6  # Rotate 90 degrees clockwise for display.
        prepared_paths = []

        def detect(path, **kwargs):
            prepared_paths.append(path)
            with Image.open(path) as prepared:
                self.assertEqual(prepared.format, "PNG")
                self.assertEqual(prepared.mode, "RGB")
                self.assertEqual(prepared.size, (2, 3))
                self.assertEqual(prepared.getpixel((1, 0)), (255, 0, 0))
                self.assertIsNone(prepared.getexif().get(274))
            return [
                SimpleNamespace(
                    image_size=(2, 3), detections=[prediction((0.5, 0, 1, 0.3))]
                )
            ]

        self.backend.side_effect = detect
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.png"
            image.save(path, exif=image.getexif())
            for source in (path, str(path), image):
                with self.subTest(input_type=type(source)):
                    detections = MegaDetector().detect(source)
                    self.assertEqual(detections, [Detection("animal", 0.9, (1, 0, 2, 1))])
                    crop = crop_animal(source, detections[0].box)
                    self.assertEqual(crop.size, (1, 1))
                    self.assertEqual(crop.getpixel((0, 0)), (255, 0, 0))
                    self.assertFalse(prepared_paths[-1].parent.exists())
            self.assertTrue(path.is_file())
        self.assertEqual(image.getexif()[274], 6)
        self.assertEqual(image.size, (3, 2))

    def test_backend_error_propagates_and_temporary_image_is_removed(self):
        failure = RuntimeError("inference failed")

        def detect(path, **kwargs):
            self.assertTrue(path.is_file())
            raise failure

        self.backend.side_effect = detect
        with self.assertRaises(RuntimeError) as raised:
            MegaDetector().detect(self.image)
        self.assertIs(raised.exception, failure)
        self.assertFalse(self.backend.call_args.args[0].parent.exists())

    def test_missing_image_result_is_an_error(self):
        self.backend.return_value = []
        with self.assertRaisesRegex(RuntimeError, "exactly one image result"):
            MegaDetector().detect(self.image)

    def test_mismatched_image_dimensions_are_an_error(self):
        self.backend.return_value = [SimpleNamespace(image_size=(6, 10), detections=[])]
        with self.assertRaisesRegex(RuntimeError, "unexpected image dimensions"):
            MegaDetector().detect(self.image)

    def test_threshold_must_be_a_probability(self):
        for threshold in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    MegaDetector(confidence_threshold=threshold)


if __name__ == "__main__":
    unittest.main()
