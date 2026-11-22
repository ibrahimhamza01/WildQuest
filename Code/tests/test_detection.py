"""CPU-only tests of animal detection and cropping with a mocked engine."""

import csv
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
from wildquest.detection import Detection, MegaDetector, crop_animal

spec = importlib.util.spec_from_file_location("detector_cli", CODE / "scripts/test_detector.py")
detector_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detector_cli)


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


class DetectorCliTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.output = self.root / "outputs/detection/run"
        self.bundle = self.root / "models/MDV6-yolov10-e"
        self.bundle.mkdir(parents=True)
        (self.bundle / "model.onnx").write_bytes(b"mock weights")
        (self.bundle / "manifest.toml").write_text(
            '[model]\nid = "MDV6-yolov10-e"\nformat = "onnx"\nfile = "model.onnx"\n'
        )
        patcher = patch.object(detector_cli, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def args(self, *inputs):
        return ["--output-dir", str(self.output), "--model-bundle", str(self.bundle), *inputs]

    def write_manifest(self, rows):
        configs = self.root / "Code/configs"
        configs.mkdir(parents=True, exist_ok=True)
        manifest = configs / "dataset_manifest.csv"
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (configs / "dataset_manifest.sha256").write_text(detector_cli.sha256(manifest))

    def test_invalid_arguments_are_rejected(self):
        for inputs in (
            ("--split", "test"),
            ("--split", "dmv_test"),
            ("--split", "val", "--image", "photo.jpg"),
            ("--split", "val", "--threshold", "nan"),
            ("--split", "val", "--sample-per-class", "0"),
        ):
            with self.subTest(inputs=inputs), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    detector_cli.main(self.args(*inputs))
                self.assertEqual(failure.exception.code, 2)
        self.assertFalse(self.output.exists())

    def test_output_path_refuses_overwrites_and_escapes(self):
        self.output.mkdir(parents=True)
        marker = self.output / "existing.json"
        marker.write_text("keep me")
        with self.assertRaisesRegex(ValueError, "already exists"):
            detector_cli.main(self.args("--split", "val"))
        self.assertEqual(marker.read_text(), "keep me")
        with self.assertRaisesRegex(ValueError, "under outputs/detection"):
            detector_cli.output_directory(self.root / "elsewhere")
        escape = self.output.parent / "escape"
        escape.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "under outputs/detection"):
            detector_cli.output_directory(escape / "other")

    def test_held_out_paths_and_symlink_aliases_are_rejected(self):
        for split in ("test", "dmv_test"):
            held_out = self.root / "data/raw/images" / split / "00/photo.jpg"
            alias = self.root / f"{split}-alias.jpg"
            alias.symlink_to(held_out)
            for path in (held_out, alias):
                with self.subTest(path=path):
                    with self.assertRaisesRegex(ValueError, "train/val manifest"):
                        detector_cli.resolve_image(path, {})

    def test_sampling_is_reproducible_balanced_and_prefers_distinct_observations(self):
        rows = [
            {"split": split, "class_id": str(class_id), "photo_id": str(index),
             "observation_uuid": f"{class_id}-{observation}"}
            for split in ("train", "val", "test", "dmv_test")
            for class_id in (0, 1)
            for index, observation in enumerate(("one", "one", "two"))
        ]
        sample = detector_cli.sample_rows(rows, "val", 2, 42)
        self.assertEqual(sample, detector_cli.sample_rows(list(reversed(rows)), "val", 2, 42))
        self.assertEqual([row["class_id"] for row in sample], ["0", "0", "1", "1"])
        self.assertTrue(all(row["split"] == "val" for row in sample))
        self.assertEqual(len({row["observation_uuid"] for row in sample}), 4)

    def test_cli_records_outputs_empty_results_and_failures(self):
        row = {
            "local_path": "data/raw/images/val/00/first.png", "split": "val", "class_id": "0",
            "scientific_name": "Anas platyrhynchos", "group": "Bird", "photo_id": "1",
            "observation_uuid": "first-observation", "license": "CC0",
        }
        first = self.root / row["local_path"]
        first.parent.mkdir(parents=True)
        Image.new("RGB", (8, 6), "blue").save(first)
        self.write_manifest([{**row, "file_sha256": detector_cli.sha256(first)}])
        second = self.root / "blank.png"
        Image.new("RGB", (7, 5)).save(second)
        missing = self.root / "missing.png"
        engine = SimpleNamespace(init=Mock(), active_device=lambda: "cpu", __version__="test")
        detector = Mock()
        detector.return_value.detect.side_effect = [
            [Detection("animal", 0.8, (1, 2, 5, 6)), Detection("animal", 0.7, (2, 2, 6, 6))],
            [],
        ]
        with (
            patch.dict(sys.modules, {"sparrow_engine": engine}),
            patch.object(detector_cli, "MegaDetector", detector),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            status = detector_cli.main(self.args("--image", str(first), str(second), str(missing)))

        self.assertEqual(status, 1)
        engine.init.assert_called_once_with(device="cpu", model_dir=str(self.bundle.parent))
        detector.assert_called_once_with(confidence_threshold=0.20)
        report = json.loads((self.output / "results.json").read_text())
        self.assertEqual(report["manifest_sha256"],
                         detector_cli.sha256(self.root / "Code/configs/dataset_manifest.csv"))
        detected, blank, failed = report["images"]
        self.assertEqual(detected["group"], "Bird")
        self.assertEqual(detected["oriented_size"], [8, 6])
        self.assertEqual(detected["detection_count"], 2)
        self.assertAlmostEqual(detected["max_box_iou"], 0.6)
        for item in detected["detections"]:
            with Image.open(self.output / item["crop_path"]) as crop:
                self.assertEqual(crop.size, (4, 4))
        self.assertEqual(blank["detection_count"], 0)
        self.assertTrue((self.output / blank["preview_path"]).is_file())
        self.assertEqual(failed["error"]["stage"], "input")
        self.assertIsNone(failed["detection_count"])
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["overall"]["coverage_pct"], 50.0)
        self.assertEqual(summary["overall"]["processing_failures"], 1)
        self.assertEqual(summary["overall"]["multiple_detections"], 1)
        self.assertEqual(summary["by_group"]["Bird"]["coverage_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
